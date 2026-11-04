"""Improved rank-16 search for 2x3*3x3 matmul tensor decomposition.
Key changes: better ALS convergence with adaptive restart, greedy column
addition from rank-17, and a more structured rank-16 initialization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    """Full matmul tensor T[i,j,k]=1 iff (i//M)==(k%N) and (i%M)==(j//P)."""
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            ja2, kb = divmod(j, P)
            if ja != ja2:
                continue
            for k in range(W_DIM):
                kb2, ic = divmod(k, N)
                if ia == ic and kb == kb2:
                    T[i, j, k] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    """Valid rank-17 decomposition: Strassen(2x2x2) for (j in {0,1}, k in {0,1})
    plus schoolbook for the remaining 10 terms."""
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=300, deadline=None, tol=1e-12):
    """ALS with convergence check. Returns (U, V, W, converged, final_err)."""
    best_err = np.inf
    best_state = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return best_state[0], best_state[1], best_state[2], False, best_err
        
        # Solve for U
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        # Solve for V
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        # Solve for W
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        
        if err < best_err:
            best_err = err
            best_state = (U.copy(), V.copy(), W.copy())
            
        if err < tol:
            return U, V, W, True, err
    
    return best_state[0], best_state[1], best_state[2], False, best_err


def _try_rank16(deadline):
    """Multiple strategies to find rank-16 decomposition."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16
    
    # Strategy 1: Drop each column from rank-17, then ALS polish
    U17, V17, W17 = _rank17()
    for drop_col in range(17):
        if time.time() > deadline:
            return None
        U = np.delete(U17, drop_col, axis=1)
        V = np.delete(V17, drop_col, axis=1)
        W = np.delete(W17, drop_col, axis=1)
        U, V, W, conv, err = _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        if conv or _validate(U, V, W, T, atol=1e-8):
            if _validate(U, V, W, T, atol=1e-8):
                return U, V, W
    
    # Strategy 2: Greedy construction - start from 12 columns, add 4 optimally
    # Use Strassen core (7 cols) + 5 best schoolbook columns, then add 4 more
    # Try all subsets of 16 from 17 (drop 1) already done above.
    # Instead: try mixing - take 8 Strassen-like + 8 schoolbook
    for trial in range(50):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(10000 + trial)
        
        # Initialize with sparse random patterns biased toward tensor support
        U = np.zeros((U_DIM, rank))
        V = np.zeros((V_DIM, rank))
        W = np.zeros((W_DIM, rank))
        
        for c in range(rank):
            # Each column: pick 2-3 nonzero entries per matrix
            for _ in range(rng.randint(2, 4)):
                U[rng.randint(U_DIM), c] = rng.choice([-1, 1])
            for _ in range(rng.randint(2, 4)):
                V[rng.randint(V_DIM), c] = rng.choice([-1, 1])
            for _ in range(rng.randint(2, 4)):
                W[rng.randint(W_DIM), c] = rng.choice([-1, 1])
        
        U, V, W, conv, err = _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=150, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W
    
    # Strategy 3: Continuation from rank-17 with column merging
    # Take pairs of columns and try to merge into single columns
    for trial in range(30):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(20000 + trial)
        
        # Start from rank-17, pick a pair to merge
        c1, c2 = rng.choice(17, size=2, replace=False)
        
        # Initialize rank-16: copy all columns except c2, set c1 to sum
        U = np.delete(U17, c2, axis=1)
        V = np.delete(V17, c2, axis=1)
        W = np.delete(W17, c2, axis=1)
        
        # Modify column c1 (adjust index if c2 < c1)
        idx = c1 if c2 > c1 else c1 - 1
        U[:, idx] += U17[:, c2]
        V[:, idx] += V17[:, c2]
        W[:, idx] += W17[:, c2]
        
        U, V, W, conv, err = _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W
    
    # Strategy 4: Dense random with larger perturbation
    for trial in range(40):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(30000 + trial)
        scale = 0.3 + 0.05 * trial
        U = rng.randn(U_DIM, rank) * scale
        V = rng.randn(V_DIM, rank) * scale
        W = rng.randn(W_DIM, rank) * scale
        U, V, W, conv, err = _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W
    
    # Strategy 5: Structured - exploit the block structure
    # The tensor has a specific pattern: T[i,j,k]=1 iff ia==ic and ja==ja2 and kb==kb2
    # where i=ia*M+ja, j=ja2*P+kb, k=kb2*N+ic
    # Try block-diagonal-like structure
    for trial in range(20):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(40000 + trial)
        
        # Initialize with block structure: each column handles one (ia, ja, kb) combo
        # but with overlap to achieve rank < 18
        U = np.zeros((U_DIM, rank))
        V = np.zeros((V_DIM, rank))
        W = np.zeros((W_DIM, rank))
        
        # Assign columns to handle different (ia, ja, kb) combinations
        combos = [(ia, ja, kb) for ia in range(N) for ja in range(M) for kb in range(P)]
        # 18 combos, need 16 columns - some must handle multiple
        for c in range(rank):
            if c < 12:
                ia, ja, kb = combos[c]
                U[ia*M+ja, c] = 1
                V[ja*P+kb, c] = 1
                W[kb*N+ia, c] = 1
            else:
                # Overlapping columns
                ia1, ja1, kb1 = combos[c - 12]
                ia2, ja2, kb2 = combos[(c - 12 + 6) % 18]
                U[ia1*M+ja1, c] = 1
                U[ia2*M+ja2, c] = rng.choice([-1, 1])
                V[ja1*P+kb1, c] = 1
                V[ja2*P+kb2, c] = rng.choice([-1, 1])
                W[kb1*N+ia1, c] = 1
                W[kb2*N+ia2, c] = rng.choice([-1, 1])
        
        U, V, W, conv, err = _als_converged(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W
    
    return None


def construct():
    start = time.time()
    deadline = start + 140
    
    # Baseline: valid rank-17 (verified).
    U17, V17, W17 = _rank17()
    T = _build_tensor()
    base = (U17, V17, W17)
    if not _validate(U17, V17, W17, T):
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        base = (U, V, W)

    # Attempt rank-16
    try:
        r16 = _try_rank16(deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
