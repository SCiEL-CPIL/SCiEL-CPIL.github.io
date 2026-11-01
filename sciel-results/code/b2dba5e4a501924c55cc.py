import time
import numpy as np
from scipy.linalg import lstsq

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    best_rank = 64
    best_factors = _schoolbook_factors(4)

    def _elapsed():
        return time.time() - start_time

    def _update_best(factors, rank):
        nonlocal best_rank, best_factors
        if rank < best_rank:
            best_rank = rank
            best_factors = factors

    def _verify_factors(U, V, W):
        n, m, p = 4, 4, 4
        rank = U.shape[1]
        U4 = np.asarray(U, dtype=float).reshape(n, m, rank)
        V4 = np.asarray(V, dtype=float).reshape(m, p, rank)
        W4 = np.asarray(W, dtype=float).reshape(p, n, rank)
        recon = np.einsum('ijr,jkr,kir->ijk', U4, V4, W4)
        target = np.ones((n, m, p), dtype=float)
        return np.allclose(recon, target, atol=1e-9)

    def _try_candidate(factors, rank):
        if rank >= best_rank:
            return
        U, V, W = factors
        if _verify_factors(U, V, W):
            _update_best(factors, rank)

    # 1. Try rank 49 (Kronecker of Strassen 2x2)
    if _elapsed() < deadline - 5.0:
        try:
            factors49 = _strassen_kron_49(4)
            _try_candidate(factors49, 49)
        except Exception:
            pass

    # 2. Try random rank-44 search
    if _elapsed() < deadline - 5.0 and best_rank > 44:
        try:
            factors44 = _random_search(4, 44, max_iters=500, deadline=deadline)
            if factors44 is not None:
                _try_candidate(factors44, 44)
        except Exception:
            pass

    # 3. Try random rank-48 search
    if _elapsed() < deadline - 5.0 and best_rank > 48:
        try:
            factors48 = _random_search(4, 48, max_iters=1000, deadline=deadline)
            if factors48 is not None:
                _try_candidate(factors48, 48)
        except Exception:
            pass

    return {
        "U": np.asarray(best_factors[0]).tolist(),
        "V": np.asarray(best_factors[1]).tolist(),
        "W": np.asarray(best_factors[2]).tolist()
    }


def _schoolbook_factors(n):
    rank = n * n * n
    U = np.zeros((n * n, rank), dtype=int)
    V = np.zeros((n * n, rank), dtype=int)
    W = np.zeros((n * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1
                V[j * n + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return (U, V, W)


def _strassen_kron_49(n):
    """Build rank-49 decomposition using Kronecker product of Strassen 2x2."""
    # Correct Strassen 2x2 factors
    U2 = np.array([
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ], dtype=int)
    
    # Verify 2x2
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    recon2 = np.einsum('ijr,jkr,kir->ijk', U2_3d, V2_3d, W2_3d)
    if not np.allclose(recon2, np.ones((2, 2, 2)), atol=1e-9):
        raise ValueError("2x2 Strassen factors are incorrect")
    
    # Build 4x4 via Kronecker
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s1 in range(7):
        for s2 in range(7):
            col = s1 * 7 + s2
            # U: (i1,j1,i2,j2) -> i*4+j
            U4[:, col] = np.kron(U2[:, s1], U2[:, s2])
            # V: (j1,k1,j2,k2) -> j*4+k
            V4[:, col] = np.kron(V2[:, s1], V2[:, s2])
            # W: (i1,k1,i2,k2) -> k*4+i
            W4[:, col] = np.kron(W2[:, s1], W2[:, s2])
    
    return (U4, V4, W4)


def _random_search(n, rank, max_iters=1000, deadline=None):
    """Random search for rank decomposition."""
    if deadline is None:
        deadline = time.time() + 10.0
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        U = np.random.randint(-2, 3, (n * n, rank), dtype=int)
        V = np.random.randint(-2, 3, (n * n, rank), dtype=int)
        
        # Solve for W
        U4 = U.reshape(n, n, rank)
        V4 = V.reshape(n, n, rank)
        
        # Build linear system for W
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is linear in W for fixed U, V
        A_mat = np.zeros((n * n * n, n * n), dtype=float)
        b_vec = np.ones(n * n * n, dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for r in range(rank):
                        A_mat[idx, k * n + i] += U[i, j, r] * V[j, k, r]
                    idx += 1
        
        try:
            W_flat, residuals, rank_svd, sv = lstsq(A_mat, b_vec, lapack_driver='gelsy')
        except Exception:
            continue
        
        W = W_flat.reshape(n * n)
        W_round = np.round(W)
        
        # Check if rounded solution works
        U_check = U.astype(float)
        V_check = V.astype(float)
        W_check = W_round.astype(float).reshape(n, n, rank)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U_check.reshape(n, n, rank), 
                          V_check.reshape(n, n, rank), W_check)
        
        if np.allclose(recon, np.ones((n, n, n)), atol=1e-9):
            return (U, V, W_round.astype(int))
        
        # Also check without rounding
        W_exact = W_flat.reshape(n, n)
        W_check2 = W_exact.reshape(n, n, rank)
        recon2 = np.einsum('ijr,jkr,kir->ijk', U_check.reshape(n, n, rank), 
                           V_check.reshape(n, n, rank), W_check2)
        if np.allclose(recon2, np.ones((n, n, n)), atol=1e-9):
            # Keep as float for now, but task requires integer/half-integer
            W_half = np.round(W_exact * 2) / 2
            if np.allclose(W_half, W_exact, atol=1e-9):
                return (U, V, W_half)
    
    return None


# EVOLVE-BLOCK-END
