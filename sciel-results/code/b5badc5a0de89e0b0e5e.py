"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy: Use Strassen's algorithm on the two 2x2x2 sub-problems (columns 0-1 and 2-3 of B),
which reduces each from rank 8 to rank 7, giving 14 terms. The remaining 2x2x1 sub-problem
(columns 4) requires rank 4 (schoolbook is optimal for 2x2x1). Total: 18.

We then attempt to find a rank-17 decomposition by sharing one Strassen term between
the two 2x2 blocks, using structured search and ALS with exact verification.

If no lower rank is found, we return the rank-18 Strassen+schoolbook decomposition.
"""

import numpy as np
import time

DIMENSIONS = [2, 2, 5]


def _build_target():
    """Build the matrix multiplication tensor T[u,v,w] for C = A*B.
    u = i*m + j (A index), v = j*p + k (B index), w = k*n + i (C index).
    T[u,v,w] = 1 if the (i,j,k) triple contributes to C[i,k].
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=tol)


def _build_strassen_block(U_rows, V_rows, W_rows):
    """
    Build Strassen's 7-term decomposition for 2x2*2x2.
    Returns (U, V, W) each of shape (4, 7) for the local indices.
    Local u: (i,j) with i,j in {0,1} -> u = i*2+j
    Local v: (j,k) with j,k in {0,1} -> v = j*2+k
    Local w: (k,i) with k,i in {0,1} -> w = k*2+i
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B11)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # u indices: u0=(0,0), u1=(0,1), u2=(1,0), u3=(1,1)
    # v indices: v0=(0,0), v1=(0,1), v2=(1,0), v3=(1,1)
    # w indices: w0=(0,0), w1=(0,1), w2=(1,0), w3=(1,1)
    
    # M1: (A00+A11)*(B00+B11) -> contributes to C00 (+), C11 (-)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2: (A10+A11)*B00 -> contributes to C10 (+), C11 (-)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00*(B10-B11) -> contributes to C01 (+), C11 (+)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11*(B01-B11) -> contributes to C00 (-), C01 (+)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5: (A00+A01)*B11 -> contributes to C00 (+), C10 (+)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A10-A00)*(B00+B10) -> contributes to C11 (+)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)*(B01+B11) -> contributes to C00 (+)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_rank18():
    """
    Build the rank-18 decomposition:
    - Block 1: Strassen for columns 0,1 of B (2x2x2) -> 7 terms
    - Block 2: Strassen for columns 2,3 of B (2x2x2) -> 7 terms
    - Block 3: Schoolbook for column 4 of B (2x2x1) -> 4 terms
    Total: 18 terms
    """
    n, m, p = DIMENSIONS
    U_rows = n * m  # 4
    V_rows = m * p  # 10
    W_rows = p * n  # 10
    
    R = 18
    U_total = np.zeros((U_rows, R))
    V_total = np.zeros((V_rows, R))
    W_total = np.zeros((W_rows, R))
    
    # Build Strassen block (local 4x4x4)
    S_U, S_V, S_W = _build_strassen_block(4, 4, 4)
    
    # Block 1: columns k=0,1
    # v indices: j=0 -> v=0,1; j=1 -> v=2,3
    # w indices: k=0 -> w=0,1; k=1 -> w=2,3
    U_total[:, 0:7] = S_U
    V_total[0, 0:7] = S_V[0]  # (0,0)
    V_total[1, 0:7] = S_V[1]  # (0,1)
    V_total[2, 0:7] = S_V[2]  # (1,0)
    V_total[3, 0:7] = S_V[3]  # (1,1)
    W_total[0, 0:7] = S_W[0]  # (0,0)
    W_total[1, 0:7] = S_W[1]  # (0,1)
    W_total[2, 0:7] = S_W[2]  # (1,0)
    W_total[3, 0:7] = S_W[3]  # (1,1)
    
    # Block 2: columns k=2,3
    # v indices: j=0 -> v=4,5; j=1 -> v=6,7
    # w indices: k=2 -> w=4,5; k=3 -> w=6,7
    U_total[:, 7:14] = S_U
    V_total[4, 7:14] = S_V[0]
    V_total[5, 7:14] = S_V[1]
    V_total[6, 7:14] = S_V[2]
    V_total[7, 7:14] = S_V[3]
    W_total[4, 7:14] = S_W[0]
    W_total[5, 7:14] = S_W[1]
    W_total[6, 7:14] = S_W[2]
    W_total[7, 7:14] = S_W[3]
    
    # Block 3: column k=4 (schoolbook 2x2x1)
    # v indices: j=0 -> v=8; j=1 -> v=9
    # w indices: k=4 -> w=8,9
    U_total[:, 14:18] = 0
    # (i=0,j=0): u=0, v=8, w=8
    U_total[0, 14] = 1
    V_total[8, 14] = 1
    W_total[8, 14] = 1
    # (i=0,j=1): u=1, v=9, w=8
    U_total[1, 15] = 1
    V_total[9, 15] = 1
    W_total[8, 15] = 1
    # (i=1,j=0): u=2, v=8, w=9
    U_total[2, 16] = 1
    V_total[8, 16] = 1
    W_total[9, 16] = 1
    # (i=1,j=1): u=3, v=9, w=9
    U_total[3, 17] = 1
    V_total[9, 17] = 1
    W_total[9, 17] = 1
    
    return U_total, V_total, W_total


def _als_update(U, V, W, T, R, max_iters=50, tol=1e-10):
    """
    ALS (Alternating Least Squares) for tensor decomposition.
    U: (U_rows, R), V: (V_rows, R), W: (W_rows, R)
    T: (U_rows, V_rows, W_rows)
    """
    U_rows = U.shape[0]
    V_rows = V.shape[0]
    W_rows = W.shape[0]
    
    for it in range(max_iters):
        # Update U: for each u, solve (V*W) @ U[:,r] = T[u]
        VW = np.einsum('vr,wr->vwr', V, W).reshape(V_rows * W_rows, R)
        U_new = np.zeros_like(U)
        for u in range(U_rows):
            b = T[u].reshape(-1).astype(np.float64)
            sol, _, _, _ = np.linalg.lstsq(VW, b, rcond=None)
            U_new[u] = sol
        U = U_new
        
        # Update V
        UW = np.einsum('ur,wr->uwr', U, W).reshape(U_rows * W_rows, R)
        V_new = np.zeros_like(V)
        for v in range(V_rows):
            b = T[:, v, :].reshape(-1).astype(np.float64)
            sol, _, _, _ = np.linalg.lstsq(UW, b, rcond=None)
            V_new[v] = sol
        V = V_new
        
        # Update W
        UV = np.einsum('ur,vr->uvr', U, V).reshape(U_rows * V_rows, R)
        W_new = np.zeros_like(W)
        for w in range(W_rows):
            b = T[:, :, w].reshape(-1).astype(np.float64)
            sol, _, _, _ = np.linalg.lstsq(UV, b, rcond=None)
            W_new[w] = sol
        W = W_new
        
        # Check convergence
        if it > 0:
            prev_err = np.sum((np.einsum('u,v,w->uvw', U[:, 0], V[:, 0], W[:, 0]) - T)**2)  # rough
        # Just continue for max_iters
    
    return U, V, W


def _try_rank17(T, deadline):
    """
    Attempt to find a rank-17 decomposition by initializing from the rank-18 
    decomposition and merging one column pair, then running ALS.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    R = 17
    
    # Start from rank-18 and try to merge columns
    U18, V18, W18 = _build_rank18()
    
    # Try different initializations
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed)
        
        # Initialize by taking rank-18 and adding small random perturbation
        # to the last two columns, then dropping one
        U_init = U18[:, :16].copy()
        V_init = V18[:, :16].copy()
        W_init = W18[:, :16].copy()
        
        # Add a 17th column as a random combination
        U_init = np.zeros((U_rows, R))
        V_init = np.zeros((V_rows, R))
        W_init = np.zeros((W_rows, R))
        
        # Copy first 16 columns from rank-18
        U_init[:, :16] = U18[:, :16]
        V_init[:, :16] = V18[:, :16]
        W_init[:, :16] = W18[:, :16]
        
        # 17th column: random
        U_init[:, 16] = np.random.randn(U_rows) * 0.1
        V_init[:, 16] = np.random.randn(V_rows) * 0.1
        W_init[:, 16] = np.random.randn(W_rows) * 0.1
        
        # Run ALS
        U, V, W = _als_update(U_init, V_init, W_init, T, R, max_iters=80)
        
        # Check if it's exact
        if _check_decomposition(U, V, W, T):
            # Try to rationalize
            for denom in [2, 4]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_decomposition(U_s, V_s, W_s, T):
                    return U_s, V_s, W_s
            return U, V, W
    
    return None


def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try rank 17
    result17 = _try_rank17(T, deadline)
    if result17 is not None:
        U, V, W = result17
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 18
    U, V, W = _build_rank18()
    
    # Verify
    if not _check_decomposition(U, V, W, T):
        # Something went wrong, fall back to schoolbook rank 20
        U_s = np.zeros((4, 20))
        V_s = np.zeros((10, 20))
        W_s = np.zeros((10, 20))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U_s[u, col] = 1
                    V_s[v, col] = 1
                    W_s[w, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
