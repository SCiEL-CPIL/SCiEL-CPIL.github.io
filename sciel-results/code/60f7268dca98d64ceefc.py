"""Matrix multiplication 3x4 x 4x8 tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0
DIMENSIONS = [3, 4, 8]

def _build_rank88():
    """Build the known rank-88 decomposition directly."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8) -> rank 32
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8) -> rank 56
    # Strassen factors for 2x2x2
    sU = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    sV = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    sW = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # 2x4x8: split j into 2 blocks of 2, k into 4 blocks of 2
    for j_block in range(2):
        for k_block in range(4):
            j_start = j_block * 2
            k_start = k_block * 2
            for t in range(7):
                # U: local (i', j') -> global (1+i', j_start+j')
                for li in range(2):
                    for lj in range(2):
                        val = sU[t, li * 2 + lj]
                        if val != 0:
                            U[(1 + li) * m + (j_start + lj), col] += val
                # V: local (j', k') -> global (j_start+j', k_start+k')
                for lj in range(2):
                    for lk in range(2):
                        val = sV[t, lj * 2 + lk]
                        if val != 0:
                            V[(j_start + lj) * p + (k_start + lk), col] += val
                # W: local (k', i') -> global (k_start+k', 1+i')
                for lk in range(2):
                    for li in range(2):
                        val = sW[t, lk * 2 + li]
                        if val != 0:
                            W[(k_start + lk) * n + (1 + li), col] += val
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify decomposition correctness."""
    n, m, p = DIMENSIONS
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_rec = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                C_rec[i, k] += w_r[k, i] * np.dot(u_r[i, :], v_r[:, k])
    
    return np.allclose(C_expected, C_rec, atol=1e-9)

def _try_refine(U0, V0, W0, target_rank, max_time=30.0):
    """Try to find a decomposition of target_rank starting from a perturbed rank-88 init."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Truncate to target_rank with random projection
    rng = np.random.RandomState(42)
    if target_rank < U0.shape[1]:
        # Random rotation then truncate
        R_u = rng.randn(U0.shape[1], target_rank)
        R_v = rng.randn(U0.shape[1], target_rank)
        R_w = rng.randn(U0.shape[1], target_rank)
        U = U0 @ R_u
        V = V0 @ R_v
        W = W0 @ R_w
    else:
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
    
    # ALS iterations
    for it in range(200):
        if time.time() - start > max_time:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                V_block = V[j * p:(j + 1) * p, :]
                W_vec = np.array([W[k * n + i, :] for k in range(p)])
                P = V_block * W_vec
                try:
                    u, _, _, _ = np.linalg.lstsq(P, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_block = np.array([U[i * m + j, :] for i in range(n)])
                W_vec = W[k * n:(k + 1) * n, :]
                P = U_block * W_vec
                try:
                    v, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_block = np.array([U[i * m + j, :] for j in range(m)])
                V_vec = np.array([V[j * p + k, :] for j in range(m)])
                P = U_block * V_vec
                try:
                    w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-10:
            # Round and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Build guaranteed rank-88
    result88 = _build_rank88()
    U88, V88, W88 = result88["U"], result88["V"], result88["W"]
    
    # Try to find lower rank
    for target_rank in [87, 86, 85, 84]:
        time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
        if time_remaining < 15:
            break
        res = _try_refine(U88, V88, W88, target_rank, max_time=min(25, time_remaining - 5))
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return rank-88
    return {"U": U88.tolist(), "V": V88.tolist(), "W": W88.tolist()}
