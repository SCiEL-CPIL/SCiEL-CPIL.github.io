"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_3x3x3(U, V, W, tol=1e-6):
    """Verify rank-r decomposition for 3x3x3."""
    # U, V, W are 9 x r
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_23_3x3x3(seed, max_iters=500, deadline=None):
    """Try to find rank 23 decomposition for 3x3x3."""
    rng = np.random.RandomState(seed)
    n = m = p = 3
    rank = 23
    U = rng.randn(9, rank)
    V = rng.randn(9, rank)
    W = rng.randn(9, rank)
    
    best_err = float('inf')
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
            
        # ALS Step
        # Solve for U
        for i in range(3):
            for j in range(3):
                M = np.zeros((3, rank))
                for k in range(3):
                    M[k, :] = V[j*3+k, :] * W[k*3+i, :]
                try:
                    U[i*3+j, :] = np.linalg.lstsq(M.T, np.ones(3), rcond=None)[0]
                except:
                    pass
        # Solve for V
        for j in range(3):
            for k in range(3):
                M = np.zeros((3, rank))
                for i in range(3):
                    M[i, :] = U[i*3+j, :] * W[k*3+i, :]
                try:
                    V[j*3+k, :] = np.linalg.lstsq(M.T, np.ones(3), rcond=None)[0]
                except:
                    pass
        # Solve for W
        for k in range(3):
            for i in range(3):
                M = np.zeros((3, rank))
                for j in range(3):
                    M[j, :] = U[i*3+j, :] * V[j*3+k, :]
                try:
                    W[k*3+i, :] = np.linalg.lstsq(M.T, np.ones(3), rcond=None)[0]
                except:
                    pass
        
        # Calculate error
        err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                    err += (s - 1.0)**2
        
        if err < best_err:
            best_err = err
            if best_err < 1e-8:
                # Try rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify_3x3x3(U_int, V_int, W_int, tol=1e-6):
                    return U_int, V_int, W_int
        if best_err < 1e-10:
            break
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to find rank 23 for 3x3x3
    U_23 = None
    V_23 = None
    W_23 = None
    
    # Try a few seeds
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
        res = _try_rank_23_3x3x3(seed, max_iters=100, deadline=deadline)
        if res is not None:
            U_23, V_23, W_23 = res
            break
            
    if U_23 is not None:
        # Construct rank 46 for 3x3x6
        rank = 46
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # First block (cols 0-2 of B, rows 0-2 of C)
        U[:, 0:23] = U_23
        V[0:9, 0:23] = V_23 # V indices: j*p+k. j in 0..2, k in 0..2 -> 0..8
        W[0:9, 0:23] = W_23 # W indices: k*n+i. k in 0..2, i in 0..2 -> 0..8
        
        # Second block (cols 3-5 of B, rows 3-5 of C)
        U[:, 23:46] = U_23
        # V indices for second block: j in 0..2, k in 3..5
        # Index in V matrix: j*6 + k
        # j=0, k=3..5 -> 3,4,5
        # j=1, k=3..5 -> 9,10,11
        # j=2, k=3..5 -> 15,16,17
        # V_23 has indices j*3+k.
        # We need to map V_23[j*3+k] to V[j*6+k+3]
        for j in range(3):
            for k in range(3):
                V[j*6 + (k+3), 23:46] = V_23[j*3+k, :]
                
        # W indices for second block: k in 3..5, i in 0..2
        # Index in W matrix: k*3 + i
        # k=3, i=0..2 -> 9,10,11
        # k=4, i=0..2 -> 12,13,14
        # k=5, i=0..2 -> 15,16,17
        # W_23 has indices k*3+i.
        for k in range(3):
            for i in range(3):
                W[(k+3)*3 + i, 23:46] = W_23[k*3+i, :]
                
        # Verify the full 3x3x6 decomposition
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    s = np.sum(U[i*3+j, :] * V[j*6+k, :] * W[k*3+i, :])
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
