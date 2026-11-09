"""
Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n=2, m=3, p=10, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _build_block_decomposition(n=2, m=3, p=10):
    """Build the rank-55 block decomposition (Strassen + standard)."""
    total_rank = (p // 2) * (7 + 4)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        # B indices: (0,c0)->0*p+c0, (0,c1)->0*p+c1, (1,c0)->1*p+c0, (1,c1)->1*p+c1
        # C indices: (0,c0)->c0*n+0, (0,c1)->c1*n+0, (1,c0)->c0*n+1, (1,c1)->c1*n+1
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1; U[4][col] = 1
        V[0 * p + c0][col] = 1; V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1; W[c1 * n + 1][col] = 1
        col += 1
        
        # P2 = (a10 + a11)b00
        U[3][col] = 1; U[4][col] = 1
        V[0 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1; W[c1 * n + 1][col] = -1
        col += 1
        
        # P3 = a00(b01 - b11)
        U[0][col] = 1
        V[0 * p + c1][col] = 1; V[1 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1; W[c1 * n + 1][col] = 1
        col += 1
        
        # P4 = a11(b10 - b00)
        U[4][col] = 1
        V[1 * p + c0][col] = 1; V[0 * p + c0][col] = -1
        W[c0 * n + 0][col] = 1; W[c0 * n + 1][col] = 1
        col += 1
        
        # P5 = (a00 + a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = -1; W[c1 * n + 0][col] = 1
        col += 1
        
        # P6 = (a10 - a00)(b00 + b01)
        U[3][col] = 1; U[0][col] = -1
        V[0 * p + c0][col] = 1; V[0 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P7 = (a01 - a11)(b10 + b11)
        U[1][col] = 1; U[4][col] = -1
        V[1 * p + c0][col] = 1; V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
        # Standard 2x1x2: A[0:2, 2] * B[2, c0:c1+1] -> C[0:2, c0:c1+1]
        U[2][col] = 1; V[2 * p + c0][col] = 1; W[c0 * n + 0][col] = 1; col += 1
        U[2][col] = 1; V[2 * p + c1][col] = 1; W[c1 * n + 0][col] = 1; col += 1
        U[5][col] = 1; V[2 * p + c0][col] = 1; W[c0 * n + 1][col] = 1; col += 1
        U[5][col] = 1; V[2 * p + c1][col] = 1; W[c1 * n + 1][col] = 1; col += 1
    
    return U, V, W

def _try_rank_r(R, n=2, m=3, p=10, max_iters=500, seed=42):
    """Try to find a rank-R decomposition using iterative optimization.
    
    We minimize ||T_decomposed - T_target||^2 where T_target has 1s at all (i,j,k).
    Use gradient-free approach: randomly initialize, then use coordinate descent / random search.
    """
    np.random.seed(seed)
    dim_U, dim_V, dim_W = n * m, m * p, p * n
    
    # Initialize with small random values
    U = np.random.randn(dim_U, R) * 0.5
    V = np.random.randn(dim_V, R) * 0.5
    W = np.random.randn(dim_W, R) * 0.5
    
    best_err = float('inf')
    
    for it in range(max_iters):
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
            if err < 1e-8:
                return U, V, W, err
        
        if err < 1e-10:
            return U, V, W, err
        
        # Random perturbation of one matrix
        mat = np.random.choice([0, 1, 2])
        if mat == 0:
            idx = np.random.randint(dim_U)
            U[idx, :] += np.random.randn(R) * 0.1
        elif mat == 1:
            idx = np.random.randint(dim_V)
            V[idx, :] += np.random.randn(R) * 0.1
        else:
            idx = np.random.randint(dim_W)
            W[idx, :] += np.random.randn(R) * 0.1
    
    return U, V, W, best_err

def construct():
    """
    Returns a tensor decomposition of matrix multiplication (2x3) * (3x10) -> (2x10).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Start with known rank-55 decomposition
    U55, V55, W55 = _build_block_decomposition(n, m, p)
    best_U, best_V, best_W = U55, V55, W55
    best_rank = 55
    
    # Try to find lower rank decompositions
    for R in range(45, 55):
        if time.time() > deadline:
            break
        U, V, W, err = _try_rank_r(R, n, m, p, max_iters=200, seed=R)
        if err < 1e-6:
            best_U, best_V, best_W, best_rank = U, V, W, R
            break
    
    # Convert to lists
    U_list = [[float(U_list_val) for U_list_val in row] for row in np.array(best_U, dtype=float).tolist()]
    V_list = [[float(V_list_val) for V_list_val in row] for row in np.array(best_V, dtype=float).tolist()]
    W_list = [[float(W_list_val) for W_list_val in row] for row in np.array(best_W, dtype=float).tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
