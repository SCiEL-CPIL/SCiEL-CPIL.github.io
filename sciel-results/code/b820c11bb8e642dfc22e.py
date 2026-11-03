"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # Tensor code / Berlekamp-Massey approach:
    # Identify the matmul tensor with parity-check structure of a tensor product code.
    # The 30 scalar products T[i,j,k]=1 have algebraic dependencies that can be
    # exploited to reduce rank below the schoolbook 30.
    #
    # Key structural insight: For 2x3 * 3x5, we can exploit that n=2 is small.
    # The output C[i,k] for i=0,1 and k=0..4 shares B[j,k] terms.
    #
    # We use an iterative alternating least squares approach:
    # 1. Initialize W randomly
    # 2. Fix W, solve linear system for U (given V)
    # 3. Fix U, solve linear system for V (given W)
    # 4. Iterate until convergence
    # 5. Try to round to half-integers
    #
    # We search from low rank upward.
    
    def try_rank(rank, seed=0):
        np.random.seed(seed)
        W = np.random.randn(W_dim, rank)
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        
        for it in range(200):
            # Solve for U given V and W
            # For each (i,k): sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            A_U = np.zeros((p * n, U_dim * rank))
            b = np.ones(p * n)
            for i in range(n):
                for k in range(p):
                    row = k * n + i
                    for j in range(m):
                        for c in range(rank):
                            col = c * U_dim + (i * m + j)
                            A_U[row, col] = V[j * p + k, c] * W[k * n + i, c]
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_flat.reshape(rank, U_dim).T
            
            # Solve for V given U and W
            A_V = np.zeros((p * n, V_dim * rank))
            for i in range(n):
                for k in range(p):
                    row = k * n + i
                    for j in range(m):
                        for c in range(rank):
                            col = c * V_dim + (j * p + k)
                            A_V[row, col] = U[i * m + j, c] * W[k * n + i, c]
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_flat.reshape(rank, V_dim).T
            
            # Check residual
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res += abs(val - 1.0)
            if res < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                res2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            res2 += abs(val - 1.0)
                if res2 < 1e-8:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                # If rounding fails, try integer rounding
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                res2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            res2 += abs(val - 1.0)
                if res2 < 1e-8:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    # Try ranks from low to high
    for rank in range(15, 31):
        for seed in range(20):
            result = try_rank(rank, seed)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
