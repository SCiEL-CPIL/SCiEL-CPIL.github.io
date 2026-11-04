import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 6
    
    # Target: Find a rank R decomposition for 3x3x3 matrix multiplication.
    # Then use 2*R for 3x3x6.
    
    # The tensor T for 3x3x3 is all ones of shape (3, 3, 3) where T[i, k, j] = 1.
    # Decomposition: sum_r U[i, j] * V[j, k] * W[k, i] = 1 for all i, j, k.
    # Note: U is (3,3), V is (3,3), W is (3,3).
    
    def residual(params, R):
        # params contains R sets of U, V, W matrices.
        # Each matrix is 3x3 = 9 elements.
        # Total variables = R * 27.
        U_list = []
        V_list = []
        W_list = []
        idx = 0
        for r in range(R):
            U_list.append(params[idx:idx+9].reshape(3,3))
            V_list.append(params[idx+9:idx+18].reshape(3,3))
            W_list.append(params[idx+18:idx+27].reshape(3,3))
            idx += 27
        
        res = np.zeros((3, 3, 3))
        for r in range(R):
            # res[i, k, j] += U[i, j] * V[j, k] * W[k, i]
            # U[i, j], V[j, k], W[k, i]
            # We need to compute the sum efficiently.
            # for i in 3:
            #   for j in 3:
            #     for k in 3:
            #       res[i, k, j] += U_list[r][i, j] * V_list[r][j, k] * W_list[r][k, i]
            
            # Vectorized:
            # U[i, j] is 3x3
            # V[j, k] is 3x3
            # W[k, i] is 3x3
            # The term is U[i, j] * V[j, k] * W[k, i].
            # This is a tensor product.
            # We can use np.einsum or loops.
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        res[i, k, j] += U_list[r][i, j] * V_list[r][j, k] * W_list[r][k, i]
        
        return (res - 1.0).flatten()
    
    best_rank = 54
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 23 to 26
    for R in range(23, 27):
        # Try a few random initializations
        for trial in range(50):
            np.random.seed(trial)
            # Initialize with small random values
            params = np.random.uniform(-1, 1, size=R * 27)
            
            try:
                sol = least_squares(residual, params, args=(R,), method='lm', max_nfev=1000)
                if sol.cost < 1e-10:
                    # Check for integer/half-integer entries
                    U_list = []
                    V_list = []
                    W_list = []
                    idx = 0
                    for r in range(R):
                        U_list.append(sol.x[idx:idx+9].reshape(3,3))
                        V_list.append(sol.x[idx+9:idx+18].reshape(3,3))
                        W_list.append(sol.x[idx+18:idx+27].reshape(3,3))
                        idx += 27
                    
                    # Round to nearest half-integer
                    U_rounded = [np.round(U * 2) / 2 for U in U_list]
                    V_rounded = [np.round(V * 2) / 2 for V in V_list]
                    W_rounded = [np.round(W * 2) / 2 for W in W_list]
                    
                    # Verify exactness
                    res_check = np.zeros((3, 3, 3))
                    for r in range(R):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    res_check[i, k, j] += U_rounded[r][i, j] * V_rounded[r][j, k] * W_rounded[r][k, i]
                    
                    if np.allclose(res_check, 1.0):
                        # Found a valid decomposition for 3x3x3
                        # Construct for 3x3x6
                        total_rank = 2 * R
                        U_full = [[0] * total_rank for _ in range(9)]
                        V_full = [[0] * total_rank for _ in range(18)]
                        W_full = [[0] * total_rank for _ in range(18)]
                        
                        # Block 1: k in 0..2
                        for r in range(R):
                            for i in range(3):
                                for j in range(3):
                                    U_full[i * 3 + j][r] = U_rounded[r][i, j]
                                    # V[j, k] where k is 0..2
                                    V_full[j * 6 + k][r] = V_rounded[r][j, k]
                                    # W[k, i] where k is 0..2
                                    W_full[k * 3 + i][r] = W_rounded[r][k, i]
                            
                        # Block 2: k in 3..5
                        for r in range(R):
                            for i in range(3):
                                for j in range(3):
                                    U_full[i * 3 + j][r + R] = U_rounded[r][i, j]
                                    # V[j, k] where k is 3..5
                                    V_full[j * 6 + (k + 3)][r + R] = V_rounded[r][j, k]
                                    # W[k, i] where k is 3..5
                                    W_full[(k + 3) * 3 + i][r + R] = W_rounded[r][k, i]
                        
                        return {"U": U_full, "V": V_full, "W": W_full}
                        
            except Exception as e:
                pass
    
    # Fallback to schoolbook 54
    rank = 54
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[i * 3 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
