import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split p=5 into p1=3 and p2=2
    p1 = 3
    p2 = 2
    
    # Part 1: 3x3 * 3x3 (Schoolbook, Rank 9)
    rank1 = n * m * p1
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U1[i * m + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n + i][col] = 1
                col += 1
                
    # Part 2: 3x3 * 3x2
    # Try to find rank 15 decomposition
    target_rank2 = 15
    success = False
    U2 = None
    V2 = None
    W2 = None
    
    # Time limit for search
    start_time = time.time()
    max_search_time = 10.0
    
    if time.time() - start_time < max_search_time:
        # Define target tensor for 3x3x2
        # T[i, j, k] = 1
        # U shape: (9, R), V shape: (6, R), W shape: (6, R)
        # U index: i*m + j
        # V index: j*p2 + k
        # W index: k*n + i
        
        def residuals(params):
            R = target_rank2
            U_flat, V_flat, W_flat = params[:9*R], params[9*R:9*R+6*R], params[9*R+6*R:]
            U = U_flat.reshape(9, R)
            V = V_flat.reshape(6, R)
            W = W_flat.reshape(6, R)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p2):
                        # Compute sum U[i,j,r] * V[j,k,r] * W[k,i,r]
                        val = 0
                        for r in range(R):
                            val += U[i*m + j, r] * V[j*p2 + k, r] * W[k*n + i, r]
                        res.append(val - 1.0)
            return res

        # Initialize with random values
        np.random.seed(42)
        R = target_rank2
        init_params = np.random.randn(9*R + 6*R + 6*R) * 0.1
        
        try:
            sol = least_squares(residuals, init_params, method='lm', max_nfev=10000)
            if sol.success or sol.cost < 1e-8:
                # Extract and round
                U_flat, V_flat, W_flat = sol.x[:9*R], sol.x[9*R:9*R+6*R], sol.x[9*R+6*R:]
                U2 = np.round(U_flat.reshape(9, R) * 2) / 2
                V2 = np.round(V_flat.reshape(6, R) * 2) / 2
                W2 = np.round(W_flat.reshape(6, R) * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p2):
                            val = 0
                            for r in range(R):
                                val += U2[i*m + j, r] * V2[j*p2 + k, r] * W2[k*n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    success = True
        except Exception:
            pass

    if success:
        rank2 = target_rank2
        # Convert to lists of lists for return
        U2_list = U2.tolist()
        V2_list = V2.tolist()
        W2_list = W2.tolist()
    else:
        # Fall back to schoolbook for 3x3x2
        rank2 = n * m * p2
        U2_list = [[0] * rank2 for _ in range(n * m)]
        V2_list = [[0] * rank2 for _ in range(m * p2)]
        W2_list = [[0] * rank2 for _ in range(p2 * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U2_list[i * m + j][col] = 1
                    V2_list[j * p2 + k][col] = 1
                    W2_list[k * n + i][col] = 1
                    col += 1

    # Combine
    total_rank = rank1 + rank2
    U = U1 + U2_list
    V = V1 + V2_list
    W = W1 + W2_list
    
    return {"U": U, "V": V, "W": W}
