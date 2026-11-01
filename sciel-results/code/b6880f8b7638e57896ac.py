import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target: Rank 22 (2 x Rank 11 for 2x3x3)
    # If we can find Rank 11 for 2x3x3, we can build Rank 22 for 2x3x6.
    
    # Define 2x3x3 Tensor
    # C_ik = sum_j A_ij B_jk
    # i in 0..1, j in 0..2, k in 0..2
    # Tensor T[i, j, k] = 1 if it's a valid product term? No.
    # The tensor is the identity mapping from A, B to C.
    # T_{i j k} = 1 if j is the summation index? 
    # Actually, the tensor T has indices i (row of A/C), j (col of A/row of B), k (col of C).
    # T[i, j, k] = 1.
    T_233 = np.zeros((2, 3, 3))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                T_233[i, j, k] = 1.0 # This is wrong. The tensor is T_{ijk} = delta_{jk}? No.
                # C_{ik} = sum_j A_{ij} B_{jk}.
                # The coefficient of A_{ij} B_{jk} in C_{ik} is 1.
                # So the tensor T has non-zero entries at T[i, j, k] for the term contributing to C[i, k]?
                # No, the tensor T is defined such that C_{ik} = sum_{j} T_{i j k} A_{ij} B_{jk}.
                # Here T_{i j k} = 1 for all i, j, k? No.
                # T_{i j k} is 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
                # Yes, for all i, j, k.
                # Wait, if i=0, j=1, k=2, A[0,1]B[1,2] contributes to C[0,2].
                # So T[0, 1, 2] = 1.
                # So T is all ones?
                # No, the tensor shape is (n, m, p).
                # T[i, j, k] = 1.
                # Yes, the matrix multiplication tensor is the all-ones tensor?
                # No, it's the identity in the middle index?
                # C_{ik} = sum_j A_{ij} B_{jk}.
                # The term A_{ij} B_{jk} is multiplied by 1.
                # So T[i, j, k] = 1.
                # Correct.
                pass
    T_233[:] = 1.0
    
    # Search for Rank 11 decomposition of T_233
    # Variables: U (2x3, 11), V (3x3, 11), W (3x2, 11)
    # Total vars: 6*11 + 9*11 + 6*11 = 33*11 = 363.
    
    best_rank_233 = 18 # Schoolbook
    best_U_233 = None
    best_V_233 = None
    best_W_233 = None
    
    for r in range(11, 13): # Try 11, 12
        if time.time() > deadline:
            break
        for restart in range(20):
            if time.time() > deadline:
                break
            np.random.seed(42 + restart)
            U = np.random.randn(2*3, r)
            V = np.random.randn(3*3, r)
            W = np.random.randn(3*2, r)
            
            def objective(params):
                U_f = params[:6*r].reshape(2*3, r)
                V_f = params[6*r:15*r].reshape(3*3, r)
                W_f = params[15*r:].reshape(3*2, r)
                
                # Compute tensor product sum
                # T_calc[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # Note: W index is k*n + i = k*2 + i.
                T_calc = np.zeros((2, 3, 3))
                for c in range(r):
                    u_col = U_f[:, c].reshape(2, 3)
                    v_col = V_f[:, c].reshape(3, 3)
                    w_col = W_f[:, c].reshape(3, 2)
                    # u_col[i, j] * v_col[j, k] * w_col[k, i]
                    # Sum over c
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
                                
                return np.sum((T_233 - T_calc)**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            
            if res.fun < 1e-9:
                best_rank_233 = r
                U_flat = res.x[:6*r].reshape(2*3, r)
                V_flat = res.x[6*r:15*r].reshape(3*3, r)
                W_flat = res.x[15*r:].reshape(3*2, r)
                best_U_233 = U_flat
                best_V_233 = V_flat
                best_W_233 = W_flat
                break
        if best_rank_233 == 11:
            break
            
    if best_U_233 is not None:
        # Construct Rank 22 for 2x3x6
        r_total = best_rank_233 * 2
        U = np.zeros((2*3, r_total))
        V = np.zeros((3*6, r_total))
        W = np.zeros((6*2, r_total))
        
        # Block 1: B columns 0-2
        U[:, :best_rank_233] = best_U_233
        V[:9, :best_rank_233] = best_V_233
        W[:6, :best_rank_233] = best_W_233 # W is 6x2, indices k*2+i. k in 0..2 for block 1?
        # W index for block 1: k in 0..2. W[k*2 + i].
        # W shape is (6, 22). First 6 rows correspond to k=0,1,2.
        
        # Block 2: B columns 3-5
        U[:, best_rank_233:] = best_U_233
        V[9:, best_rank_233:] = best_V_233
        W[6:, best_rank_233:] = best_W_233 # k in 3..5. W[k*2+i]. Rows 6..11.
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: Schoolbook Rank 36
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
