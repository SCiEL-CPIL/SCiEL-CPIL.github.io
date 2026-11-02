import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m   # 9
    V_dim = m * p   # 15
    W_dim = p * n   # 15
    
    # Build target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s kill
    
    # First, try explicit low-rank constructions (fast, deterministic)
    # These are known decompositions that should work exactly
    
    # Try rank 13: Strassen(3x3x3) + schoolbook for 2 extra columns
    # Strassen 3x3x3 uses 7 multiplications, plus 3+3 for columns 3,4
    # But we need to verify this is correct for the full 3x3x5 case
    
    # Actually, let's try a more systematic approach:
    # For 3x3 x 3x5, we can think of it as:
    # - First 3 columns of B: 3x3x3 multiplication (rank 7 with Strassen)
    # - Last 2 columns of B: 3x3 x 3x2 multiplication
    
    # But the tensor structure is more subtle. Let me try direct optimization.
    
    # Try ranks from low to high, with multiple seeds
    best_result = None
    best_rank = n * m * p  # schoolbook = 45
    
    # Schoolbook as fallback
    U_school = [[0] * best_rank for _ in range(U_dim)]
    V_school = [[0] * best_rank for _ in range(V_dim)]
    W_school = [[0] * best_rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try optimization for lower ranks
    for rank in range(10, 46):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(15):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + rank)
            
            # Initialize U, V, W
            U = np.random.randn(U_dim, rank) * 0.1
            V = np.random.randn(V_dim, rank) * 0.1
            W = np.random.randn(W_dim, rank) * 0.1
            
            # Optimize using scipy
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    U_flat = params[:U_dim * rank]
                    V_flat = params[U_dim * rank:U_dim * rank + V_dim * rank]
                    W_flat = params[U_dim * rank + V_dim * rank:]
                    U_c = U_flat.reshape(U_dim, rank)
                    V_c = V_flat.reshape(V_dim, rank)
                    W_c = W_flat.reshape(W_dim, rank)
                    
                    D = np.einsum('uc,vc,wc->uvw', U_c, V_c, W_c)
                    err = np.sum((T - D) ** 2)
                    return err
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                res = minimize(objective, params, method='BFGS', 
                              options={'maxiter': 200, 'gtol': 1e-10})
                
                if res.fun < 1e-6:
                    # Check rounding to nearest 0.5
                    U_final = res.x[:U_dim * rank].reshape(U_dim, rank)
                    V_final = res.x[U_dim * rank:U_dim * rank + V_dim * rank].reshape(V_dim, rank)
                    W_final = res.x[U_dim * rank + V_dim * rank:].reshape(W_dim, rank)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_final * 2) / 2
                    V_round = np.round(V_final * 2) / 2
                    W_round = np.round(W_final * 2) / 2
                    
                    # Verify
                    D_check = np.einsum('uc,vc,wc->uvw', U_round, V_round, W_round)
                    if np.allclose(T, D_check, atol=1e-6):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                            break  # Found a valid decomposition at this rank
            except Exception:
                pass
        
        if best_result is not None and best_rank <= rank:
            break  # Found the best possible rank
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        return {
            "U": U_school,
            "V": V_school,
            "W": W_school
        }
