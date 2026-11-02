import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Helper to create schoolbook solution
    def get_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution (e.g., rank 22)
    # This is a heuristic search. If it fails, we fall back to schoolbook.
    # We only attempt this if we have time.
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Known minimal rank for 2x3x4 is 22.
    # We will attempt to find a rank 22 solution.
    # Strategy: Random sparse initialization + Least Squares + Rounding.
    
    def try_find_rank(r):
        if time.time() > deadline:
            return None
            
        # Target tensor T
        # T[i, j, k] = 1
        # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        # Reshape indices for vectorization
        # U_flat: (n*m, r)
        # V_flat: (m*p, r)
        # W_flat: (p*n, r)
        
        # We need to solve for U, V, W.
        # This is a 3-tensor CP decomposition.
        # We can fix W to random values and solve for U, V? No, it's bilinear in U and V for fixed W.
        # Actually, for fixed W, the equations are linear in U and V?
        # T_ijk = sum_c U_ijc V_jkc W_kic
        # This is not linear in U and V simultaneously if W is fixed?
        # It is linear in U if V*W is known, but V*W depends on c.
        
        # Alternative: Use alternating least squares (ALS) or similar.
        # Given the small size, we can use a simple random search with refinement.
        
        # Initialize U, V, W with random sparse entries from {-1, 0, 1}
        # But we need exact half-integers.
        
        # Let's try a specific known structure if possible, or just return schoolbook.
        # Given the complexity and time limit, a robust search is hard to guarantee.
        # However, we can try a few random restarts.
        
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Random sparse init
            # Density 0.5
            U_rand = np.random.choice([-1, 0, 1], size=(n*m, r))
            V_rand = np.random.choice([-1, 0, 1], size=(m*p, r))
            W_rand = np.random.choice([-1, 0, 1], size=(p*n, r))
            
            # Convert to float for optimization
            U_opt = U_rand.astype(float)
            V_opt = V_rand.astype(float)
            W_opt = W_rand.astype(float)
            
            # Define error function
            def error(params):
                U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
                U_m = U_flat.reshape(n*m, r)
                V_m = V_flat.reshape(m*p, r)
                W_m = W_flat.reshape(p*n, r)
                
                # Compute T_approx
                # T_ijk = sum_c U_ijc V_jkc W_kic
                # We can compute this efficiently using einsum
                # U: (n, m, r), V: (m, p, r), W: (p, n, r)
                U_3d = U_m.reshape(n, m, r)
                V_3d = V_m.reshape(m, p, r)
                W_3d = W_m.reshape(p, n, r)
                
                # sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                
                # Target is all 1s
                T_target = np.ones((n, m, p))
                return (T_approx - T_target).flatten()
            
            # Use scipy.optimize.least_squares
            try:
                from scipy.optimize import least_squares
                x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                res = least_squares(error, x0, method='lm', max_nfev=100)
                
                if res.cost < 1e-12:
                    # Extract solution
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    U_3d_r = U_rounded.reshape(n, m, r)
                    V_3d_r = V_rounded.reshape(m, p, r)
                    W_3d_r = W_rounded.reshape(p, n, r)
                    T_check = np.einsum('ijc,jkc,kic->ijk', U_3d_r, V_3d_r, W_3d_r)
                    
                    if np.allclose(T_check, np.ones((n, m, p)), atol=1e-9):
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
                
        return None

    # Attempt rank 22
    sol = try_find_rank(22)
    if sol:
        return sol
        
    # Attempt rank 23 (just in case)
    if time.time() < deadline - 5:
        sol = try_find_rank(23)
        if sol:
            return sol
            
    # Fallback
    return get_schoolbook()

# EVOLVE-BLOCK-END
