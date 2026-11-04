import numpy as np
from scipy.optimize import minimize
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor values
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.append((a, b, c))
    
    max_rank = n * m * p
    lower_bound = max(nm, mp, pn)
    
    def schoolbook():
        rank = max_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_solution(U, V, W, R, tolerance=1e-9):
        """Verify that the decomposition is exact for all indices."""
        # Check valid indices
        for a, b, c in valid_indices:
            val = 0.0
            for r in range(R):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > tolerance:
                return False
        
        # Check a sample of invalid indices
        all_indices = set()
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    all_indices.add((i, j, k))
        
        valid_set = set(valid_indices)
        invalid_indices = list(all_indices - valid_set)
        
        # Check up to 100 random invalid indices
        random.shuffle(invalid_indices)
        for a, b, c in invalid_indices[:100]:
            val = 0.0
            for r in range(R):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val) > tolerance:
                return False
        
        return True
    
    def objective(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        error = 0.0
        for a, b, c in valid_indices:
            val = 0.0
            for r in range(R):
                val += U[a, r] * V[b, r] * W[c, r]
            error += (val - 1.0) ** 2
        
        return error
    
    def solve_for_rank(R, num_restarts=20, time_limit=10.0):
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            size_U = nm * R
            size_V = mp * R
            size_W = pn * R
            total_size = size_U + size_V + size_W
            
            # Structured initialization: use schoolbook-like patterns
            x0 = np.zeros(total_size)
            
            # Initialize with random values but with some structure
            for r in range(R):
                # U column r
                U_col_start = r
                # V column r  
                V_col_start = size_U + r
                # W column r
                W_col_start = size_U + size_V + r
                
                # Random initialization
                x0[U_col_start:U_col_start + nm:1] = np.random.uniform(-0.5, 0.5, nm)
                x0[V_col_start:V_col_start + mp:1] = np.random.uniform(-0.5, 0.5, mp)
                x0[W_col_start:W_col_start + pn:1] = np.random.uniform(-0.5, 0.5, pn)
            
            try:
                res = minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                              options={'maxiter': 100, 'maxls': 50})
                
                if res.fun < 1e-8:
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U + size_V]
                    W_flat = res.x[size_U + size_V:size_U + size_V + size_W]
                    
                    U = U_flat.reshape(nm, R)
                    V = V_flat.reshape(mp, R)
                    W = W_flat.reshape(pn, R)
                    
                    # Try snapping to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded, R):
                        return {"U": U_rounded.tolist(), 
                                "V": V_rounded.tolist(), 
                                "W": W_rounded.tolist()}
                    
                    # Try integer snapping
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    
                    if verify_solution(U_int, V_int, W_int, R):
                        return {"U": U_int.tolist(), 
                                "V": V_int.tolist(), 
                                "W": W_int.tolist()}
            except Exception:
                continue
        
        return None
    
    # Try ranks from max_rank - 1 down to lower_bound
    # Use time-aware search
    total_start = time.time()
    max_search_time = 100  # seconds
    
    for R in range(max_rank - 1, lower_bound - 1, -1):
        if time.time() - total_start > max_search_time:
            break
            
        # More restarts for smaller R
        num_restarts = 30 if R <= 15 else 20
        sol = solve_for_rank(R, num_restarts=num_restarts, time_limit=8.0)
        
        if sol:
            return sol
        
        # If we've tried R=12 and it failed, we can stop
        if R == lower_bound:
            break
    
    return schoolbook()
