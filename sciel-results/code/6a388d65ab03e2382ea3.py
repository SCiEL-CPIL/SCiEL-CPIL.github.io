"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use the schoolbook decomposition as a safe fallback (rank 42).
- Attempt to find a lower-rank decomposition using scipy.optimize.least_squares
  with multiple random restarts, followed by rounding to half-integers and
  exact verification with Fraction arithmetic.
- The search is guided by time limits to ensure completion within 120s.
"""

import time
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    # Convert to Fraction for exact arithmetic
                    # Inputs are expected to be int or float (half-integers)
                    u_val = Fraction(str(U[u_idx][c]))
                    v_val = Fraction(str(V[v_idx][c]))
                    w_val = Fraction(str(W[w_idx][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Build target tensor T
    # T[alpha, beta, gamma] = 1 if indices match
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    T_flat = T.flatten()

    def make_residual_func(R):
        def res(params):
            len_U = N_U * R
            len_V = N_V * R
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            
            # T_calc[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            return (T_calc.flatten() - T_flat)
        
        return res

    def try_rank(R, time_limit):
        if R <= 0:
            return None
        
        res_func = make_residual_func(R)
        total_vars = (N_U + N_V + N_W) * R
        
        best_sol = None
        best_cost = np.inf
        
        start_time_local = time.time()
        
        # Number of restarts depends on time limit
        # Estimate: each least_squares call takes ~0.1-0.5s for small R
        # We'll just run until time limit
        
        for restart in range(1000): # Cap restarts
            if time.time() > start_time_local + time_limit:
                break
            
            # Random initialization
            # Scale: values in [-1, 1] might be too small or large.
            # Let's try uniform [-2, 2]
            x0 = np.random.uniform(-2, 2, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # Check if error is close to 0
        # If cost < 1e-6, try rounding to half-integers
        if best_cost < 1e-4:
            # Round to half-integers
            # Multiply by 2, round, divide by 2
            x_rounded = np.round(best_sol * 2) / 2.0
            
            # Verify exactness with rounded solution using float first for speed
            res_rounded = res_func(x_rounded)
            if np.linalg.norm(res_rounded) < 1e-9:
                # Extract U, V, W
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded[:len_U].reshape(N_U, R)
                V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded[len_U+len_V:].reshape(N_W, R)
                
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Final exact verification with Fractions
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If float check passed but exact failed, it might be due to precision issues in rounding
                # Try a more precise rounding or check if we can snap to integers if close
                # For now, if exact fails, we don't return it.
        
        return None

    # Ensure we have the schoolbook solution as a fallback
    best_result = _schoolbook()
    schoolbook_rank = n * m * p
    
    # Try to find lower rank
    # Heuristic: try R from schoolbook-1 down to some lower bound
    # We will allocate time budget dynamically.
    # Start with a few high ranks (close to schoolbook) as they are easier to solve?
    # Or start low? Low rank is harder to find.
    # Let's try a range.
    
    # Allocate 100s for search
    search_deadline = start_time + 100
    
    # Try ranks from 41 down to 20
    # If we find one, we can stop or continue to find lower?
    # The task asks for rank-minimal. So we should try to go as low as possible.
    # However, finding the true minimum is hard.
    # Let's try ranks in increasing order of difficulty? No, usually higher rank is easier to fit.
    # But we want the MINIMUM.
    # So we should try low ranks first? If a low rank solution exists, it's a subset of the solution space.
    # But random search might miss it.
    # Let's try a mix.
    
    # Let's try ranks 30 to 41 first, as they are closer to schoolbook and might be easier to converge.
    # If we find a solution at 40, we then try 39, 38, etc.
    
    # Actually, let's just iterate R from 20 to 41.
    # If we find a solution at R, we record it and continue to try R-1.
    
    found_rank = schoolbook_rank
    
    # To maximize chance of finding *any* improvement, we might want to spend time on
    # ranks that are likely to have solutions.
    # For 2x3x7, rank 42 is trivial. Rank 41 might be possible.
    
    # Let's try R from 41 down to 20.
    # If we find a solution at R, we set best_result and continue to R-1.
    # We stop if time is up.
    
    for R in range(schoolbook_rank - 1, 19, -1):
        if time.time() > search_deadline:
            break
        
        # Allocate remaining time for this rank, with a cap per rank to avoid getting stuck
        time_left = search_deadline - time.time()
        time_for_rank = min(10, time_left) # Max 10s per rank
        
        if time_for_rank <= 1:
            break
            
        result = try_rank(R, time_for_rank)
        if result is not None:
            best_result = result
            found_rank = R
            # Continue to try lower ranks
    
    return best_result
