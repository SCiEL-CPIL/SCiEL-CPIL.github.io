import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback (rank 12)
    def schoolbook_decomposition():
        rank = n * m * p
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
        return U, V, W, rank
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Check if entries are close to integers or half-integers
    def is_integer_or_half_integer(arr, tol=1e-6):
        for val in arr:
            if abs(val - round(val)) > tol and abs(val - (round(val) + 0.5)) > tol and abs(val - (round(val) - 0.5)) > tol:
                return False
        return True
    
    def to_integer_or_half_integer(arr):
        result = []
        for val in arr:
            # Try to snap to nearest integer or half-integer
            int_val = round(val)
            half_val1 = round(val - 0.5) + 0.5
            half_val2 = round(val + 0.5) - 0.5
            
            candidates = [int_val, half_val1, half_val2]
            best = min(candidates, key=lambda x: abs(val - x))
            result.append(best)
        return result
    
    # Try to find a decomposition for a given rank using structured search
    def try_rank_structured(rank, max_attempts=30):
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + rank)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Strategy: Use a parameterized family
            # For rank r, we have r "terms" each with:
            #   U_r: n x m matrix (4 values)
            #   V_r: m x p matrix (6 values)  
            #   W_r: p x n matrix (6 values)
            # Total: r * 16 parameters
            
            # Initialize with a structured pattern
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Random initialization with some structure
            for r in range(rank):
                # Each term contributes to some subset of (i,j,k) triples
                # Initialize with random signs and magnitudes
                U[i*m+j, r] = np.random.choice([-1, 0, 1]) if np.random.random() < 0.5 else 0
                V[j*p+k, r] = np.random.choice([-1, 0, 1]) if np.random.random() < 0.5 else 0
                W[k*n+i, r] = np.random.choice([-1, 0, 1]) if np.random.random() < 0.5 else 0
            
            # Now refine using local search
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # Use a gradient-free approach: try flipping individual entries
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current residuals
                residuals = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for r in range(rank):
                                total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residuals[(i,j,k)] = total - 1
                
                # Check if we're close to solution
                max_res = max(abs(v) for v in residuals.values())
                if max_res < 1e-6:
                    # Snap to integers/half-integers
                    U_list = to_integer_or_half_integer(U.flatten())
                    V_list = to_integer_or_half_integer(V.flatten())
                    W_list = to_integer_or_half_integer(W.flatten())
                    
                    U_mat = np.array(U_list).reshape(n*m, rank)
                    V_mat = np.array(V_list).reshape(m*p, rank)
                    W_mat = np.array(W_list).reshape(p*n, rank)
                    
                    if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank):
                        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank
                    break
                
                # Pick the worst residual and try to fix it
                worst_key = max(residuals.keys(), key=lambda x: abs(residuals[x]))
                i, j, k = worst_key
                res = residuals[worst_key]
                
                # Try adjusting one of the entries involved in this (i,j,k)
                # The entry U[i*m+j, r] affects this term
                # Changing U[i*m+j, r] by delta changes the residual by delta * V[j*p+k, r] * W[k*n+i, r]
                
                best_improvement = 0
                best_change = None
                
                for r in range(rank):
                    # Try changing U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    w_val = W[k*n+i, r]
                    if v_val != 0 and w_val != 0:
                        # We want to change U[i*m+j, r] by delta such that
                        # delta * v_val * w_val ≈ -res
                        # delta ≈ -res / (v_val * w_val)
                        delta = -res / (v_val * w_val)
                        # Snap delta to nearest 0.5
                        delta_snapped = round(delta * 2) / 2
                        if delta_snapped != 0:
                            new_U = U[i*m+j, r] + delta_snapped
                            # Estimate improvement
                            new_res = res + delta_snapped * v_val * w_val
                            improvement = abs(res) - abs(new_res)
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_change = ('U', i*m+j, r, delta_snapped)
                    
                    # Try changing V[j*p+k, r]
                    u_val = U[i*m+j, r]
                    w_val = W[k*n+i, r]
                    if u_val != 0 and w_val != 0:
                        delta = -res / (u_val * w_val)
                        delta_snapped = round(delta * 2) / 2
                        if delta_snapped != 0:
                            new_V = V[j*p+k, r] + delta_snapped
                            new_res = res + delta_snapped * u_val * w_val
                            improvement = abs(res) - abs(new_res)
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_change = ('V', j*p+k, r, delta_snapped)
                    
                    # Try changing W[k*n+i, r]
                    u_val = U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    if u_val != 0 and v_val != 0:
                        delta = -res / (u_val * v_val)
                        delta_snapped = round(delta * 2) / 2
                        if delta_snapped != 0:
                            new_W = W[k*n+i, r] + delta_snapped
                            new_res = res + delta_snapped * u_val * v_val
                            improvement = abs(res) - abs(new_res)
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_change = ('W', k*n+i, r, delta_snapped)
                
                if best_change is not None and best_improvement > 0:
                    mat_name, row, r, delta = best_change
                    if mat_name == 'U':
                        U[row, r] += delta
                    elif mat_name == 'V':
                        V[row, r] += delta
                    else:
                        W[row, r] += delta
                else:
                    # No improvement found, try random perturbation
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = np.random.randint(0, n*m)
                        r = np.random.randint(0, rank)
                        delta = np.random.choice([-0.5, 0.5])
                        U[row, r] += delta
                    elif mat == 'V':
                        row = np.random.randint(0, m*p)
                        r = np.random.randint(0, rank)
                        delta = np.random.choice([-0.5, 0.5])
                        V[row, r] += delta
                    else:
                        row = np.random.randint(0, p*n)
                        r = np.random.randint(0, rank)
                        delta = np.random.choice([-0.5, 0.5])
                        W[row, r] += delta
        
        return None
    
    # Try Newton's method approach
    def try_rank_newton(rank, max_attempts=20):
        if time.time() > deadline:
            return None
        
        try:
            from scipy.optimize import fsolve
        except ImportError:
            return None
        
        np.random.seed(100 + rank)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            scale = 0.5
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # Define residual function
            def residuals(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(rank):
                                val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            res.append(val - 1)
                return np.array(res)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                solution = fsolve(residuals, x0, full_output=True, maxfev=5000)
                x_sol, info, ier, msg = solution
                
                if ier == 1:
                    U_sol = x_sol[:n*m*rank].reshape(n*m, rank)
                    V_sol = x_sol[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = x_sol[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Check if close to integer/half-integer
                    U_list = to_integer_or_half_integer(U_sol.flatten())
                    V_list = to_integer_or_half_integer(V_sol.flatten())
                    W_list = to_integer_or_half_integer(W_sol.flatten())
                    
                    U_mat = np.array(U_list).reshape(n*m, rank)
                    V_mat = np.array(V_list).reshape(m*p, rank)
                    W_mat = np.array(W_list).reshape(p*n, rank)
                    
                    if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank):
                        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank
            except Exception:
                pass
        
        return None
    
    # Try to find solutions for ranks 10, 9, 8
    for rank in [10, 9, 8]:
        if time.time() > deadline:
            break
        
        # First try structured search
        result = try_rank_structured(rank, max_attempts=20)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
        
        # Then try Newton's method
        result = try_rank_newton(rank, max_attempts=15)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
