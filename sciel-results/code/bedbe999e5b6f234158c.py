"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # 1. Construct Schoolbook Solution (Rank 50)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=float)
    V_school = np.zeros((mp, rank_school), dtype=float)
    W_school = np.zeros((pn, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(), 
        "V": V_school.tolist(), 
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # 2. Attempt to find a lower rank (e.g., 49) using numerical optimization
    # We use a very short time budget to avoid impacting the main return.
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds max for search
    
    # Target rank to try
    target_rank = best_rank - 1 # Try rank 49
    
    # Build target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1.0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def objective(params):
        """
        Objective function for rank R decomposition.
        params: flat array of size R * (nm + mp + pn)
        """
        R = target_rank
        size_U, size_V, size_W = nm, mp, pn
        
        # Reshape and transpose to get (size, R) matrices
        U = params[:R * size_U].reshape(R, size_U).T
        V = params[R * size_U:R * (size_U + size_V)].reshape(R, size_V).T
        W = params[R * (size_U + size_V):].reshape(R, size_W).T
        
        # Compute reconstructed tensor
        # T_comp[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Using einsum for speed
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Sum of squared errors
        err = np.sum((T_comp - T) ** 2)
        return err

    def get_factors(params, R):
        size_U, size_V, size_W = nm, mp, pn
        U = params[:R * size_U].reshape(R, size_U).T
        V = params[R * size_U:R * (size_U + size_V)].reshape(R, size_V).T
        W = params[R * (size_U + size_V):].reshape(R, size_W).T
        return U, V, W

    # Try a few random restarts
    # We initialize close to the schoolbook solution to help convergence
    # However, schoolbook has rank 50, target is 49.
    # We can't directly initialize rank 49 from rank 50 easily without dropping a term.
    # We drop the last term of the schoolbook solution and try to fit the rest.
    
    # Create an initial guess based on schoolbook minus one term
    # We take the first 49 terms of the schoolbook solution.
    # But the shapes don't match directly.
    # Let's just use random initialization but with a strict time limit.
    
    attempts = 5
    for attempt in range(attempts):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + attempt)
        total_size = target_rank * (nm + mp + pn)
        x0 = np.random.uniform(-1, 1, total_size)
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 20, 'maxfun': 20})
            
            if res.fun < 1e-4:
                # Check if we can round to half-integers
                U, V, W = get_factors(res.x, target_rank)
                
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                T_comp = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(T_comp, T, atol=1e-8):
                    # Found a valid lower rank decomposition
                    best_result = {
                        "U": U_r.tolist(), 
                        "V": V_r.tolist(), 
                        "W": W_r.tolist()
                    }
                    best_rank = target_rank
                    break # Stop searching, found improvement
        except Exception:
            pass

    # If time is running out, ensure we return the best valid result
    if time.time() > 140.0:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
