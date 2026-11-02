"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
                
    # Flatten T for faster residual calculation
    # Residual R = T - sum_r U[:,r] V[:,r] W[:,r]
    # We can compute the residual norm efficiently.
    
    def compute_residual_norm(U, V, W):
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Using einsum is fast
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T - T_recon
        return np.sum(diff**2)

    def verify(U, V, W, tol=1e-6):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=tol, rtol=tol)

    def round_half_integer(M):
        return np.round(M * 2) / 2.0

    best_result = None
    best_rank = n * m * p  # 63

    # Schoolbook initialization
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    best_rank = rank_sb

    # Search for lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Let's try ranks from 25 to 62
    # We use a local search approach:
    # 1. Initialize U, V, W randomly or with structure.
    # 2. Optimize to minimize residual.
    # 3. Round to half-integers.
    # 4. If residual is ~0, accept.

    # To make optimization faster, we can fix one matrix or use a specific parameterization,
    # but generic random init + Powell/L-BFGS is robust for small N.
    
    # We will try to find a decomposition for rank r.
    # Objective function for scipy: flatten U, V, W into a single vector x.
    
    def objective(x, r):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r:].reshape(p*n, r)
        return compute_residual_norm(U, V, W)

    def try_rank(r, num_restarts=10, time_budget_per_restart=2.0):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(restart * 1000 + r)
            
            # Random initialization
            # Scale initial values to be reasonable (around 1)
            U_init = rng.randn(n*m, r)
            V_init = rng.randn(m*p, r)
            W_init = rng.randn(p*n, r)
            
            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            # Use Powell method (derivative-free, robust for small problems)
            # Or L-BFGS-B if we compute gradients, but Powell is safer without explicit grad
            # However, Powell can be slow. Let's try a few iterations of gradient descent 
            # or just use scipy with tight limits.
            
            # For speed, we can also try a simple gradient descent with backtracking
            # or just rely on the fact that the landscape is not too crazy for these small sizes.
            
            # Let's use scipy.optimize.minimize with method='Powell'
            # maxiter needs to be small to fit in time budget
            max_iter = 1000
            try:
                res = minimize(objective, x0, args=(r,), method='Powell', 
                               options={'maxiter': max_iter, 'disp': False})
            except Exception:
                continue
                
            if res.success or res.fun < 1e-4:
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = x_opt[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Round to half-integers
                U_r = round_half_integer(U_opt)
                V_r = round_half_integer(V_opt)
                W_r = round_half_integer(W_opt)
                
                # Verify
                if verify(U_r, V_r, W_r, tol=1e-5):
                    return U_r, V_r, W_r, r
            
            # Also check if the unrounded solution is effectively zero error 
            # (which implies it's already on the grid or close)
            if res.fun < 1e-8:
                 U_r = round_half_integer(U_opt)
                 V_r = round_half_integer(V_opt)
                 W_r = round_half_integer(W_opt)
                 if verify(U_r, V_r, W_r, tol=1e-5):
                     return U_r, V_r, W_r, r

        return None

    # Try ranks from lower bound up
    # We prioritize lower ranks
    # Lower bound 21. 
    # Let's try a range. 
    # To save time, we can skip some ranks if we find a solution quickly.
    
    ranks_to_try = list(range(25, 63))
    # Add specific interesting ranks
    # For 3x3x7, rank 30-40 is a good guess.
    
    # Sort by rank
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
            
        result = try_rank(r, num_restarts=5)
        if result:
            U_r, V_r, W_r, rank_r = result
            if verify(U_r, V_r, W_r):
                best_result = (U_r, V_r, W_r, rank_r)
                best_rank = rank_r
                # If we found a low rank, we might want to check if we can go lower?
                # The loop goes upwards, so if we found 30, we continue to 31? 
                # No, we want the MINIMAL rank. So we should try from low to high.
                # If we find a solution at r, that is the minimal for this search path 
                # IF we are sure no lower rank exists. But we are scanning up.
                # So if we find rank 30, we should stop? 
                # Wait, the loop is `for r in ranks_to_try` which is ascending.
                # So the first one found is the minimal one we find.
                # We can break immediately.
                break

    # If no better rank found, use schoolbook
    U_best, V_best, W_best, rank_best = best_result
    
    # Convert to lists
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()
    
    # Clean up to ensure exact half-integers representation
    def clean_half_int(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                val2 = round(val * 2) / 2.0
                if val2 == int(val2):
                    new_row.append(int(val2))
                else:
                    new_row.append(val2)
            result.append(new_row)
        return result

    U_out = clean_half_int(U_out)
    V_out = clean_half_int(V_out)
    W_out = clean_half_int(W_out)
    
    # Final verification
    U_final = np.array(U_out, dtype=np.float64)
    V_final = np.array(V_out, dtype=np.float64)
    W_final = np.array(W_out, dtype=np.float64)
    
    if not verify(U_final, V_final, W_final):
        # Fallback to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        U_out = clean_half_int(U_out)
        V_out = clean_half_int(V_out)
        W_out = clean_half_int(W_out)

    return {"U": U_out, "V": V_out, "W": W_out}
