import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Build the list of non-zero entries in the target tensor
    # T[a, b, c] = 1 if valid, else 0
    # We only need to store the non-zero ones for efficiency
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c, 1.0))
                
    num_ones = len(target_entries)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    school_result = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    # Try to find a lower rank solution
    # We search for integer/half-integer solutions.
    # We will optimize real values and then snap.
    
    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        # And if they reconstruct the tensor exactly
        # Snap to half-integers first
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if they are exactly half-integers
        if not np.allclose(U, U_s, atol=1e-9):
            return False
        if not np.allclose(V, V_s, atol=1e-9):
            return False
        if not np.allclose(W, W_s, atol=1e-9):
            return False
            
        # Verify reconstruction
        # We only need to check the target entries and a sample of zeros
        # To be safe, we check all target entries are 1.
        # And we check that the sum of squares of the error is 0.
        
        # Efficient check:
        # Compute the tensor product for the target entries
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # This is slow for dense U, V, W.
        # We can use einsum
        # T_recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        # But this creates a 9x15x15 array. Size 2025. This is fine.
        
        T_recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s, optimize=True)
        
        # Check target entries
        for a, b, c, val in target_entries:
            if not np.isclose(T_recon[a, b, c], val, atol=1e-9):
                return False
                
        # Check a random sample of zero entries
        # Total entries 2025. Target 45. Zeros 1980.
        # We check 100 random zeros.
        indices_a = np.random.randint(0, n * m, 100)
        indices_b = np.random.randint(0, m * p, 100)
        indices_c = np.random.randint(0, p * n, 100)
        
        # Ensure these are not target entries?
        # It's rare to hit a target entry by chance.
        # If we do, the value should be 1, not 0.
        # So we just check if the value is 0 or 1?
        # No, if it's a zero entry, it must be 0.
        # If it's a target entry, it must be 1.
        # We don't know which is which for the random sample easily.
        # Better: Construct the full target tensor T and compare.
        
        T_target = np.zeros((n * m, m * p, p * n))
        for a, b, c, val in target_entries:
            T_target[a, b, c] = val
            
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return False
            
        return True

    def try_rank(R, num_restarts=5):
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            # Initialize randomly
            # Scale: entries should be around 1/sqrt(R) to keep sum ~ 1
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Optimize
            # We minimize sum (sum_r U V W - T)^2
            # This is a non-convex problem.
            # We use a simple gradient descent step.
            
            # To speed up, we can use scipy.optimize.minimize with L-BFGS-B
            # But defining the gradient is needed.
            # Let's try a few iterations of a custom update.
            
            # Actually, for small R, we can just try to snap the random init?
            # No, random init won't work.
            
            # Let's use scipy.optimize.least_squares
            # Variables: flatten U, V, W
            # Residuals: T_recon - T_target
            
            def residuals(params):
                U_flat = params[:n * m * R]
                V_flat = params[n * m * R : n * m * R + m * p * R]
                W_flat = params[n * m * R + m * p * R :]
                
                U_r = U_flat.reshape(n * m, R)
                V_r = V_flat.reshape(m * p, R)
                W_r = W_flat.reshape(p * n, R)
                
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                # Flatten T_recon and T_target
                return (T_recon - T_target).flatten()
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = scipy.optimize.least_squares(residuals, x0, max_nfev=1000, ftol=1e-12, xtol=1e-12)
            except Exception:
                continue
                
            if res.success or np.linalg.norm(res.fun) < 1e-8:
                U_sol = res.x[:n * m * R].reshape(n * m, R)
                V_sol = res.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                W_sol = res.x[n * m * R + m * p * R :].reshape(p * n, R)
                
                if verify_solution(U_sol, V_sol, W_sol):
                    return {"U": U_sol, "V": V_sol, "W": W_sol}
                    
        return None

    import scipy.optimize
    
    # Try ranks from low to high? Or high to low?
    # Low ranks are harder to find. High ranks are easier.
    # We want the lowest rank.
    # Let's try a range.
    
    # Theoretical lower bound is 15.
    # Schoolbook is 45.
    # Let's try 40, 35, 30, 25, 20, 15.
    
    ranks_to_try = [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 15]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        print(f"Trying rank {R}", file=sys.stderr)
        sol = try_rank(R, num_restarts=3)
        if sol:
            print(f"Found solution for rank {R}", file=sys.stderr)
            return sol
            
    # If nothing found, return schoolbook
    return school_result

