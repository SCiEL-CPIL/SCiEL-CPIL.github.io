import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    
    # Precompute the 84 non-zero tensor entries
    # T[i, j, k] = 1 where output index is k*n+i, input1 index is i*m+j, input2 index is j*p+k
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    num_constraints = len(indices) # 84

    def compute_residuals(U, V, W):
        """
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Returns array of shape (num_constraints,)
        residual = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
        """
        residuals = np.zeros(num_constraints)
        for idx, (i, j, k) in enumerate(indices):
            u_val = U[i * m + j, :]
            v_val = V[j * p + k, :]
            w_val = W[k * n + i, :]
            residuals[idx] = np.sum(u_val * v_val * w_val) - 1.0
        return residuals

    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = None
    best_rank = n * m * p
    
    # We will try ranks from a lower bound up to 83.
    # Lower bound for 3x4x7 is not strictly 45, but let's start reasonably low.
    # Actually, let's just try a few ranks. The search space is huge.
    # Given the time limit, we might not find anything better than 84.
    # But we must try.
    
    # Let's try to use the "Strassen-Winograd" inspiration?
    # The prompt suggests: "Use the Strassen-Winograd recursive block decomposition strategy..."
    # This is a specific constructive approach.
    # Let's see if we can construct a solution using 2x2x2 Strassen blocks.
    # 3x4 = [2x4; 1x4]. 4x7 = [4x2 4x5]? No, 4x7.
    # Let's partition 3x4 as A1 (2x4) and A2 (1x4).
    # Let's partition 4x7 as B1 (4x2) and B2 (4x5)? No, we need to match dimensions.
    # C = A * B.
    # C is 3x7.
    # Let's try to decompose C into blocks.
    # This is complex to implement dynamically and guarantee correctness within the time limit.
    # The optimization approach is safer, even if it fails to find a lower rank, it returns schoolbook.
    
    # Let's refine the optimization approach.
    # The previous program used L-BFGS-B.
    # Let's try to initialize with a random linear combination of the schoolbook basis?
    # Or just random.
    
    # To improve chances, let's try a few specific ranks.
    # Rank 60? 50?
    
    # Let's stick to the optimization but make it more robust.
    # We will try ranks: 80, 70, 60, 50.
    
    ranks_to_try = [80, 70, 60, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        num_restarts = 5
        lam = 1e-5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            # Initialization
            if restart == 0:
                # Start from schoolbook-like structure?
                # We can't easily map schoolbook to rank R if R < 84.
                # Random is best.
                x0 = np.random.randn(total_size) * 1.0
            else:
                x0 = np.random.randn(total_size) * 0.5
            
            # Objective function
            def objective(params, R, lam):
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U + size_V]
                W_flat = params[size_U + size_V:]
                
                U = U_flat.reshape((n * m, R))
                V = V_flat.reshape((m * p, R))
                W = W_flat.reshape((p * n, R))
                
                res = compute_residuals(U, V, W)
                ss_res = np.sum(res * res)
                reg = lam * (np.sum(U * U) + np.sum(V * V) + np.sum(W * W))
                return ss_res + reg

            try:
                # Use L-BFGS-B
                # We need to import minimize? The current program does.
                # I should include the import in the file.
                from scipy.optimize import minimize
                
                res_opt = minimize(
                    objective, 
                    x0, 
                    args=(R, lam), 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res_opt.fun < 1e-6:
                U_flat = res_opt.x[:size_U]
                V_flat = res_opt.x[size_U:size_U + size_V]
                W_flat = res_opt.x[size_U + size_V:]
                
                U_cont = U_flat.reshape((n*m, R))
                V_cont = V_flat.reshape((m*p, R))
                W_cont = W_flat.reshape((p*n, R))
                
                # Snap to half-integers
                U_snap = np.round(U_cont * 2) / 2
                V_snap = np.round(V_cont * 2) / 2
                W_snap = np.round(W_cont * 2) / 2
                
                # Verify
                res_snapped = compute_residuals(U_snap, V_snap, W_snap)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    best_rank = R
                    best_solution = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        return schoolbook_solution()
