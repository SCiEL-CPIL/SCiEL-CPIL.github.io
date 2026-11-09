"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    start_time = time.time()
    DEATHLINE = 145.0  # Safety margin before 150s kill

    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Schoolbook rank-60 decomposition."""
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition of matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank_decomposition(target_rank, max_iterations=50, seed=42):
        """
        Try to find a rank-`target_rank` decomposition by solving
        the bilinear system via iterative optimization.
        
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        We fix two matrices and solve for the third, iterating.
        """
        if elapsed() > DEATHLINE - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        best_residual = float('inf')
        best_solution = None
        
        for iteration in range(max_iterations):
            if elapsed() > DEATHLINE - 5:
                break
            
            # Fix U and V, solve for W
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For each (i,j,k): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # This is linear in W!
            
            # Build the system for W
            # For each output index (k*n+i), we have equations from all (i,j,k)
            # Let's build A_W * W_flat = b_W
            
            num_w = p * n
            num_u = n * m
            num_v = m * p
            
            # For each (i,j,k), the coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
            # Group by (k*n+i): for fixed i,k, sum over j of (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            
            # Actually, let me think more carefully.
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # 
            # Fix i,k: sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = n (sum of j from 0 to m-1, each = 1)
            # Wait no, each equation is separate. Let me build the full linear system.
            
            # Number of equations: n * m * p
            # Unknowns: p * n * target_rank (W flattened)
            
            num_eq = n * m * p
            num_unknowns_w = num_w * target_rank
            
            # Build A matrix where each row corresponds to one (i,j,k) equation
            # and each column corresponds to one W[k*n+i, r] unknown
            
            A_W = np.zeros((num_eq, num_unknowns_w))
            b_W = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_base = k * n + i
                        for r in range(target_rank):
                            A_W[eq_idx, w_base * target_rank + r] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve least squares for W
            try:
                W_flat, residuals, rank_w, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_new = W_flat.reshape(num_w, target_rank)
                residual = np.sum((A_W @ W_flat - b_W) ** 2)
            except:
                W_new = W
                residual = best_residual
            
            W = W_new
            best_residual = min(best_residual, residual)
            
            # Fix V and W, solve for U
            num_unknowns_u = num_u * target_rank
            A_U = np.zeros((num_eq, num_unknowns_u))
            b_U = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_base = i * m + j
                        for r in range(target_rank):
                            A_U[eq_idx, u_base * target_rank + r] = V[j * p + k, r] * W[k * n + i, r]
                        eq_idx += 1
            
            try:
                U_flat, residuals, rank_u, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_new = U_flat.reshape(num_u, target_rank)
                residual_u = np.sum((A_U @ U_flat - b_U) ** 2)
            except:
                U_new = U
                residual_u = best_residual
            
            U = U_new
            best_residual = min(best_residual, residual_u)
            
            # Fix U and W, solve for V
            num_unknowns_v = num_v * target_rank
            A_V = np.zeros((num_eq, num_unknowns_v))
            b_V = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        v_base = j * p + k
                        for r in range(target_rank):
                            A_V[eq_idx, v_base * target_rank + r] = U[i * m + j, r] * W[k * n + i, r]
                        eq_idx += 1
            
            try:
                V_flat, residuals, rank_v, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                V_new = V_flat.reshape(num_v, target_rank)
                residual_v = np.sum((A_V @ V_flat - b_V) ** 2)
            except:
                V_new = V
                residual_v = best_residual
            
            V = V_new
            best_residual = min(best_residual, residual_v)
            
            # Check convergence
            if best_residual < 1e-10:
                # Verify and return
                U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
                
                if verify(U_list, V_list, W_list, tol=1e-8):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with the trivial decomposition
    best = build_trivial()
    best_rank = n * m * p

    if elapsed() > DEATHLINE - 5:
        return best

    # Try to find lower rank decompositions
    # Start from just below 60 and go down
    # The theoretical minimum is at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But realistically, we might find something around 40-55
    
    # Try various target ranks with different seeds
    target_ranks = [55, 50, 45, 40, 35]
    
    for tr in target_ranks:
        if elapsed() > DEATHLINE - 15:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if elapsed() > DEATHLINE - 10:
                break
            
            result = try_rank_decomposition(tr, max_iterations=30, seed=seed + tr)
            
            if result is not None:
                result_rank = len(result["U"][0])
                if result_rank < best_rank:
                    best = result
                    best_rank = result_rank
                    break  # Found a valid decomposition at this rank, try lower
        
        if elapsed() > DEATHLINE - 10:
            break
    
    # Final verification
    if best is not None:
        if not verify(best["U"], best["V"], best["W"], tol=1e-8):
            # Fallback to trivial
            best = build_trivial()
            best_rank = n * m * p
    
    return best
# EVOLVE-BLOCK-END
