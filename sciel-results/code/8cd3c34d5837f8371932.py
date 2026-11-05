import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18
    
    # Construct target tensor T_target
    # T_target[a, b, c] = 1 if a,b,c form a valid multiplication term (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        # Check if R matches T_target exactly
        if np.allclose(R, T_target, atol=1e-9):
            # Convert to lists
            U_list = U_s.tolist()
            V_list = V_s.tolist()
            W_list = W_s.tolist()
            return True, U_list, V_list, W_list
        return False, None, None, None

    # Strategy 1: Direct search for low rank
    # The known lower bound for 3x3x6 is 47? Or maybe 48?
    # Schoolbook is 54.
    # Let's try ranks 48 to 53.
    
    # To make search faster and more robust, we can use the structure.
    # But let's stick to the random init + LBFGS approach but with better params.
    # We will try multiple random inits for each rank.
    
    # Improvement: Use a "warm start" or better initialization?
    # Maybe initialize with schoolbook and add noise?
    # Or use the fact that for 3x3x3, rank 23 is known.
    # 3x3x6 can be seen as two 3x3x3 blocks (k=0..2 and k=3..5) but with shared U.
    # Actually, U is 9xR.
    # For k=0..2, the terms are (i, j, k).
    # For k=3..5, the terms are (i, j, k).
    # The U indices are same for both blocks.
    # The V indices are different (j*p+k).
    # The W indices are different (k*n+i).
    
    # If we had a rank-23 decomposition for 3x3x3:
    # U_333 (9x23), V_333 (9x23), W_333 (9x23)
    # We can construct a rank-46 decomposition for 3x3x6:
    # U_full (9x46) = [U_333, U_333]
    # V_full (18x46) = 
    #   Rows 0..8 (k=0..2 part): [V_333, 0]
    #   Rows 9..17 (k=3..5 part): [0, V_333]
    # W_full (18x46) =
    #   Rows 0..8 (k=0..2 part): [W_333, 0]  -- Wait, W index is k*n+i.
    #   For k=0,1,2: W indices 0..8.
    #   For k=3,4,5: W indices 9..17.
    #   So Rows 0..8: [W_333, 0]
    #   Rows 9..17: [0, W_333]
    
    # This construction works IF we have a valid rank-23 3x3x3 decomposition.
    # Is rank 23 known for 3x3x3?
    # Yes, Strassen-like constructions or exhaustive search found rank 23?
    # Actually, the best known for 3x3x3 is 23? No, it's 23 for complex?
    # Wait, the rank of 3x3 matrix multiplication over reals is 23?
    # Schoolbook is 27. Strassen is 27? No, Strassen is 7 for 2x2.
    # For 3x3, the rank is 23?
    # If so, then rank 46 for 3x3x6 is a strong candidate.
    
    # However, finding a rank-23 decomposition for 3x3x3 is a non-trivial task in itself.
    # The current program attempts to find a decomposition for 3x3x6 directly.
    # If we can assume we can find or hardcode a 3x3x3 rank-23 solution, we can get 46.
    # But I don't have the hardcoded values.
    
    # Let's check if the direct search can find rank 48 or 50.
    # The previous program tried 46 to 53 with 10 attempts for <48, 5 for >=48.
    # It probably timed out or failed to find anything better than 54.
    
    # Let's improve the search.
    # 1. Use a better objective function? No, sum of squares is standard.
    # 2. Use multiple random seeds.
    # 3. Use `scipy.optimize.minimize` with `Nelder-Mead` or `Powell` as backup?
    #    LBFGS requires gradients. The gradient of sum((R-T)^2) w.r.t U,V,W is easy.
    #    But scipy.optimize.minimize with method='L-BFGS-B' uses numerical gradients if not provided.
    #    We can provide analytic gradients to speed it up.
    
    def obj_and_grad(params):
        U_p = params[:dim_U*rank].reshape(dim_U, rank)
        V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
        
        R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
        diff = R - T_target
        obj_val = np.sum(diff**2)
        
        # Gradients
        # dR/du_ir = V_jr W_kr
        # dL/du_ir = 2 * sum_{j,k} diff[ijk] * V_jr * W_kr
        # This is 2 * (diff * V * W) contracted over j,k for each i,r.
        # grad_U[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
        
        # Using einsum:
        # grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V_p, W_p)
        # grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U_p, W_p)
        # grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U_p, V_p)
        
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V_p, W_p)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U_p, W_p)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U_p, V_p)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj_val, grad

    # Let's try ranks 48 to 52 with analytic gradients.
    # This should be much faster.
    
    best_solution = None
    
    for rank in range(48, 54):
        if time.time() > deadline - 20: break
        
        # Number of attempts
        # More attempts for lower ranks?
        # Or just fixed attempts.
        num_attempts = 20
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10: break
            
            # Random init
            # Scale: entries should be around 1.
            # If we use too small, it might get stuck in 0.
            # If too large, might diverge.
            # Let's try std=1.0
            np.random.seed(1000 + rank * 10 + attempt)
            U = np.random.randn(dim_U, rank)
            V = np.random.randn(dim_V, rank)
            W = np.random.randn(dim_W, rank)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(
                    obj_and_grad,
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                
                if res.success:
                    U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                    V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    
                    # Check objective value
                    if res.fun < 1e-8:
                        success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                        if success:
                            best_solution = {"U": U_s, "V": V_s, "W": W_s}
                            return best_solution
                            
            except Exception as e:
                pass

    # If we didn't find a low rank, try rank 54 (schoolbook)
    # Or maybe rank 50 with more attempts?
    # For now, fallback to schoolbook.
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
