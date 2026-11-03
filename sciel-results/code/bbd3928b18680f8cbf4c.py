import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    def get_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
            
        # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
        for block in range(4):
            # 1. (a00+a11)(b00+b11)
            U[1][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 2. (a10+a11)b00
            U[4][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            W[(block*2)*2 + 1][col] = 1
            W[(block*2+1)*2 + 1][col] = -1
            col += 1
            # 3. a00(b01-b11)
            U[1][col] = 1
            V[1*8 + block*2+1][col] = 1
            V[2*8 + block*2+1][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 4. a11(b10-b00)
            U[5][col] = 1
            V[2*8 + block*2][col] = 1
            V[1*8 + block*2][col] = -1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2)*2 + 1][col] = 1
            col += 1
            # 5. (a00+a01)b11
            U[1][col] = 1
            U[2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            col += 1
            # 6. (a10-a00)(b00+b01)
            U[4][col] = 1
            U[1][col] = -1
            V[1*8 + block*2][col] = 1
            V[1*8 + block*2+1][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 7. (a01-a11)(b10+b11)
            U[2][col] = 1
            U[5][col] = -1
            V[2*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            col += 1
        return {"U": U, "V": V, "W": W}

    # Build target tensor
    target_tensor = np.zeros((6, 24, 16))
    for i in range(2):
        for j in range(3):
            for k in range(8):
                u_idx = i * 3 + j
                v_idx = j * 8 + k
                w_idx = k * 2 + i
                target_tensor[u_idx, v_idx, w_idx] = 1.0

    deadline = time.time() + 100  # Leave 20s margin for safety

    def try_rank_43():
        target_rank = 43
        sol_44 = get_rank_44()
        U44 = np.array(sol_44['U']).T  # 6 x 44
        V44 = np.array(sol_44['V']).T  # 24 x 44
        W44 = np.array(sol_44['W']).T  # 16 x 44

        # Try dropping different terms and optimizing
        # The Strassen terms come in groups of 7 per block.
        # Try dropping one term from each block's set to see if optimization can recover.
        
        # Strategy: try multiple initializations by dropping different terms
        # and using random perturbations
        
        import scipy.optimize as opt
        
        def residuals(params):
            U = params[:target_rank * 6].reshape(target_rank, 6).T
            V = params[target_rank * 6 : target_rank * 30].reshape(target_rank, 24).T
            W = params[target_rank * 30 :].reshape(target_rank, 16).T
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target_tensor).flatten()

        # Try dropping each of the 44 terms individually
        best_result = None
        best_cost = np.inf
        
        for drop_idx in range(44):
            if time.time() > deadline:
                break
                
            # Create initial guess by dropping term drop_idx
            mask = np.ones(44, dtype=bool)
            mask[drop_idx] = False
            U_init = U44[:, mask]
            V_init = V44[:, mask]
            W_init = W44[:, mask]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residuals, x0, method='trf', 
                                           max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    best_result = result
                    if cost < 1e-8:
                        break
            except Exception:
                continue
        
        if best_result is not None and best_result.cost < 1e-6:
            x = best_result.x
            U = x[:target_rank * 6].reshape(target_rank, 6).T
            V = x[target_rank * 6 : target_rank * 30].reshape(target_rank, 24).T
            W = x[target_rank * 30 :].reshape(target_rank, 16).T
            
            # Verify
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.linalg.norm(T - target_tensor) < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Attempt rank 43
    result_43 = try_rank_43()
    if result_43 is not None:
        return result_43
    
    # Fallback to rank 44
    return get_rank_44()
