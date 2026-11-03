import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor for 2x3x2 block
    target_232 = np.zeros((6, 6, 4))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                target_232[i*3+j, j*2+k, k*2+i] = 1.0

    deadline = time.time() + 140
    
    # Try to find rank-4 decomposition of 2x3x2 tensor
    target_rank = 4
    best_result = None
    best_norm = np.inf
    
    # Multiple random initializations to find rank-4 solution
    np.random.seed(42)
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        U_init = np.random.randn(6, target_rank) * 0.5
        V_init = np.random.randn(6, target_rank) * 0.5
        W_init = np.random.randn(4, target_rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[:target_rank * 6].reshape(target_rank, 6).T
            V = params[target_rank * 6 : target_rank * 12].reshape(target_rank, 6).T
            W = params[target_rank * 12 :].reshape(target_rank, 4).T
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target_232).flatten()
        
        try:
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=2000, 
                                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if np.linalg.norm(result.fun) < best_norm:
                best_norm = np.linalg.norm(result.fun)
                best_result = result
                if best_norm < 1e-10:
                    break
        except:
            continue
    
    if best_result is not None and best_norm < 1e-10:
        # Found rank-4 solution for 2x3x2 block
        U_block = best_result.x[:target_rank * 6].reshape(target_rank, 6).T
        V_block = best_result.x[target_rank * 6 : target_rank * 12].reshape(target_rank, 6).T
        W_block = best_result.x[target_rank * 12 :].reshape(target_rank, 4).T
        
        # Construct full solution: 4 blocks (for k=0..7, grouped as 2x2 blocks)
        # Total rank = 4 * 4 = 16
        U_full = np.zeros((6, 16))
        V_full = np.zeros((24, 16))
        W_full = np.zeros((16, 16))
        
        for b in range(4):
            # Block b corresponds to B columns 2b, 2b+1
            U_full[:, b*target_rank:(b+1)*target_rank] = U_block
            
            for r in range(target_rank):
                for j in range(3):
                    for k_local in range(2):
                        v_idx_block = j*2 + k_local
                        v_idx_full = j*8 + 2*b + k_local
                        V_full[v_idx_full, b*target_rank + r] = V_block[v_idx_block, r]
            
            for r in range(target_rank):
                for k_local in range(2):
                    for i in range(2):
                        w_idx_block = k_local*2 + i
                        w_idx_full = (2*b + k_local)*2 + i
                        W_full[w_idx_full, b*target_rank + r] = W_block[w_idx_block, r]
        
        # Verify the full solution
        T_full = np.einsum('ur,vr,wr->uvw', U_full, V_full, W_full)
        # Check against target
        target_full = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    target_full[i*3+j, j*8+k, k*2+i] = 1.0
        
        if not np.allclose(T_full, target_full, atol=1e-8):
            # Fallback to rank 44
            return _get_rank_44()
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to rank 44
        return _get_rank_44()

def _get_rank_44():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8
    for block in range(4):
        b00 = 1*8 + block*2
        b01 = 1*8 + block*2 + 1
        b10 = 2*8 + block*2
        b11 = 2*8 + block*2 + 1
        
        U[1][col] = 1; U[5][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        U[4][col] = 1; U[5][col] = 1
        V[b00][col] = 1
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        U[1][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2)*2 + 1][col] = 1
        col += 1
        
        U[1][col] = 1; U[2][col] = 1
        V[b11][col] = 1
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        U[4][col] = 1; U[1][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        U[2][col] = 1; U[5][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
