import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    # Known rank 44 decomposition
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
        # Strassen for 4 blocks of 2x2
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

    # Try to find a lower rank decomposition using numerical optimization
    # This is a structural change: using numerical search instead of fixed algebraic construction.
    # We try rank 43.
    target_rank = 43
    deadline = time.time() + 100 # Leave 20s margin
    
    try:
        import scipy.optimize as opt
        # Flatten U, V, W
        # U: 6 x R, V: 24 x R, W: 16 x R
        # Total variables: R * (6 + 24 + 16) = R * 46
        # Equations: 48 (components of the tensor)
        # Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We want T to match the identity multiplication tensor.
        
        # The tensor for matrix multiplication M_{2,3,8}
        # T[i, j, k] = 1 if i, j, k correspond to valid multiplication term A[i,j] B[j,k] -> C[i,k]
        # Indices:
        # U index i: 0..5 (i_row * 3 + i_col)
        # V index j: 0..23 (j_row * 8 + j_col)
        # W index k: 0..15 (k_col * 2 + k_row)
        
        # The tensor T has non-zero entries where j_col == k_col? No.
        # A[i,j] * B[j,k] -> C[i,k]
        # U index: i*3 + j
        # V index: j*8 + k
        # W index: k*2 + i
        
        # So T[U_idx, V_idx, W_idx] = 1 if U_idx%3 == V_idx//8 and V_idx%8 == W_idx//2
        # Let's construct the target tensor
        target_tensor = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    u_idx = i * 3 + j
                    v_idx = j * 8 + k
                    w_idx = k * 2 + i
                    target_tensor[u_idx, v_idx, w_idx] = 1.0

        def residuals(params):
            U = params[:target_rank * 6].reshape(target_rank, 6).T # 6 x R
            V = params[target_rank * 6 : target_rank * 30].reshape(target_rank, 24).T # 24 x R
            W = params[target_rank * 30 :].reshape(target_rank, 16).T # 16 x R
            
            # Compute tensor
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # This is a tensor contraction.
            # We can compute it as:
            # T = np.einsum('ur,vr,wr->uvw', U, V, W)
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            return (T - target_tensor).flatten()

        # Initial guess: Use the rank 44 solution, drop one term, and optimize.
        sol_44 = get_rank_44()
        U44 = np.array(sol_44['U']).T
        V44 = np.array(sol_44['V']).T
        W44 = np.array(sol_44['W']).T
        
        # Drop the last term (index 43)
        U_init = U44[:, :43].flatten()
        V_init = V44[:, :43].flatten()
        W_init = W44[:, :43].flatten()
        
        x0 = np.concatenate([U_init, V_init, W_init])
        
        # Use least_squares
        result = opt.least_squares(residuals, x0, method='trf', max_nfev=1000)
        
        if result.success and np.linalg.norm(result.fun) < 1e-6:
            U = result.x[:target_rank * 6].reshape(target_rank, 6).T
            V = result.x[target_rank * 6 : target_rank * 30].reshape(target_rank, 24).T
            W = result.x[target_rank * 30 :].reshape(target_rank, 16).T
            
            # Convert to list of lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception as e:
        pass
        
    return get_rank_44()
