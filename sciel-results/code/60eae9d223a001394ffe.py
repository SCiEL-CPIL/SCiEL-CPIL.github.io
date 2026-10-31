import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 4
    
    # Strassen coefficients for 2x2
    beta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    gamma = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    delta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    rank_49 = 49
    U49 = np.zeros((n*m, rank_49))
    V49 = np.zeros((m*p, rank_49))
    W49 = np.zeros((p*n, rank_49))
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * beta[t][i_prime][j_prime]
                    if coeff != 0:
                        U49[row_u, col] = coeff
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * gamma[t][j_prime][k_prime]
                    if coeff != 0:
                        V49[row_v, col] = coeff
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * delta[t][i_prime][k_prime]
                    if coeff != 0:
                        W49[row_w, col] = coeff
                        
    # Verify 49-rank decomposition
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U49[i*4+j] * V49[j*4+k] * W49[k*4+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    
    if max_err > 1e-10:
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
        
    if time.time() > deadline - 20:
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
        
    # Attempt to reduce rank to 48
    # Flatten the 49 rank-1 tensors into a 4096 x 49 matrix
    # Each column r is the vectorization of u_r ⊗ v_r ⊗ w_r
    # u_r is 16x1, v_r is 16x1, w_r is 16x1
    # Tensor is 16x16x16, flattened to 4096x1
    
    T_flat = np.zeros((4096, 49))
    for r in range(49):
        # u_r is column r of U49
        # v_r is column r of V49
        # w_r is column r of W49
        # Tensor T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
        # Flatten: index = (i*16 + j)*16 + k
        # We need to map U, V, W indices correctly.
        # U index: i*4+j
        # V index: j*4+k
        # W index: k*4+i
        
        # Let's construct the tensor explicitly for each r
        # This is expensive but 49 is small.
        tensor_r = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_val = U49[i*4+j, r]
                    v_val = V49[j*4+k, r]
                    w_val = W49[k*4+i, r]
                    tensor_r[i*4+j, j*4+k, k*4+i] = u_val * v_val * w_val
        T_flat[:, r] = tensor_r.flatten()
        
    # Check rank of T_flat
    # If rank is 48, then one column is in the span of the others.
    # We can try to eliminate the last column (r=48)
    # Solve T_flat[:, :48] * c = T_flat[:, 48]
    # If a solution exists, then v_48 = sum c_i v_i
    # Then T = sum_{i=0}^{47} v_i + v_48 = sum_{i=0}^{47} (1+c_i) v_i
    
    try:
        # Use least squares to find c
        c, residuals, rank, sv = np.linalg.lstsq(T_flat[:, :48], T_flat[:, 48], rcond=None)
        
        # Check if the fit is good
        reconstructed = T_flat[:, :48] @ c
        err = np.linalg.norm(T_flat[:, 48] - reconstructed)
        
        if err < 1e-10:
            # We can eliminate column 48
            # New coefficients for columns 0..47 are 1 + c
            new_U = U49[:, :48].copy()
            new_V = V49[:, :48].copy()
            new_W = W49[:, :48].copy()
            
            # Update the coefficients
            # The term for column i is now (1 + c[i]) * (u_i ⊗ v_i ⊗ w_i)
            # We can scale U, V, W by distributing the factor (1+c[i])
            # Since we need integer/half-integer entries, we should be careful.
            # c[i] might be fractional.
            # However, the problem allows half-integers.
            # Let's just scale U by (1+c[i])
            for i in range(48):
                factor = 1.0 + c[i]
                new_U[:, i] *= factor
                # V and W remain unchanged?
                # T_i = u_i ⊗ v_i ⊗ w_i
                # New term is (1+c_i) T_i = ((1+c_i)u_i) ⊗ v_i ⊗ w_i
                # So we can just scale U.
                
            # Verify the new 48-rank decomposition
            max_err_new = 0.0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = np.sum(new_U[i*4+j] * new_V[j*4+k] * new_W[k*4+i])
                        err = abs(val - 1.0)
                        if err > max_err_new:
                            max_err_new = err
                            
            if max_err_new < 1e-10:
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
                
    except Exception:
        pass
        
    return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
