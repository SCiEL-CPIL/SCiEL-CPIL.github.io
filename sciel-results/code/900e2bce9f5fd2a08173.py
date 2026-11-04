import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    DEADLINE = start_time + 135

    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_mat[:, r].reshape(n, m)
            v_r = V_mat[:, r].reshape(m, p)
            w_r = W_mat[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    # Try to find rank-13 decomposition for 2x4x2 via gradient descent with multiple restarts
    def find_rank13_2x4x2():
        rank = 13
        target = np.ones(16)
        
        def compute_tensor(U, V, W):
            result = np.zeros(16)
            for r in range(rank):
                u_r = U[:, r].reshape(2, 4)
                v_r = V[:, r].reshape(4, 2)
                w_r = W[:, r].reshape(2, 2)
                for i in range(2):
                    for j in range(4):
                        for k in range(2):
                            result[i*8 + j*2 + k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            return result
        
        def compute_error(U, V, W):
            t = compute_tensor(U, V, W)
            return np.sum((t - target)**2)
        
        def compute_grads(U, V, W):
            t = compute_tensor(U, V, W)
            err = t - target
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                u_r = U[:, r].reshape(2, 4)
                v_r = V[:, r].reshape(4, 2)
                w_r = W[:, r].reshape(2, 2)
                
                for i in range(2):
                    for j in range(4):
                        for k in range(2):
                            idx = i*8 + j*2 + k
                            e = err[idx]
                            grad_U[i*4+j, r] += e * v_r[j, k] * w_r[k, i]
                            grad_V[j*2+k, r] += e * u_r[i, j] * w_r[k, i]
                            grad_W[k*2+i, r] += e * u_r[i, j] * v_r[j, k]
            
            return grad_U, grad_V, grad_W
        
        best_result = None
        best_error = float('inf')
        
        # Multiple random restarts with different seeds and scales
        for seed in [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666]:
            if time.time() > DEADLINE - 45:
                break
            
            np.random.seed(seed)
            scale = 0.3 + 0.4 * np.random.rand()
            U = np.random.randn(8, rank) * scale
            V = np.random.randn(8, rank) * scale
            W = np.random.randn(4, rank) * scale
            
            lr = 0.05
            best_error_this = compute_error(U, V, W)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            stuck_count = 0
            
            for iteration in range(2000):
                if time.time() > DEADLINE - 45:
                    break
                
                grad_U, grad_V, grad_W = compute_grads(U, V, W)
                
                # Adaptive step size with backtracking
                for _ in range(10):
                    U_new = U - lr * grad_U
                    V_new = V - lr * grad_V
                    W_new = W - lr * grad_W
                    new_error = compute_error(U_new, V_new, W_new)
                    if new_error < best_error_this:
                        U, V, W = U_new, V_new, W_new
                        best_error_this = new_error
                        if new_error < 1e-14:
                            break
                        stuck_count = 0
                        break
                    else:
                        lr *= 0.7
                
                if new_error < 1e-14:
                    break
                
                if lr < 1e-10:
                    break
                
                # Periodic restart of learning rate if stuck
                if stuck_count > 50:
                    lr = max(lr, 0.01)
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if best_error_this < best_error:
                    best_error = best_error_this
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-12:
                        break
            
            if best_error < 1e-12:
                break
        
        final_error = compute_error(best_U, best_V, best_W)
        if final_error < 1e-8:
            return best_U, best_V, best_W
        
        # Final polish pass with very small steps if close
        if final_error < 1e-4:
            lr = 1e-6
            for _ in range(500):
                if time.time() > DEADLINE - 40:
                    break
                grad_U, grad_V, grad_W = compute_grads(best_U, best_V, best_W)
                best_U -= lr * grad_U
                best_V -= lr * grad_V
                best_W -= lr * grad_W
                final_error = compute_error(best_U, best_V, best_W)
                if final_error < 1e-12:
                    break
        
        if final_error < 1e-8:
            return best_U, best_V, best_W
        return None

    result_13 = find_rank13_2x4x2()
    
    if result_13 is not None and time.time() < DEADLINE - 20:
        U13, V13, W13 = result_13
        U52 = np.zeros((8, 52))
        V52 = np.zeros((32, 52))
        W52 = np.zeros((16, 52))
        
        for block in range(4):
            U52[:, block*13:(block+1)*13] = U13
            for j in range(4):
                for k_local in range(2):
                    V52[j*8 + block*2 + k_local, block*13:(block+1)*13] = V13[j*2 + k_local]
            for k_local in range(2):
                for i in range(2):
                    W52[(block*2 + k_local)*2 + i, block*13:(block+1)*13] = W13[k_local*2 + i]
        
        # Round to nearest half-integer to get exact integer/half-integer entries
        U52 = np.round(U52 * 2) / 2
        V52 = np.round(V52 * 2) / 2
        W52 = np.round(W52 * 2) / 2
        
        if verify(U52, V52, W52):
            return {
                "U": U52.tolist(),
                "V": V52.tolist(),
                "W": W52.tolist()
            }
        
        # If rounding broke it, try without rounding
        U52 = np.zeros((8, 52))
        V52 = np.zeros((32, 52))
        W52 = np.zeros((16, 52))
        
        for block in range(4):
            U52[:, block*13:(block+1)*13] = U13
            for j in range(4):
                for k_local in range(2):
                    V52[j*8 + block*2 + k_local, block*13:(block+1)*13] = V13[j*2 + k_local]
            for k_local in range(2):
                for i in range(2):
                    W52[(block*2 + k_local)*2 + i, block*13:(block+1)*13] = W13[k_local*2 + i]
        
        if verify(U52, V52, W52):
            return {
                "U": U52.tolist(),
                "V": V52.tolist(),
                "W": W52.tolist()
            }

    # Fallback: rank-56 Strassen decomposition
    rank_56 = 56
    U = np.zeros((8, rank_56))
    V = np.zeros((32, rank_56))
    W = np.zeros((16, rank_56))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    if not verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
