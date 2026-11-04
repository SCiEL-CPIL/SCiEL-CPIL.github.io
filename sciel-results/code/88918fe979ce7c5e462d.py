import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Copy rank 17 solution from the prompt
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17

    def check_exact(U, V, W):
        # Snap to half integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Convert to integers (multiply by 2)
        Ui = np.rint(U_s * 2).astype(int)
        Vi = np.rint(V_s * 2).astype(int)
        Wi = np.rint(W_s * 2).astype(int)
        
        # Reshape
        U_t = Ui.reshape(n, m, -1)
        V_t = Vi.reshape(m, p, -1)
        W_t = Wi.reshape(p, n, -1)
        
        # Compute product: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Should be 8 (since 1/2 * 1/2 * 1/2 = 1/8, and we want 1, so 8 * 1 = 8)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.all(prod == 8):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def compute_grads(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        return dU, dV, dW, np.sum(res**2)

    # Try to find lower rank
    for target_rank in [16, 15]:
        if time.time() > deadline:
            break
            
        # Increase number of seeds for lower ranks as they are harder
        num_seeds = 50 if target_rank == 16 else 100
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Local search / Gradient Descent
            lr = 0.1
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                dU, dV, dW, err = compute_grads(U, V, W, target_rank)
                
                if err < 1e-5:
                    ok, Us, Vs, Ws = check_exact(U, V, W)
                    if ok:
                        return {
                            "U": Us.tolist(),
                            "V": Vs.tolist(),
                            "W": Ws.tolist()
                        }
                
                # Update
                U -= lr * dU.reshape(n * m, target_rank)
                V -= lr * dV.reshape(m * p, target_rank)
                W -= lr * dW.reshape(p * n, target_rank)
                
                # Snap occasionally
                if step % 20 == 0:
                    ok, Us, Vs, Ws = check_exact(U, V, W)
                    if ok:
                        return {
                            "U": Us.tolist(),
                            "V": Vs.tolist(),
                            "W": Ws.tolist()
                        }

    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
