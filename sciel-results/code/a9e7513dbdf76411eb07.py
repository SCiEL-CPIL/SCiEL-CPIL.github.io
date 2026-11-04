    def construct():
        # ... setup ...
        # Rank 17 solution ...
        
        best_rank = 17
        best_U, best_V, best_W = U17, V17, W17
        
        def check_exact(U, V, W):
            # Snap to half integers
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            
            # Convert to integers
            Ui = np.rint(U_s * 2).astype(int)
            Vi = np.rint(V_s * 2).astype(int)
            Wi = np.rint(W_s * 2).astype(int)
            
            # Check if sum is 8
            # Reshape
            U_t = Ui.reshape(2, 3, -1)
            V_t = Vi.reshape(3, 3, -1)
            W_t = Wi.reshape(3, 2, -1)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            if np.all(prod == 8):
                return True, U_s, V_s, W_s
            return False, None, None

        for target_rank in [16, 15]:
            # ... random restarts ...
            for seed in range(100): # More restarts possible due to speed
                if time.time() > deadline: break
                np.random.seed(seed)
                U = np.random.randn(2*3, target_rank) * 0.5
                V = np.random.randn(3*3, target_rank) * 0.5
                W = np.random.randn(3*2, target_rank) * 0.5
                
                # Optimize
                lr = 0.1
                for step in range(1000):
                    if time.time() > deadline: break
                    
                    # Compute error/gradients
                    U_t = U.reshape(2, 3, target_rank)
                    V_t = V.reshape(3, 3, target_rank)
                    W_t = W.reshape(3, 2, target_rank)
                    
                    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    res = prod - 1.0
                    err = np.sum(res**2)
                    
                    if err < 1e-6:
                        ok, Us, Vs, Ws = check_exact(U, V, W)
                        if ok:
                            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    
                    dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
                    dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
                    dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
                    
                    U -= lr * dU.reshape(6, target_rank)
                    V -= lr * dV.reshape(9, target_rank)
                    W -= lr * dW.reshape(6, target_rank)
                    
                    # Snap periodically
                    if step % 10 == 0:
                        ok, Us, Vs, Ws = check_exact(U, V, W)
                        if ok:
                            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    