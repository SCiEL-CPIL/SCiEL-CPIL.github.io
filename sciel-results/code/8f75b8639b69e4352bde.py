"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to find a lower rank decomposition via gradient-based search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)

    # Build the rank-26 baseline
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 7 terms
        terms = [
            # (U_entries, V_entries, W_entries)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        # Standard 19 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to find a lower rank decomposition using gradient descent
    # We minimize || T_target - T_recon ||_F^2
    
    def gradient_descent_search(target_R, initial_U, initial_V, initial_W, max_iters=5000, lr=0.05):
        """
        Attempt to find a valid decomposition of rank target_R 
        by minimizing the error using gradient descent.
        """
        if target_R >= best_R:
            return None
        
        # Initialize with a perturbation of the best known solution
        # We take the first target_R columns of the best solution
        # and add small noise to escape local minima
        np.random.seed(42)
        
        U_cur = initial_U[:, :target_R].copy()
        V_cur = initial_V[:, :target_R].copy()
        W_cur = initial_W[:, :target_R].copy()
        
        # Add small noise
        U_cur += np.random.normal(0, 0.1, U_cur.shape)
        V_cur += np.random.normal(0, 0.1, V_cur.shape)
        W_cur += np.random.normal(0, 0.1, W_cur.shape)
        
        T_target = np.ones((n, m, p))
        
        # Precompute shapes for reshaping
        U_shape = (n, m, target_R)
        V_shape = (m, p, target_R)
        W_shape = (p, n, target_R)
        
        current_error = np.inf
        best_local_U, best_local_V, best_local_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        best_local_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute reconstruction
            U_resh = U_cur.reshape(U_shape)
            V_resh = V_cur.reshape(V_shape)
            W_resh = W_cur.reshape(W_shape)
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_resh, V_resh, W_resh)
            error = np.sum((T_recon - T_target) ** 2)
            
            if error < best_local_error:
                best_local_error = error
                best_local_U, best_local_V, best_local_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                
            if error < 1e-9:
                # Check for exact match
                if verify(U_cur, V_cur, W_cur, target_R):
                    return (U_cur, V_cur, W_cur)
            
            # Compute gradients
            # dT/dU_ijk = sum_r V_jkr * W_kir
            # dT/dV_jkr = sum_r U_ijk * W_kir
            # dT/dW_kir = sum_r U_ijk * V_jkr
            
            # Gradient of error w.r.t U:
            # dE/dU_ijk = 2 * (T_recon - T_target)_ijk * (d T_ijk / d U_ijk)
            # But T_ijk = sum_r U_ijk V_jkr W_kir
            # So d T_ijk / d U_ijk is not direct. 
            # Let's use chain rule via intermediate terms.
            
            # Let X_r = U[:,:,r] * V[:,:,r] * W[:,:,r] elementwise? No, 
            # T_ijk = sum_r U_ijk V_jkr W_kir.
            
            # Gradient w.r.t U_ijk:
            # dE/dU_ijk = sum_k,l (T_recon_ijk - T_target_ijk) * dT_ijk/dU_ijk
            # dT_ijk/dU_ijk = sum_r V_jkr W_kir ... wait, indices.
            # T_ijk = sum_r U_ijk V_jkr W_kir
            # The term involving U_ijk is U_ijk * (sum_r V_jkr W_kir)? No.
            # The sum is over r. For a fixed i,j,k, T_ijk = sum_r U_ijk(r) V_jkr(r) W_kir(r).
            # Wait, U is indexed by (i,j,r). V by (j,k,r). W by (k,i,r).
            # T_ijk = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
            
            # dT_ijk / dU[i,j,r] = V[j,k,r] * W[k,i,r]
            # dE/dU[i,j,r] = sum_i,j,k (T_ijk - T_ijk_target) * dT_ijk/dU[i,j,r]
            # Note that U[i,j,r] only contributes to T[i,j,k] for all k?
            # No, U[i,j,r] is a single scalar. It appears in T[i,j,k] for ALL k?
            # Let's check: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # Yes, U[i,j,r] multiplies V[j,k,r] W[k,i,r] in the sum for T[i,j,k].
            # So dT[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r].
            
            # Therefore:
            # dE/dU[i,j,r] = sum_k (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
            
            # Similarly:
            # dE/dV[j,k,r] = sum_i (T[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
            # dE/dW[k,i,r] = sum_j (T[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
            
            # We can compute these using einsum.
            # diff = T_recon - T_target (shape n,m,p)
            diff = T_recon - T_target
            
            # grad_U[i,j,r] = sum_k diff[i,j,k] * V[j,k,r] * W[k,i,r]
            # einsum: 'ijk,jkr,kir->ijr'
            grad_U = np.einsum('ijk,jkr,kir->ijr', diff, V_resh, W_resh)
            
            # grad_V[j,k,r] = sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
            # einsum: 'ijk,ijr,kir->jkr'
            grad_V = np.einsum('ijk,ijr,kir->jkr', diff, U_resh, W_resh)
            
            # grad_W[k,i,r] = sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
            # einsum: 'ijk,ijr,jkr->kir'
            grad_W = np.einsum('ijk,ijr,jkr->kir', diff, U_resh, V_resh)
            
            # Update
            U_cur -= lr * grad_U
            V_cur -= lr * grad_V
            W_cur -= lr * grad_W
            
            # Adaptive learning rate? Simple decay
            if it % 100 == 99:
                lr *= 0.95
                
        # Check if we found a valid solution
        if best_local_error < 1e-9:
            if verify(best_local_U, best_local_V, best_local_W, target_R):
                return (best_local_U, best_local_V, best_local_W)
                
        return None

    # Try to find rank 25, 24, 23
    for target_R in [25, 24, 23]:
        if time.time() > deadline - 20:
            break
            
        # Try a few different random initializations/noise levels
        for noise_level in [0.01, 0.1, 0.5]:
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + target_R + int(noise_level*100))
            
            # Initialize from best known solution
            U_init = best_U[:, :target_R].copy()
            V_init = best_V[:, :target_R].copy()
            W_init = best_W[:, :target_R].copy()
            
            U_init += np.random.normal(0, noise_level, U_init.shape)
            V_init += np.random.normal(0, noise_level, V_init.shape)
            W_init += np.random.normal(0, noise_level, W_init.shape)
            
            result = gradient_descent_search(target_R, U_init, V_init, W_init)
            
            if result is not None:
                best_U, best_V, best_W, best_R = result[0], result[1], result[2], target_R
                break # Move to next lower rank
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
