import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
        n, m, p = DIMENSIONS
        start_time = time.time()
        deadline = start_time + 110  # 10s buffer
        
        nm, mp, pn = n*m, m*p, p*n
        
        # Build Target Tensor (Full)
        T_full = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_full[a, b, c] = 1.0

        # Schoolbook Decomposition (Rank 60)
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1

        def round_to_half(x):
            return np.round(x * 2) / 2

        def verify_decomposition(U, V, W, T, tol=1e-6):
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(pred - T))
            return err < tol

        def compute_loss_grad(U, V, W, T):
            r = U.shape[1]
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            error = pred - T
            loss = np.sum(error**2)
            
            dU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            dV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            dW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            
            return loss, dU, dV, dW

        def adam_step(params, grads, m_params, v_params, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
            updated = []
            for i in range(len(params)):
                p = params[i]
                g = grads[i]
                m_p = m_params[i]
                v_p = v_params[i]
                
                m_p[:] = b1 * m_p + (1 - b1) * g
                v_p[:] = b2 * v_p + (1 - b2) * (g * g)
                
                m_hat = m_p / (1 - b1**t)
                v_hat = v_p / (1 - b2**t)
                
                p_new = p - lr * m_hat / (np.sqrt(v_hat) + eps)
                updated.append(p_new)
            return updated

        def try_rank_sub(r, seed, T_sub, nm_sub, mp_sub, pn_sub, max_iter=2000):
            np.random.seed(seed)
            
            U = np.random.randn(nm_sub, r) * 0.5
            V = np.random.randn(mp_sub, r) * 0.5
            W = np.random.randn(pn_sub, r) * 0.5
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            t = 0
            best_loss = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                loss, dU, dV, dW = compute_loss_grad(U, V, W, T_sub)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if loss < 1e-8:
                        return best_U, best_V, best_W, loss
                
                lr = 0.01 / (1 + 0.01 * it)
                t += 1
                U, V, W = adam_step([U, V, W], [dU, dV, dW], 
                                     [m_U, m_V, m_W], [v_U, v_V, v_W], t, lr=lr)
                
                if it % 100 == 0 and loss > 1e-6:
                     U += np.random.randn(*U.shape) * 0.01
                     V += np.random.randn(*V.shape) * 0.01
                     W += np.random.randn(*W.shape) * 0.01

            if best_loss < 1e-5:
                return best_U, best_V, best_W, best_loss
            return None

        # Divide and Conquer Strategy
        # Split inner dimension m=4 into two parts of size 2
        m_sub = 2
        n_sub, p_sub = n, p
        nm_sub = n_sub * m_sub
        mp_sub = m_sub * p_sub
        pn_sub = p_sub * n_sub
        
        T_sub = np.zeros((nm_sub, mp_sub, pn_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1.0

        # Try to find low rank for sub-tensor
        found_sub = None
        for r_sub in [29, 28, 27, 26]:
            if time.time() > deadline - 20:
                break
            for seed in range(10):
                if time.time() > deadline - 5:
                    break
                res = try_rank_sub(r_sub, seed, T_sub, nm_sub, mp_sub, pn_sub)
                if res is not None:
                    U_sub, V_sub, W_sub, loss = res
                    # Round to half-integers
                    U_sub_r = round_to_half(U_sub)
                    V_sub_r = round_to_half(V_sub)
                    W_sub_r = round_to_half(W_sub)
                    
                    if verify_decomposition(U_sub_r, V_sub_r, W_sub_r, T_sub):
                        found_sub = (U_sub_r, V_sub_r, W_sub_r, r_sub)
                        break
            if found_sub:
                break
        
        if found_sub:
            U_sub_r, V_sub_r, W_sub_r, r_sub = found_sub
            
            # Construct Full Decomposition
            rank_full = 2 * r_sub
            U_full = np.zeros((nm, rank_full))
            V_full = np.zeros((mp, rank_full))
            W_full = np.zeros((pn, rank_full))
            
            # Fill U_full
            # First block (j=0,1)
            for i in range(n):
                for j in range(m_sub):
                    U_full[i*m + j, :r_sub] = U_sub_r[i*m_sub + j, :]
            # Second block (j=2,3)
            for i in range(n):
                for j in range(m_sub):
                    U_full[i*m + (j + m_sub), r_sub:] = U_sub_r[i*m_sub + j, :]
            
            # Fill V_full
            # First block (j=0,1)
            for j in range(m_sub):
                for k in range(p):
                    V_full[j*p + k, :r_sub] = V_sub_r[j*p_sub + k, :]
            # Second block (j=2,3)
            for j in range(m_sub):
                for k in range(p):
                    V_full[(j + m_sub)*p + k, r_sub:] = V_sub_r[j*p_sub + k, :]
            
            # Fill W_full
            # W is shared structure, just copy
            W_full[:, :r_sub] = W_sub_r
            W_full[:, r_sub:] = W_sub_r
            
            # Verify Full
            if verify_decomposition(U_full, V_full, W_full, T_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

        # Fallback to Schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    # EVOLVE-BLOCK-END
