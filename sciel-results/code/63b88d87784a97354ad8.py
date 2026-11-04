import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # The trivial (schoolbook) decomposition has rank n*m*p = 120
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1
    
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor using vectorized einsum."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def compute_tensor(U, V, W):
        """Compute the tensor from U, V, W."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

    def compute_gradients(U, V, W, target):
        """Compute gradients of the loss function."""
        rank = U.shape[1]
        T_curr = compute_tensor(U, V, W)
        err = T_curr - target
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        dU = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
        dV = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
        dW = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
        
        return dU, dV, dW, err

    def adam_update(params, grads, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step."""
        for i in range(len(params)):
            m_t = beta1 * state['m'][i] + (1 - beta1) * grads[i]
            v_t = beta2 * state['v'][i] + (1 - beta2) * grads[i] ** 2
            m_hat = m_t / (1 - beta1 ** (state['t'] + 1))
            v_hat = v_t / (1 - beta2 ** (state['t'] + 1))
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            state['m'][i] = m_t
            state['v'][i] = v_t
        state['t'] += 1

    def try_rank(target_rank, num_attempts=3, max_iters=500):
        """Try to find a decomposition of the given rank."""
        if time.time() > deadline - 20:
            return None
        
        target = np.ones((n, m, p))
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 20:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Random initialization
                np.random.seed(target_rank * 1000 + attempt)
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            elif attempt == 1:
                # Sparse initialization
                np.random.seed(target_rank * 1000 + attempt)
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
                    idx_v = np.random.choice(m * p, size=min(3, m * p), replace=False)
                    idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
                    U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
            else:
                # Perturbed trivial initialization
                np.random.seed(target_rank * 1000 + attempt)
                if target_rank < trivial_rank:
                    # Start from a subset of trivial terms and perturb
                    U = np.zeros((n * m, target_rank))
                    V = np.zeros((m * p, target_rank))
                    W = np.zeros((p * n, target_rank))
                    for r in range(target_rank):
                        i = r // (m * p) % n
                        j = (r // p) % m
                        k = r % p
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] = 1.0 + np.random.randn() * 0.1
                        V[b, r] = 1.0 + np.random.randn() * 0.1
                        W[c, r] = 1.0 + np.random.randn() * 0.1
                else:
                    U = U_trivial[:, :target_rank].copy()
                    V = V_trivial[:, :target_rank].copy()
                    W = W_trivial[:, :target_rank].copy()
                    U += np.random.randn(n * m, target_rank) * 0.05
                    V += np.random.randn(m * p, target_rank) * 0.05
                    W += np.random.randn(p * n, target_rank) * 0.05
            
            # Adam optimizer state
            state = {
                'm': [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)],
                'v': [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)],
                't': 0
            }
            params = [U, V, W]
            
            best_loss = np.inf
            best_params = [U.copy(), V.copy(), W.copy()]
            patience = 0
            max_patience = 100
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                dU, dV, dW, err = compute_gradients(U, V, W, target)
                grads = [dU, dV, dW]
                
                loss = np.sum(err ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = [U.copy(), V.copy(), W.copy()]
                    patience = 0
                else:
                    patience += 1
                
                if patience > max_patience:
                    break
                
                # Check convergence
                if loss < 1e-10:
                    if check_decomposition(U, V, W):
                        return U.copy(), V.copy(), W.copy()
                    break
                
                # Adam update
                params = [U, V, W]
                for i in range(3):
                    m_t = 0.9 * state['m'][i] + 0.1 * grads[i]
                    v_t = 0.999 * state['v'][i] + 0.001 * grads[i] ** 2
                    m_hat = m_t / (1 - 0.9 ** (state['t'] + 1))
                    v_hat = v_t / (1 - 0.999 ** (state['t'] + 1))
                    params[i] -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
                    state['m'][i] = m_t
                    state['v'][i] = v_t
                state['t'] += 1
                U, V, W = params
        
        # Final check with best found
        if best_loss < 1e-8:
            U, V, W = best_params
            if check_decomposition(U, V, W):
                return U, V, W
        
        return None

    # Try ranks from 119 down to 100
    for target_rank in range(119, 99, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank, num_attempts=3, max_iters=300)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break  # Found a valid decomposition, stop searching
    
    if best_rank == trivial_rank:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
    else:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
