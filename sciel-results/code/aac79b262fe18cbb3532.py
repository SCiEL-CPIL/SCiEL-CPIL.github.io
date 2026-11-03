import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill

    def _verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition using einsum."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        
        U_3d = U.T.reshape(r, n, m)
        V_3d = V.T.reshape(r, m, p)
        W_3d = W.T.reshape(r, p, n)
        
        T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        
        return np.allclose(T_recon, target, atol=tol)

    def _schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63

    def _loss_and_grad(U_3d, V_3d, W_3d):
        """Compute loss and gradients for tensor decomposition."""
        target = np.ones((n, m, p))
        T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        diff = T_recon - target
        loss = np.sum(diff ** 2)
        
        dU = np.einsum('ijk,rjk,rki->rij', diff, V_3d, W_3d)
        dV = np.einsum('ijk,rij,rki->rjk', diff, U_3d, W_3d)
        dW = np.einsum('ijk,rij,rjk->rki', diff, U_3d, V_3d)
        
        return loss, dU, dV, dW

    def _adam_step(params, grads, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """Simple Adam optimizer step."""
        if 'm' not in state:
            state['m'] = np.zeros_like(params)
            state['v'] = np.zeros_like(params)
            state['t'] = 0
        
        state['t'] += 1
        t = state['t']
        
        state['m'] = beta1 * state['m'] + (1 - beta1) * grads
        state['v'] = beta2 * state['v'] + (1 - beta2) * (grads ** 2)
        
        m_hat = state['m'] / (1 - beta1 ** t)
        v_hat = state['v'] / (1 - beta2 ** t)
        
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params

    def _try_rank(r_target, max_iters=300, n_restarts=8, lr=0.05):
        """Try to find a rank-r decomposition using Adam optimizer."""
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with different strategies
            if restart % 3 == 0:
                # Standard random
                U_3d = np.random.randn(r_target, n, m) * 0.3
                V_3d = np.random.randn(r_target, m, p) * 0.3
                W_3d = np.random.randn(r_target, p, n) * 0.3
            elif restart % 3 == 1:
                # Sparse random
                U_3d = np.zeros((r_target, n, m))
                V_3d = np.zeros((r_target, m, p))
                W_3d = np.zeros((r_target, p, n))
                for _ in range(r_target * 2):
                    r_idx = np.random.randint(r_target)
                    U_3d[r_idx, np.random.randint(n), np.random.randint(m)] = np.random.randn()
                    V_3d[r_idx, np.random.randint(m), np.random.randint(p)] = np.random.randn()
                    W_3d[r_idx, np.random.randint(p), np.random.randint(n)] = np.random.randn()
            else:
                # Structured init: try to mimic schoolbook structure
                U_3d = np.zeros((r_target, n, m))
                V_3d = np.zeros((r_target, m, p))
                W_3d = np.zeros((r_target, p, n))
                for r_idx in range(r_target):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    U_3d[r_idx, i, j] = 1.0
                    V_3d[r_idx, j, k] = 1.0
                    W_3d[r_idx, k, i] = 1.0
                    # Add some noise
                    U_3d[r_idx] += np.random.randn(n, m) * 0.1
                    V_3d[r_idx] += np.random.randn(m, p) * 0.1
                    W_3d[r_idx] += np.random.randn(p, n) * 0.1
            
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            state = {}
            current_lr = lr
            
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                U_3d = params[0: r_target * n * m].reshape(r_target, n, m)
                V_3d = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_3d = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                loss, dU, dV, dW = _loss_and_grad(U_3d, V_3d, W_3d)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                
                if loss < 1e-8:
                    break
                
                grads = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                
                params = _adam_step(params, grads, state, lr=current_lr)
                
                # Decay learning rate
                current_lr *= 0.995
                
                # Every 50 iterations, try a small perturbation to escape local minima
                if it % 50 == 49 and loss > 1e-4:
                    params += np.random.randn(params.shape[0]) * 0.01 * np.exp(-it / 100)
        
        if best_params is not None and best_loss < 1e-6:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            if _verify(U_out, V_out, W_out, tol=1e-7):
                return U_out, V_out, W_out, best_loss
        
        return None

    # Try ranks from high to low, starting from reasonable candidates
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Practical candidates: try a range of ranks
    
    candidate_ranks = list(range(62, 20, -1))
    
    for r_target in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(r_target, max_iters=200, n_restarts=6, lr=0.03)
        
        if result is not None:
            U_best, V_best, W_best, _ = result
            best_rank = r_target
            # Once we find a valid decomposition, we could continue trying lower ranks
            # but for now, break to save time
            break
    
    # If no improvement found, try a few more structured approaches
    if best_rank == n * m * p and time.time() < deadline - 30:
        # Try a hybrid approach: use Strassen-like structure for the 3x3x3 part
        # and schoolbook for the remaining 3x3x4 part, then optimize
        
        # For now, just do additional random searches at rank 60-62
        for r_target in [60, 61, 62]:
            if time.time() > deadline - 10:
                break
            result = _try_rank(r_target, max_iters=100, n_restarts=10, lr=0.02)
            if result is not None:
                U_best, V_best, W_best, _ = result
                best_rank = r_target
                break
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
