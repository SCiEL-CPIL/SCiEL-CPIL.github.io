"""Matrix multiplication 3x4x6 - model-and-optimize approach."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank_scipy(r, max_time_per_rank=5.0):
        """Try to find a rank-r decomposition using scipy.optimize.minimize."""
        np.random.seed(42)
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 2:
                return None
            
            # Initialize
            scale = 1.0 / np.sqrt(r)
            U0 = np.random.randn(n * m, r) * scale
            V0 = np.random.randn(m * p, r) * scale
            W0 = np.random.randn(p * n, r) * scale
            
            # Flatten for optimization
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def loss_and_grad(x):
                U = x[:n*m*r].reshape(n, m, r)
                V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                W = x[n*m*r + m*p*r :].reshape(p, n, r)
                
                T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T - 1.0
                loss = np.sum(err ** 2)
                
                # Gradients
                dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
                dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
                dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            try:
                result = opt.minimize(loss_and_grad, x0, method='L-BFGS-B', 
                                      jac=True, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.success or result.fun < 1e-6:
                    U_opt = result.x[:n*m*r].reshape(n, m, r)
                    V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                    W_opt = result.x[n*m*r + m*p*r :].reshape(p, n, r)
                    
                    if _verify(U_opt.reshape(n*m, r), V_opt.reshape(m*p, r), W_opt.reshape(p*n, r), n, m, p, tol=1e-6):
                        return U_opt.reshape(n*m, r), V_opt.reshape(m*p, r), W_opt.reshape(p*n, r)
            except Exception:
                continue
                
            if time.time() > deadline - 2:
                return None

        return None

    # Try ranks from low to high, but focus on close to 72 first as it's more likely
    # Actually, if rank 72 is the minimum, we should just return it.
    # But we must try to improve.
    # Try 71, 70, 69, 68.
    for r in [71, 70, 69, 68]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank_scipy(r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
