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


def _loss_and_grad(params, n, m, p, r):
    """Compute loss and gradient for scipy.optimize."""
    size_u = n * m * r
    size_v = m * p * r
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    W = W_flat.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)
    
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _snap_to_half_int(U, V, W, n, m, p):
    """Snap to nearest half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
        return U_r, V_r, W_r
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank(r, n_restarts=10, max_iter=1000):
        """Try to find a rank-r decomposition using L-BFGS-B."""
        best_result = None
        best_loss = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart * 137)
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(n, m, r) * scale
            V = np.random.randn(m, p, r) * scale
            W = np.random.randn(p, n, r) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(
                    _loss_and_grad, 
                    params, 
                    args=(n, m, p, r),
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.fun < best_loss:
                best_loss = res.fun
                if best_loss < 1e-6:
                    size_u = n * m * r
                    size_v = m * p * r
                    U_opt = res.x[:size_u].reshape(n, m, r)
                    V_opt = res.x[size_u:size_u + size_v].reshape(m, p, r)
                    W_opt = res.x[size_u + size_v:].reshape(p, n, r)
                    
                    U_flat = U_opt.reshape(n*m, r)
                    V_flat = V_opt.reshape(m*p, r)
                    W_flat = W_opt.reshape(p*n, r)
                    
                    # Try to snap to half-integers
                    snapped = _snap_to_half_int(U_flat, V_flat, W_flat, n, m, p)
                    if snapped is not None:
                        return snapped
                    else:
                        # Return the continuous solution if snapping fails but verification passes
                        if _verify(U_flat, V_flat, W_flat, n, m, p, tol=1e-5):
                            return U_flat, V_flat, W_flat
                        
        return None

    # Try ranks from low to high, but prioritize higher ranks first?
    # Actually, if 72 is the min, we should check 71, 70...
    # If a lower rank exists, it's a "jump".
    for r in range(71, 59, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, n_restarts=10, max_iter=1000)
        if result is not None:
            U, V, W = result
            best = {'U': U, 'V': V, 'W': W}
            best_rank = r
            break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
