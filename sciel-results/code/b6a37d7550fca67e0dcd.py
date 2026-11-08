import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W):
    T = _build_target_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_decomp, T, atol=1e-9)

def _to_half_int(x):
    return np.round(x * 2) / 2

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _adam_step(params, grads, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m = state['m']
    v = state['v']
    t = state['t']
    t += 1
    for i in range(len(params)):
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    state['t'] = t

def _optimize(U, V, W, T, max_iters=500, lr=0.01):
    n, m, p = DIMENSIONS
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Initialize Adam state
    state = {'m': [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)],
             'v': [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)],
             't': 0}
    
    params = [U, V, W]
    
    for it in range(max_iters):
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        err = T_decomp - T
        loss = np.sum(err**2)
        
        if loss < 1e-10:
            break
            
        grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V)
        
        grads = [grad_U, grad_V, grad_W]
        
        _adam_step(params, grads, state, lr=lr)
        
        U, V, W = params
        
        # Normalize V to prevent scaling ambiguity
        # We want to keep the product U*V*W invariant.
        # If we scale V_r by c, we must scale U_r by 1/c.
        # Let's normalize V_r to have norm 1.
        for r in range(V.shape[1]):
            norm = np.linalg.norm(V[:, r])
            if norm > 1e-12:
                V[:, r] /= norm
                U[:, r] *= norm
            else:
                # If norm is zero, we have a problem.
                # Add small noise
                V[:, r] = np.random.randn(m*p) * 0.1
                U[:, r] = np.random.randn(n*m) * 0.1

    return U, V, W, loss

def _try_rank(rank, num_trials=5, max_iters=200):
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    for trial in range(num_trials):
        np.random.seed(42 + trial * 137 + rank)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        U, V, W, loss = _optimize(U, V, W, T, max_iters=max_iters, lr=0.01)
        
        if loss < 1e-8:
            U_h = _to_half_int(U)
            V_h = _to_half_int(V)
            W_h = _to_half_int(W)
            if _verify(U_h, V_h, W_h):
                return {'U': U_h.tolist(), 'V': V_h.tolist(), 'W': W_h.tolist()}
    return None

def construct():
    start = time.time()
    deadline = start + 110
    best = _schoolbook()
    
    # Search for lower ranks starting from 49 down to 40
    # We only need to check if a lower rank exists.
    # If the true rank is 50, we return schoolbook.
    # If the true rank is 49, we might find it.
    for rank in range(49, 39, -1):
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, num_trials=3, max_iters=150)
        if result is not None:
            best = result
            break
            
    return best
# EVOLVE-BLOCK-END
