import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

def _trivial_decomposition(n, m, p):
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
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol

def _loss_and_grad(x, n, m, p, r):
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)

    dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2

    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank

    # Try to find a better rank using optimization
    # We try ranks 71, 70, ... down to 60
    # Given the time limit, we will only try a few.
    
    # For this specific task, if we can't find a better one quickly, we return trivial.
    # The prompt suggests a Kronecker product approach.
    # I will attempt a quick optimization for rank 71.
    
    def try_rank(r, max_time=10.0):
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
        U0 = rng.randn(n * m, r) * scale
        V0 = rng.randn(m * p, r) * scale
        W0 = rng.randn(p * n, r) * scale
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: _loss_and_grad(x, n, m, p, r),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                return None
                
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
            
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                return U_opt, V_opt, W_opt
                
        except Exception:
            pass
            
        return None

    # Try a few ranks
    for r in range(trivial_rank - 1, max(trivial_rank - 5, 60), -1):
        if time.time() > deadline - 15:
            break
            
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            best = {'U': U, 'V': V, 'W': W}
            best_rank = r
            break

    if best_rank < trivial_rank:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
