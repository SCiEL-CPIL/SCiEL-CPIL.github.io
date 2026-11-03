"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(V[0]) if V else 0
    if rank == 0:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to 3D
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute tensor
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _error_fast(U, V, W, n, m, p, r):
    """Compute the error using vectorized operations."""
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.sum((T - 1.0) ** 2)

def _gradient(U, V, W, n, m, p, r):
    """Compute gradients for U, V, W."""
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    dT = 2.0 * (T - 1.0)
    
    dU3 = np.einsum('ijk,jkc,kic->ijc', dT, V3, W3)
    dV3 = np.einsum('ijk,ijc,kic->jkc', dT, U3, W3)
    dW3 = np.einsum('ijk,ijc,jkc->kic', dT, U3, V3)
    
    return dU3.reshape(-1, r), dV3.reshape(-1, r), dW3.reshape(-1, r)

def _lbfgs_optimize(U, V, W, n, m, p, r, max_iter=5000, ftol=1e-15, gtol=1e-12):
    """Optimize using L-BFGS-B."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U_ = x[:dim_U*r].reshape(dim_U, r)
        V_ = x[dim_U*r:dim_U*r+dim_V*r].reshape(dim_V, r)
        W_ = x[dim_U*r+dim_V*r:].reshape(dim_W, r)
        return _error_fast(U_, V_, W_, n, m, p, r)
    
    def gradient(x):
        U_ = x[:dim_U*r].reshape(dim_U, r)
        V_ = x[dim_U*r:dim_U*r+dim_V*r].reshape(dim_V, r)
        W_ = x[dim_U*r+dim_V*r:].reshape(dim_W, r)
        dU, dV, dW = _gradient(U_, V_, W_, n, m, p, r)
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
    
    x = result.x
    U_new = x[:dim_U*r].reshape(dim_U, r)
    V_new = x[dim_U*r:dim_U*r+dim_V*r].reshape(dim_V, r)
    W_new = x[dim_U*r+dim_V*r:].reshape(dim_W, r)
    
    return U_new, V_new, W_new, result.fun

def _adam_optimize(U, V, W, n, m, p, r, max_iters=20000, lr=0.001, seed=42):
    """Optimize using Adam optimizer."""
    np.random.seed(seed)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    
    for t in range(1, max_iters + 1):
        dU, dV, dW = _gradient(U, V, W, n, m, p, r)
        
        mU = beta1 * mU + (1 - beta1) * dU
        vU = beta2 * vU + (1 - beta2) * dU**2
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * dV
        vV = beta2 * vV + (1 - beta2) * dV**2
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW**2
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        loss = _error_fast(U, V, W, n, m, p, r)
        if loss < best_loss:
            best_loss = loss
            best_state = (U.copy(), V.copy(), W.copy())
        
        if loss < 1e-12:
            break
    
    return best_state[0], best_state[1], best_state[2], best_loss

def _structured_init(n, m, p, r, seed=42):
    """Create structured initializations."""
    np.random.seed(seed)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Strategy 1: Random
    U1 = np.random.randn(dim_U, r) * 0.1
    V1 = np.random.randn(dim_V, r) * 0.1
    W1 = np.random.randn(dim_W, r) * 0.1
    
    # Strategy 2: Sparse random
    U2 = np.zeros((dim_U, r))
    V2 = np.zeros((dim_V, r))
    W2 = np.zeros((dim_W, r))
    for col in range(r):
        row_U = np.random.randint(0, dim_U)
        row_V = np.random.randint(0, dim_V)
        row_W = np.random.randint(0, dim_W)
        U2[row_U, col] = np.random.choice([-1, 1])
        V2[row_V, col] = np.random.choice([-1, 1])
        W2[row_W, col] = np.random.choice([-1, 1])
    
    # Strategy 3: Block structured
    U3 = np.zeros((dim_U, r))
    V3 = np.zeros((dim_V, r))
    W3 = np.zeros((dim_W, r))
    for col in range(r):
        i = col % n
        j = (col // n) % m
        k = (col // (n*m)) % p
        U3[i*m+j, col] = 1.0
        V3[j*p+k, col] = 1.0
        W3[k*n+i, col] = 1.0
    
    # Strategy 4: Random with larger scale
    U4 = np.random.randn(dim_U, r) * 0.5
    V4 = np.random.randn(dim_V, r) * 0.5
    W4 = np.random.randn(dim_W, r) * 0.5
    
    return [(U1, V1, W1), (U2, V2, W2), (U3, W3, V3), (U4, V4, W4)]

def _try_rank(n, m, p, r, deadline, max_attempts=10):
    """Try to find a valid decomposition at rank r."""
    best_loss = float('inf')
    best_state = None
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        inits = _structured_init(n, m, p, r, seed=attempt*100 + 42)
        
        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            
            # Try Adam first
            U1, V1, W1, loss1 = _adam_optimize(U0, V0, W0, n, m, p, r, 
                                               max_iters=10000, lr=0.001, seed=attempt)
            
            if loss1 < best_loss:
                best_loss = loss1
                best_state = (U1, V1, W1)
            
            if loss1 < 1e-10:
                return U1, V1, W1, loss1
            
            # Try L-BFGS from Adam result
            if time.time() < deadline - 5:
                U2, V2, W2, loss2 = _lbfgs_optimize(U1, V1, W1, n, m, p, r,
                                                    max_iter=2000, ftol=1e-15, gtol=1e-12)
                if loss2 < best_loss:
                    best_loss = loss2
                    best_state = (U2, V2, W2)
                
                if loss2 < 1e-10:
                    return U2, V2, W2, loss2
            
            # Try L-BFGS directly from initialization
            if time.time() < deadline - 5:
                U3, V3, W3, loss3 = _lbfgs_optimize(U0, V0, W0, n, m, p, r,
                                                    max_iter=2000, ftol=1e-15, gtol=1e-12)
                if loss3 < best_loss:
                    best_loss = loss3
                    best_state = (U3, V3, W3)
                
                if loss3 < 1e-10:
                    return U3, V3, W3, loss3
    
    if best_state is not None:
        return best_state[0], best_state[1], best_state[2], best_loss
    return None, None, None, best_loss

def _round_and_validate(U, V, W, n, m, p, r):
    """Round to half-integers and validate."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r, V_r, W_r
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    if _validate(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p):
        return U_i, V_i, W_i
    
    return None, None, None

def _discrete_polish(U, V, W, n, m, p, r, iters=1000, seed=123):
    """
    Perform discrete local search to find exact integer/half-integer solution.
    """
    coeffs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    rng = random.Random(seed)
    
    # Round to nearest 0.5 first
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    cur_err = _error_fast(U, V, W, n, m, p, r)
    best_err = cur_err
    best_state = (U.copy(), V.copy(), W.copy())
    
    for it in range(iters):
        which = rng.randint(0, 2)
        if which == 0:
            row = rng.randint(0, n*m-1); col = rng.randint(0, r-1)
            old = U[row, col]; U[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-12:
                        break
            else:
                U[row, col] = old
        elif which == 1:
            row = rng.randint(0, m*p-1); col = rng.randint(0, r-1)
            old = V[row, col]; V[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-12:
                        break
            else:
                V[row, col] = old
        else:
            row = rng.randint(0, p*n-1); col = rng.randint(0, r-1)
            old = W[row, col]; W[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-12:
                        break
            else:
                W[row, col] = old
                
    return best_err, best_state

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_result = None
    
    # Create schoolbook solution as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, the true lower bound is higher
    # Try ranks from 20 up to schoolbook_rank - 1
    for r in range(20, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        U, V, W, loss = _try_rank(n, m, p, r, deadline, max_attempts=5)
        
        if U is not None and loss < 1e-8:
            # Try to round to half-integers
            U_r, V_r, W_r = _round_and_validate(U, V, W, n, m, p, r)
            
            if U_r is not None:
                best_rank = r
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return best_result
            else:
                # Try discrete polish
                if time.time() < deadline - 5:
                    err, state = _discrete_polish(U, V, W, n, m, p, r, iters=500, seed=r)
                    if err < 1e-12:
                        U2, V2, W2 = state
                        if _validate(U2.tolist(), V2.tolist(), W2.tolist(), n, m, p):
                            best_rank = r
                            best_result = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                            return best_result
    
    return best_result

# EVOLVE-BLOCK-END
