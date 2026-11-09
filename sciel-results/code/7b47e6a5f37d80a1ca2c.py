import numpy as np
import time
from scipy.optimize import minimize
import sys

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    anisotropy = 0.8
    field = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Add field to both A and B (half each)
    for j in range(N):
        ops = [I2] * N
        ops[j] = sz
        term = kron_all(ops)
        A += (field / 2.0) * term
        B += (field / 2.0) * term
        
    # Add bonds
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    bonds_A = [(0, 1), (2, 3), (4, 5)]
    bonds_B = [(1, 2), (3, 4)]
    
    for i, j in bonds_A:
        # Sx Sx + Sy Sy + lambda Sz Sz
        # Sx Sx
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        A += kron_all(ops)
        
        # Sy Sy
        ops = [I2] * N
        ops[i] = sy
        ops[j] = sy
        A += kron_all(ops)
        
        # lambda Sz Sz
        ops = [I2] * N
        ops[i] = sz
        ops[j] = sz
        A += anisotropy * kron_all(ops)
        
    for i, j in bonds_B:
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        B += kron_all(ops)
        
        ops = [I2] * N
        ops[i] = sy
        ops[j] = sy
        B += kron_all(ops)
        
        ops = [I2] * N
        ops[i] = sz
        ops[j] = sz
        B += anisotropy * kron_all(ops)
        
    return A, B

def _log_sum_exp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def _exp_normalize(x):
    # Ensure numerical stability
    x = np.asarray(x)
    m = np.max(x)
    ex = np.exp(x - m)
    s = np.sum(ex)
    if s == 0:
        return np.ones_like(x) / len(x)
    return ex / s

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Exact propagator
    # Use scipy.linalg.expm for accuracy
    from scipy.linalg import expm
    t_values = [0.25, 0.5, 1.0]
    exact_propagators = [expm(-1j * t * H) for t in t_values]
    
    # Precompute expm of A and B? No, coefficients change, so A and B are fixed, but the exponent is -i a_k t A.
    # We can precompute the eigen-decomposition of A and B to speed up expm.
    # A and B are Hermitian.
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    
    def expm_A(alpha):
        # exp(-i alpha A)
        # vA diag(exp(-i alpha wA)) vA^dagger
        diag = np.exp(-1j * alpha * wA)
        return (vA * diag) @ vA.conj().T
        
    def expm_B(beta):
        # exp(-i beta B)
        diag = np.exp(-1j * beta * wB)
        return (vB * diag) @ vB.conj().T

    def objective(params):
        # params: 24 numbers, first 12 are log(a_i), next 12 are log(b_i)
        # We use log-space to ensure positivity and sum=1
        log_a = params[:12]
        log_b = params[12:]
        
        a = _exp_normalize(log_a)
        b = _exp_normalize(log_b)
        
        # Check bounds [-50, 50]
        # Since sum=1 and positive, max is 1, min is >0. So bounds are satisfied.
        
        err = 0.0
        for t, U_exact in zip(t_values, exact_propagators):
            U_approx = np.eye(dim, dtype=complex)
            for k in range(12):
                U_approx = expm_B(b[k] * t) @ U_approx
                U_approx = expm_A(a[k] * t) @ U_approx
            # Frobenius norm squared
            diff = U_approx - U_exact
            err += np.vdot(diff, diff).real
            
        return err / 3.0

    # Initial guess: uniform
    x0 = np.zeros(24)
    
    # Use L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
        if time.time() > deadline:
            raise TimeoutError
        best_x = res.x
        best_val = res.fun
    except Exception:
        # Fallback to simple gradient descent or just return uniform if optimization fails
        best_x = x0
        best_val = objective(x0)
        
    # Check if we can do a second pass with better start or more iterations if time permits
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 500})
            if time.time() < deadline:
                if res2.fun < best_val:
                    best_x = res2.x
                    best_val = res2.fun
        except Exception:
            pass
            
    log_a = best_x[:12]
    log_b = best_x[12:]
    a = _exp_normalize(log_a)
    b = _exp_normalize(log_b)
    
    # Final sanity check
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        a = np.ones(12) / 12
        b = np.ones(12) / 12
        
    return list(map(float, a)), list(map(float, b))

# EVOLVE-BLOCK-START
# The function coefficients is defined above
# EVOLVE-BLOCK-END
