import numpy as np
import time
import sys
import scipy.optimize
import scipy.linalg

# Global deadline guard
_T0 = time.time()
_DEADLINE = 120.0  # Safety margin before 150s limit

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """
    Build A and B for a 6-site open XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Anisotropy delta = 0.8, Field h = 0.3
    """
    N = 6
    delta = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h * sum S_i^z
    # Split into half for A and half for B
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term
        
    # Bond terms
    # Bonds are (i, i+1) for i=0 to N-2
    # Odd bonds in 1-based indexing: (1,2), (3,4), (5,6) -> 0-based: (0,1), (2,3), (4,5)
    # Even bonds in 1-based indexing: (2,3), (4,5) -> 0-based: (1,2), (3,4)
    
    odd_bonds = [(0,1), (2,3), (4,5)]
    even_bonds = [(1,2), (3,4)]
    
    for i, j in odd_bonds:
        # S_i^x S_j^x + S_i^y S_j^y + delta S_i^z S_j^z
        ops_xx = [I2]*N
        ops_xx[i] = sx
        ops_xx[j] = sx
        A += kron_all(ops_xx)
        
        ops_yy = [I2]*N
        ops_yy[i] = sy
        ops_yy[j] = sy
        A += kron_all(ops_yy)
        
        ops_zz = [I2]*N
        ops_zz[i] = sz
        ops_zz[j] = sz
        A += delta * kron_all(ops_zz)
        
    for i, j in even_bonds:
        ops_xx = [I2]*N
        ops_xx[i] = sx
        ops_xx[j] = sx
        B += kron_all(ops_xx)
        
        ops_yy = [I2]*N
        ops_yy[i] = sy
        ops_yy[j] = sy
        B += kron_all(ops_yy)
        
        ops_zz = [I2]*N
        ops_zz[i] = sz
        ops_zz[j] = sz
        B += delta * kron_all(ops_zz)
        
    return A, B

# Pre-build Hamiltonians and exact propagators
try:
    A, B = _build_hamiltonians()
    AB = A + B
    t_vals = [0.25, 0.5, 1.0]
    exact_propagators = [scipy.linalg.expm(-1j * t * AB) for t in t_vals]
    # Pre-compute A and B exponentials for various coefficients? No, coefficients change.
    # But we can pre-compute the structure.
except Exception:
    A = None
    B = None
    exact_propagators = []

def _compute_error(coeffs):
    """
    coeffs: list of 22 numbers (a1..a11, b1..b11)
    Returns: mean spectral norm error
    """
    if A is None:
        return 1e10
        
    a = list(coeffs[:11])
    b = list(coeffs[11:])
    
    a12 = 1.0 - sum(a)
    b12 = 1.0 - sum(b)
    
    a_full = a + [a12]
    b_full = b + [b12]
    
    # Check bounds
    for val in a_full + b_full:
        if val < -50 or val > 50:
            return 1e10
            
    errors = []
    for t, U_exact in zip(t_vals, exact_propagators):
        U = np.eye(2**6, dtype=complex)
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Order k=1..12
        for k in range(12):
            # exp(-i a_k t A)
            U = scipy.linalg.expm(-1j * a_full[k] * t * A) @ U
            # exp(-i b_k t B)
            U = scipy.linalg.expm(-1j * b_full[k] * t * B) @ U
            
        err = scipy.linalg.norm(U - U_exact, 2)
        errors.append(err)
        
    return np.mean(errors)

def coefficients(stages):
    """
    Returns a, b: two lists of 12 finite numbers in [-50, 50] with sum(a)=sum(b)=1.
    """
    if stages != 12:
        # Fallback to simple Strang if stages != 12, though task implies 12
        # The task says "Choose the coefficients of a twelve-stage product formula"
        # So stages should be 12.
        a = [1.0/12] * 12
        b = [1.0/12] * 12
        return a, b

    if A is None:
        # Fallback
        a = [1.0/12] * 12
        b = [1.0/12] * 12
        return a, b

    best_coeffs = None
    best_error = 1e10
    
    # Initial guess: uniform
    uniform_guess = [1.0/12] * 22
    
    # Differential Evolution
    # Bounds for the 22 free variables
    bounds = [(-50, 50) for _ in range(22)]
    
    # We want to minimize error
    # maxiter and popsize control the search
    # Given 150s limit, we need to be careful.
    # 22 dimensions is moderate.
    
    try:
        # Use a relatively fast DE configuration
        # workers=1 to avoid overhead
        res = scipy.optimize.differential_evolution(
            _compute_error, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            workers=1
        )
        
        if res.success or res.fun < best_error:
            best_error = res.fun
            best_coeffs = res.x
            print(f"DE done: {best_error}", file=sys.stderr)
            
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)
        best_coeffs = uniform_guess
        best_error = _compute_error(uniform_guess)

    # Polish with L-BFGS-B
    if _remaining() > 5.0:
        try:
            res_polish = scipy.optimize.minimize(
                _compute_error, 
                best_coeffs, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if res_polish.fun < best_error:
                best_error = res_polish.fun
                best_coeffs = res_polish.x
                print(f"Polish done: {best_error}", file=sys.stderr)
        except Exception as e:
            print(f"Polish failed: {e}", file=sys.stderr)

    # Construct final a and b
    a = list(best_coeffs[:11])
    b = list(best_coeffs[11:])
    a.append(1.0 - sum(a))
    b.append(1.0 - sum(b))
    
    # Ensure finite and in bounds (clamp if necessary, though DE should handle it)
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    # Re-normalize if clamping changed sums? 
    # If clamping happens, sums might not be 1. 
    # The constraint sum=1 is critical.
    # If we clamp, we should adjust the last element or distribute error.
    # Given the optimizer stays within bounds, this is unlikely to trigger.
    # But to be safe:
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-9:
        a[-1] += 1.0 - sum_a
    if abs(sum_b - 1.0) > 1e-9:
        b[-1] += 1.0 - sum_b
        
    return a, b

# EVOLVE-BLOCK-END
