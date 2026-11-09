"""Nelder-Mead optimization of 12-stage product formula coefficients for Heisenberg chain."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    The full Hamiltonian H = A + B where:
    - A = sum over odd bonds (1-2, 3-4, 5-6) of XXZ interaction + half uniform field
    - B = sum over even bonds (2-3, 4-5) of XXZ interaction + half uniform field
    
    XXZ interaction on bond (i, i+1): Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + lambda Sz_i Sz_{i+1}
    Uniform field: h * sum_i Sz_i, split as h/2 in A and h/2 in B
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    # Pauli matrices / spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Identity for 2^N dimensional space
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply operator op on given site (0-indexed) to full 2^N space."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    # Odd bonds: (0,1), (2,3), (4,5)
    for i in range(0, N, 2):
        j = i + 1
        A += op_on_site(sx, i) @ op_on_site(sx, j)
        A += op_on_site(sy, i) @ op_on_site(sy, j)
        A += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        A += (h/2) * op_on_site(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    # Even bonds: (1,2), (3,4)
    for i in range(1, N-1, 2):
        j = i + 1
        B += op_on_site(sx, i) @ op_on_site(sx, j)
        B += op_on_site(sy, i) @ op_on_site(sy, j)
        B += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        B += (h/2) * op_on_site(sz, i)
    
    return A, B

# Precomputed eigen-decompositions
_eig_A = None
_eig_B = None
_eig_H = None

def _precompute():
    """Precompute eigen-decompositions of A, B, and H=A+B."""
    global _eig_A, _eig_B, _eig_H
    A, B = _build_hamiltonians()
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    H = A + B
    wH, vH = np.linalg.eigh(H)
    _eig_A = (wA, vA)
    _eig_B = (wB, vB)
    _eig_H = (wH, vH)
    return A, B

def _compute_error_fast(a, b):
    """Compute mean spectral-norm error at t = 0.25, 0.5, 1.0 using precomputed eigendecomps."""
    wA, vA = _eig_A
    wB, vB = _eig_B
    wH, vH = _eig_H
    
    dim = 2**6
    errors = []
    for t in [0.25, 0.5, 1.0]:
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # exp(-i a_k t A) = vA @ diag(exp(-i a_k t wA)) @ vA^H
            expA = vA @ np.diag(np.exp(-1j * a[k] * t * wA)) @ vA.conj().T
            U = U @ expA
            # exp(-i b_k t B)
            expB = vB @ np.diag(np.exp(-1j * b[k] * t * wB)) @ vB.conj().T
            U = U @ expB
        
        # Exact: exp(-i t (A+B))
        U_exact = vH @ np.diag(np.exp(-1j * t * wH)) @ vH.conj().T
        
        diff = U - U_exact
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients using scipy L-BFGS-B."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Precompute eigen-decompositions
    _precompute()
    
    # Use all 24 free parameters: a[0..10] and b[0..10], with a[11] and b[11] determined by sum=1
    # Initial guess: uniform 1/12
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    
    def objective(params):
        """Objective function for optimizer. params: 22 values (a[0..10], b[0..10])."""
        a = list(a0)
        b = list(b0)
        
        for i in range(11):
            a[i] = params[i]
            b[i] = params[11 + i]
        
        # Enforce sum = 1
        a[11] = 1.0 - sum(a[:11])
        b[11] = 1.0 - sum(b[:11])
        
        # Clamp to [-50, 50]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        
        return _compute_error_fast(a, b)
    
    x0 = a0[:11] + b0[:11]
    
    # Track best result
    best_x = x0
    best_f = objective(x0)
    
    # Use L-BFGS-B which handles 22 dimensions well
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'maxfun': 2000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            }
        )
        
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Check time, if we have more time, try a second phase with different initial guess
    if time.time() < deadline - 20:
        # Try asymmetric initial guesses
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb initial guess
            np.random.seed(42 + trial)
            x0_trial = x0 + np.random.randn(22) * 0.1
            
            try:
                result2 = minimize(
                    objective,
                    x0_trial,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'maxfun': 1000,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
                
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = result2.x
            except Exception:
                pass
    
    # Extract a and b from best_x
    a = list(a0)
    b = list(b0)
    for i in range(11):
        a[i] = best_x[i]
        b[i] = best_x[11 + i]
    a[11] = 1.0 - sum(a[:11])
    b[11] = 1.0 - sum(b[:11])
    
    # Final clamp
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
