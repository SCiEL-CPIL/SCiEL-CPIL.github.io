"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _build_heisenberg_matrices():
    """Build A (even bonds + half field) and B (odd bonds + half field) for 6-site XXZ chain."""
    n = 6
    delta = 0.8
    h = 0.3
    dim = 2 ** n
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site (0-indexed)."""
        result = op
        for s in range(n - 1, site, -1):
            result = np.kron(result, ident)
        result = np.kron(result, op)
        for s in range(site - 1, -1, -1):
            result = np.kron(ident, result)
        return result
    
    # Even bonds: (0,1), (2,3), (4,5) -> indices 1,3,5 in 1-indexed
    even_bonds = [(0, 1), (2, 3), (4, 5)]
    # Odd bonds: (1,2), (3,4) -> indices 2,4 in 1-indexed
    odd_bonds = [(1, 2), (3, 4)]
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i, j in even_bonds:
        A += op_on_site(sx, i) @ op_on_site(sx, j)
        A += op_on_site(sy, i) @ op_on_site(sy, j)
        A += delta * op_on_site(sz, i) @ op_on_site(sz, j)
    
    for i, j in odd_bonds:
        B += op_on_site(sx, i) @ op_on_site(sx, j)
        B += op_on_site(sy, i) @ op_on_site(sy, j)
        B += delta * op_on_site(sz, i) @ op_on_site(sz, j)
    
    # Uniform field: h * sum_j S^z_j, split half to A and half to B
    field = np.zeros((dim, dim), dtype=complex)
    for j in range(n):
        field += op_on_site(sz, j)
    
    A += 0.5 * h * field
    B += 0.5 * h * field
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    errors = []
    for t in times:
        U = np.eye(2 ** 6, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _objective(params, A, B, times):
    """Objective function for optimization."""
    a = params[:12].tolist()
    b = params[12:].tolist()
    # Enforce sum constraints by adjusting last element
    a_sum = sum(a)
    b_sum = sum(b)
    a[11] += (1.0 - a_sum)
    b[11] += (1.0 - b_sum)
    
    # Check bounds
    if any(abs(x) > 50.0 for x in a + b):
        return 1e10
    
    return _compute_error(a, b, A, B, times)

def coefficients(stages):
    """Find optimal 12-stage product formula coefficients."""
    global _start_time, _DEADLINE
    
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    times = [0.25, 0.5, 1.0]
    
    # Build matrices (cache them)
    if not hasattr(coefficients, '_A_cached'):
        A, B = _build_heisenberg_matrices()
        coefficients._A_cached = A
        coefficients._B_cached = B
    A = coefficients._A_cached
    B = coefficients._B_cached
    
    # Initial guess: symmetric 4th-order-like splitting
    # Use a simple symmetric pattern as starting point
    base_a = [1/12] * 12
    base_b = [1/12] * 12
    
    best_params = base_a + base_b
    best_err = _compute_error(base_a, base_b, A, B, times)
    
    # Try several initializations
    initializations = [
        base_a + base_b,
        # Slightly perturbed symmetric
        [1/12 + 0.01*np.sin(2*np.pi*k/12) for k in range(12)] + 
        [1/12 + 0.01*np.cos(2*np.pi*k/12) for k in range(12)],
        # Clustered
        [0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25] + 
        [0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25],
    ]
    
    for init in initializations:
        if time.time() > _DEADLINE - 10:
            break
        
        # Normalize to ensure sums are close to 1
        a_init = init[:12].copy()
        b_init = init[12:].copy()
        a_init[11] += 1.0 - sum(a_init)
        b_init[11] += 1.0 - sum(b_init)
        
        try:
            result = minimize(
                _objective,
                init,
                args=(A, B, times),
                method='L-BFGS-B',
                bounds=[(-50, 50)] * 24,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < _DEADLINE - 15:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(A, B, times),
                method='L-BFGS-B',
                bounds=[(-50, 50)] * 24,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception:
            pass
    
    a = best_params[:12].tolist()
    b = best_params[12:].tolist()
    
    # Ensure exact sum = 1
    a[11] += 1.0 - sum(a)
    b[11] += 1.0 - sum(b)
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Re-normalize after clamping
    a[11] += 1.0 - sum(a)
    b[11] += 1.0 - sum(b)
    
    return a, b
# EVOLVE-BLOCK-END
