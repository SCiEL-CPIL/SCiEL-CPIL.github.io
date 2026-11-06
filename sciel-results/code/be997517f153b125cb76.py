"""
Product-formula coefficient design for Heisenberg-chain time evolution.
Uses local optimization (L-BFGS-B) to minimize the mean spectral-norm error
over three evolution times, with a deadline guard.
"""
import time
import numpy as np
from scipy.linalg import expm, norm as sp_norm
from scipy.optimize import minimize

# Deadline guard: 150s wall-clock limit, use 140s as safety margin
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _build_hamiltonians():
    """
    Build A and B: odd-bond and even-bond halves of a 6-site open XXZ Heisenberg chain
    with anisotropy 0.8 and field 0.3.
    
    The chain has sites 0..5. Bonds are between (i, i+1) for i=0..4.
    Even bonds: (0,1), (2,3), (4,5)  -> B
    Odd bonds:  (1,2), (3,4)         -> A
    
    Each bond term: J * (Sx_i Sx_j + Sy_i Sy_j + dz * Sz_i Sz_j)
    Plus uniform field h * sum_i Sz_i, split evenly: h/2 per half.
    
    For XXZ: J=1, dz=0.8, h=0.3
    """
    # Single-site Pauli matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    N = 6
    dim = 2 ** N
    J = 1.0
    dz = 0.8
    h = 0.3
    
    def site_op(op, site):
        """Apply operator op on a given site (0-indexed from left)."""
        result = np.array([[1.0]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, id2)
        return result
    
    def bond_op(i, j):
        """Compute Sx_i Sx_j + Sy_i Sy_j + dz * Sz_i Sz_j for bond (i,j)."""
        sx_i = site_op(sx, i)
        sx_j = site_op(sx, j)
        sy_i = site_op(sy, i)
        sy_j = site_op(sy, j)
        sz_i = site_op(sz, i)
        sz_j = site_op(sz, j)
        return sx_i @ sx_j + sy_i @ sy_j + dz * (sz_i @ sz_j)
    
    # Even bonds: (0,1), (2,3), (4,5)
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        B += J * bond_op(i, j)
    # Add half the uniform field to B
    for s in range(N):
        B += (h / 2.0) * site_op(sz, s)
    
    # Odd bonds: (1,2), (3,4)
    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1, 2), (3, 4)]:
        A += J * bond_op(i, j)
    # Add half the uniform field to A
    for s in range(N):
        A += (h / 2.0) * site_op(sz, s)
    
    return A, B

def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of M."""
    try:
        s = sp_norm(M, 2)
        if np.isnan(s) or np.isinf(s):
            return 1e10
        return s
    except:
        return 1e10

def _objective(params, A, B, times):
    """
    Compute mean spectral-norm error for given coefficients.
    params: 24-element vector: first 12 are a, next 12 are b.
    """
    a = params[:12]
    b = params[12:]
    
    # Enforce bounds loosely in objective (penalty)
    penalty = 0.0
    for val in params:
        if val < -50:
            penalty += (val + 50) ** 2
        elif val > 50:
            penalty += (val - 50) ** 2
    
    # Sum constraints
    penalty += (np.sum(a) - 1.0) ** 2 * 100
    penalty += (np.sum(b) - 1.0) ** 2 * 100
    
    total_err = 0.0
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            # exp(-i * b_k * t * B) @ exp(-i * a_k * t * A)
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        
        U_exact = expm(-1j * t * (A + B))
        err = _spectral_norm(U - U_exact)
        total_err += err
    
    return total_err / len(times) + penalty

def coefficients(stages):
    """
    Optimize coefficients of a 12-stage product formula.
    Returns (a, b): two lists of 12 finite numbers in [-50, 50] with sum(a)=sum(b)=1.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    stages = int(stages)
    if stages != 12:
        # Fallback: simple equal split
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: symmetric splitting
    a0 = [1.0 / 12] * 12
    b0 = [1.0 / 12] * 12
    x0 = np.array(a0 + b0)
    
    # Bounds
    bounds = [(-50, 50)] * 24
    
    best_params = x0.copy()
    best_obj = _objective(x0, A, B, times)
    
    # Try L-BFGS-B optimization
    if _time_remaining() > 30:
        try:
            res = minimize(
                _objective, x0,
                args=(A, B, times),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if _time_remaining() > 5:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
        except Exception:
            pass
    
    # If time allows, try a second optimization from a different starting point
    if _time_remaining() > 25:
        # Try asymmetric initial guess
        a1 = [1.0/12] * 12
        b1 = [1.0/12] * 12
        # Perturb
        np.random.seed(42)
        for i in range(12):
            a1[i] += np.random.uniform(-0.05, 0.05)
            b1[i] += np.random.uniform(-0.05, 0.05)
        # Renormalize
        a1 = np.array(a1) / np.sum(a1)
        b1 = np.array(b1) / np.sum(b1)
        x1 = np.concatenate([a1, b1])
        
        try:
            res2 = minimize(
                _objective, x1,
                args=(A, B, times),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if _time_remaining() > 5:
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    best_params = res2.x
        except Exception:
            pass
    
    # Extract and clamp
    a = best_params[:12].tolist()
    b = best_params[12:].tolist()
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]
    
    # Ensure sums are exactly 1 (renormalize)
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [v / sum_a for v in a]
    else:
        a = [1.0/12] * 12
    if abs(sum_b) > 1e-15:
        b = [v / sum_b for v in b]
    else:
        b = [1.0/12] * 12
    
    # Final clamp after renormalization
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]
    
    # Verify finite
    for v in a + b:
        if not np.isfinite(v):
            a = [1.0/12] * 12
            b = [1.0/12] * 12
            break
    
    return a, b
