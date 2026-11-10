"""Optimized product-formula coefficients for 6-site Heisenberg chain."""
import numpy as np
from scipy.linalg import expm
import time
import sys

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_chain(mats):
        result = np.array([[1]], dtype=complex)
        for m in mats:
            result = np.kron(result, m)
        return result
    
    def bond_term(i, j):
        t1 = kron_chain([sx if k == i else (sx if k == j else I2) for k in range(N)])
        t2 = kron_chain([sy if k == i else (sy if k == j else I2) for k in range(N)])
        t3 = kron_chain([sz if k == i else (sz if k == j else I2) for k in range(N)])
        return t1 + t2 + anisotropy * t3
    
    def field_term():
        result = np.zeros((dim, dim), dtype=complex)
        for k in range(N):
            mats = [sz if i == k else I2 for i in range(N)]
            result += kron_chain(mats)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    for i in range(0, N, 2):
        A += bond_term(i, i+1)
    A += 0.5 * field_term()
    
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(1, N, 2):
        B += bond_term(i, i+1)
    B += 0.5 * field_term()
    
    return A, B

def coefficients(stages):
    stages = int(stages)
    if stages != 12:
        # Fallback for non-12 stage cases
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigen-decompositions for fast matrix exponentials
    evals_A, V_A = np.linalg.eigh(A)
    V_A_inv = np.linalg.inv(V_A)
    evals_B, V_B = np.linalg.eigh(B)
    V_B_inv = np.linalg.inv(V_B)
    
    # Precompute exact propagators
    U_exact_cache = {}
    for t in times:
        U_exact_cache[t] = expm(-1j * t * H)
    
    def fast_expm_A(c, t):
        """Fast expm for c * A * t using precomputed eigen-decomposition"""
        if abs(c) < 1e-15:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * c * t * evals_A)
        return V_A @ np.diag(phase) @ V_A_inv
    
    def fast_expm_B(c, t):
        """Fast expm for c * B * t using precomputed eigen-decomposition"""
        if abs(c) < 1e-15:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * c * t * evals_B)
        return V_B @ np.diag(phase) @ V_B_inv
    
    def compute_error(a, b):
        """Compute mean spectral norm error for given coefficients"""
        total_err = 0.0
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                # Apply exp(-i b_k t B) then exp(-i a_k t A)
                Fk_B = fast_expm_B(b[k], t)
                Fk_A = fast_expm_A(a[k], t)
                U = U @ Fk_B @ Fk_A
            total_err += np.linalg.norm(U - U_exact_cache[t], 2)
        return total_err / len(times)
    
    # Start with uniform initialization
    x0 = np.ones(24) / 12.0
    a0 = x0[:12]
    b0 = x0[12:]
    best_a = a0.copy()
    best_b = b0.copy()
    best_err = compute_error(a0, b0)
    
    # Try a few different starting points and local optimizations
    from scipy.optimize import minimize
    
    def objective(x):
        a = x[:12]
        b = x[12:]
        # Enforce sum constraints by redistributing
        a = a + (1.0 - np.sum(a)) / 12
        b = b + (1.0 - np.sum(b)) / 12
        return compute_error(a, b)
    
    # Multi-start optimization with different perturbations
    np.random.seed(42)
    
    # First try the uniform start
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_err:
            best_err = res.fun
            best_a = res.x[:12] + (1.0 - np.sum(res.x[:12])) / 12
            best_b = res.x[12:] + (1.0 - np.sum(res.x[12:])) / 12
    except Exception:
        pass
    
    # Try several random perturbations
    for i in range(8):
        if time.time() > deadline - 5.0:
            break
        # Different perturbation scales
        scales = [0.02, 0.05, 0.1, 0.15, 0.03, 0.08, 0.12, 0.06]
        scale = scales[i % len(scales)]
        x_r = x0 + np.random.randn(24) * scale
        # Clip to reasonable range
        x_r = np.clip(x_r, -2.0, 2.0)
        
        try:
            res = minimize(objective, x_r, method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_a = res.x[:12] + (1.0 - np.sum(res.x[:12])) / 12
                best_b = res.x[12:] + (1.0 - np.sum(res.x[12:])) / 12
        except Exception:
            pass
    
    # Enforce sum constraints and bounds
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    best_a = best_a + (1.0 - np.sum(best_a)) / 12
    best_b = best_b + (1.0 - np.sum(best_b)) / 12
    
    # Final check: ensure finite values
    if not (np.all(np.isfinite(best_a)) and np.all(np.isfinite(best_b))):
        best_a = np.ones(12) / 12
        best_b = np.ones(12) / 12
    
    return list(best_a), list(best_b)
