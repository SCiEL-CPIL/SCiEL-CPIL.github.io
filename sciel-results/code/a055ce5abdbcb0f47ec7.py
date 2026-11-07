"""Structural redesign: stable root-finding + smooth penalized objective + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Physicists' Hermite polynomial constants
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_coeffs(c0, c1, c2):
    """Return [a0..a5] for q(y) = a0 + a1*y + ... + a5*y^5, y=x^2."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    a0 = c1 * (-48.0) + c2 * (-1344.0) + c3 * (-921600.0)
    a1 = c1 * 16.0 + c2 * 2688.0 + c3 * 1424640.0
    a2 = c2 * (-1344.0) + c3 * (-1003520.0)
    a3 = c2 * 256.0 + c3 * 332160.0
    a4 = c3 * (-46080.0)
    a5 = c3 * 4096.0
    return np.array([a0, a1, a2, a3, a4, a5]), c3

def find_largest_sign_change_root_stable(coeffs):
    """Find largest y>=0 root where polynomial changes sign, using companion matrix."""
    # Companion matrix for polynomial a5*y^5 + a4*y^4 + ... + a0
    # The companion matrix has eigenvalues equal to the roots
    a = coeffs  # [a0, a1, a2, a3, a4, a5]
    if abs(a[5]) < 1e-15:
        # Degenerate: reduce degree
        # Find highest non-zero coefficient
        deg = 5
        while deg > 0 and abs(a[deg]) < 1e-15:
            deg -= 1
        if deg == 0:
            return -np.inf
        a = a[:deg+1]
    
    n = len(a) - 1  # degree
    if n == 0:
        return -np.inf
    
    # Companion matrix: for monic polynomial y^n + c_{n-1}y^{n-1} + ... + c_0
    # First normalize to monic
    leading = a[n]
    if abs(leading) < 1e-15:
        return -np.inf
    c = a[:n] / leading  # [c0, c1, ..., c_{n-1}]
    
    # Companion matrix
    C = np.zeros((n, n))
    if n >= 1:
        C[0, :n-1] = 1.0
        C[1:, 0] = -c[::-1]
    
    try:
        eigvals = np.linalg.eigvals(C)
    except Exception:
        return -np.inf
    
    real_roots = []
    for ev in eigvals:
        if abs(ev.imag) < 1e-10 * max(1.0, abs(ev.real)):
            real_roots.append(ev.real)
    
    best_root = -np.inf
    for root in real_roots:
        if root < 1e-12:
            continue
        # Check sign change using Horner's method for stability
        eps = 1e-9 * max(1.0, root)
        y_minus = root - eps
        y_plus = root + eps
        
        # Horner evaluation
        val_minus = 0.0
        val_plus = 0.0
        for coef in a[::-1]:
            val_minus = val_minus * y_minus + coef
            val_plus = val_plus * y_plus + coef
        
        if val_minus * val_plus < 0:
            if root > best_root:
                best_root = root
    
    return best_root

def evaluate_value(c0, c1, c2):
    """Compute value = r / (2*pi) where r is largest sign-changing root of q(y)."""
    coeffs, c3 = compute_q_coeffs(c0, c1, c2)
    
    if c3 < 0:
        coeffs = -coeffs
        c3 = -c3
    
    if c3 <= 1e-15:
        return 1e10
    
    r = find_largest_sign_change_root_stable(coeffs)
    if r < 0 or not np.isfinite(r):
        return 1e10
    
    return r / (2.0 * np.pi)

def smooth_objective(params):
    """Penalized smooth objective for L-BFGS-B."""
    c0, c1, c2 = params
    try:
        coeffs, c3 = compute_q_coeffs(c0, c1, c2)
        
        # Penalty for c3 <= 0 (need positive leading coeff)
        penalty_c3 = 0.0
        if c3 < 0.01:
            penalty_c3 = (0.01 - c3) ** 2 * 100.0
        
        r = find_largest_sign_change_root_stable(coeffs)
        
        if r < 0 or not np.isfinite(r):
            # No valid root: penalize heavily
            # Use distance to nearest valid configuration
            return 1e10 + penalty_c3
        
        value = r / (2.0 * np.pi)
        return value + penalty_c3
    except Exception:
        return 1e10

def construct():
    """Search for optimal coefficients with stable numerics."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_value = evaluate_value(*best_params)
    
    # Phase 1: L-BFGS-B from seed with smooth objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 80:
        try:
            res = minimize(
                smooth_objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_value and np.isfinite(res.fun):
                best_params = res.x.copy()
                best_value = evaluate_value(*best_params)
        except Exception:
            pass
    
    # Phase 2: Differential evolution for global exploration
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            bounds = [(-3.0, 3.0), (-2.0, 2.0), (-0.05, 0.05)]
            de_result = differential_evolution(
                smooth_objective, bounds,
                maxiter=80, popsize=12, tol=1e-8,
                seed=42, polish=True,
                mutation=(0.4, 0.9), recombination=0.6
            )
            if de_result.fun < best_value and np.isfinite(de_result.fun):
                best_params = de_result.x.copy()
                best_value = evaluate_value(*best_params)
        except Exception:
            pass
    
    # Phase 3: Multiple L-BFGS-B restarts from perturbed points
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        rng = np.random.RandomState(7)
        for i in range(40):
            if time.time() > deadline - 20:
                break
            # Perturb around current best with decreasing scale
            scale = 0.1 * (1.0 - i / 50.0) + 0.01
            perturb = best_params + rng.normal(0, scale, 3)
            try:
                res = minimize(
                    smooth_objective, perturb,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_value and np.isfinite(res.fun):
                    best_params = res.x.copy()
                    best_value = evaluate_value(*best_params)
            except Exception:
                pass
    
    # Phase 4: Final Nelder-Mead polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            nm_result = minimize(
                smooth_objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            if nm_result.fun < best_value and np.isfinite(nm_result.fun):
                best_params = nm_result.x.copy()
                best_value = evaluate_value(*best_params)
        except Exception:
            pass
    
    # Final verification
    final_val = evaluate_value(*best_params)
    if final_val < best_value:
        best_value = final_val
    
    # Ensure we never return worse than seed
    seed_val = evaluate_value(*seed)
    if seed_val < best_value:
        best_params = seed.copy()
        best_value = seed_val
    
    return best_params
# EVOLVE-BLOCK-END
