"""Structural change: root-space parameterization of q(x) with feasibility projection."""
from scipy.optimize import differential_evolution
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients (highest degree first)
def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    h_prev = np.array([1.0])
    h_curr = np.array([2.0, 0.0])
    for k in range(1, n):
        new_len = len(h_curr) + 1
        h_2x = np.zeros(new_len)
        h_2x[:-1] = 2.0 * h_curr
        h_2k = np.zeros(new_len)
        pad = new_len - len(h_prev)
        h_2k[pad:] = 2.0 * k * h_prev
        h_next = h_2x - h_2k
        h_prev = h_curr
        h_curr = h_next
    return h_curr

H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

H0_0 = H0_c[-1]   # 1
H4_0 = H4_c[-1]   # 12
H8_0 = H8_c[-1]   # 20160
H12_0 = H12_c[-1] # 462

def _build_q_from_coeffs(c0, c1, c2):
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = np.zeros(13)
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = 13 - len(hc)
        hc_p = np.concatenate([np.zeros(pad), hc])
        g += c * hc_p
    # g has degree 12, coefficients [x^12, x^11, ..., x^0] in g[0..12]
    # q = g/x^2 has degree 10, coefficients [x^10, x^9, ..., x^0]
    # q[i] = g[i+2], so q = g[2:13]
    q = g[2:13]  # degree 10, length 11
    return q

def _compute_value(c0, c1, c2):
    q = _build_q_from_coeffs(c0, c1, c2)
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) == 0:
        return float('inf')
    valid_roots = []
    for r in real_roots:
        eps = 1e-9 * max(1.0, abs(r))
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if not valid_roots:
        return float('inf')
    r = max(valid_roots)
    return r**2 / (2.0 * np.pi)




def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_value(*seed)
    best_params = seed.copy()
    
    # Strategy 1: Differential Evolution for global exploration
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                lambda p: _compute_value(p[0], p[1], p[2]),
                bounds,
                maxiter=50,
                popsize=30,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Multi-start Nelder-Mead for local refinement
    if time.time() < deadline - 30:
        np.random.seed(42)
        n_starts = 15
        for i in range(n_starts):
            if time.time() > deadline - 30:
                break
            if i == 0:
                trial = best_params.copy()
            elif i == 1:
                trial = seed.copy()
            else:
                if i % 2 == 0:
                    trial = best_params + np.random.normal(0, 0.5, 3)
                else:
                    trial = seed + np.random.normal(0, 0.5, 3)
            
            try:
                res = minimize(
                    lambda p: _compute_value(p[0], p[1], p[2]),
                    trial,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Strategy 3: Fine-tune with very small perturbations
    if time.time() < deadline - 10:
        np.random.seed(123)
        for i in range(50):
            if time.time() > deadline - 10:
                break
            trial = best_params + np.random.normal(0, 1e-5, 3)
            v = _compute_value(*trial)
            if v < best_val:
                best_val = v
                best_params = trial.copy()
    
    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: _compute_value(p[0], p[1], p[2]),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
