import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute Hermite polynomial coefficient vectors (highest power first)
def _hermite_coeffs(n):
    H0 = [1]
    H1 = [2, 0]
    if n == 0:
        return H0
    if n == 1:
        return H1
    a, b = H0, H1
    for k in range(1, n):
        two_x_b = [2 * c for c in b] + [0]
        sub = [2 * k * c for c in a]
        L = max(len(two_x_b), len(sub))
        two_x_b = [0] * (L - len(two_x_b)) + two_x_b
        sub = [0] * (L - len(sub)) + sub
        res = [two_x_b[i] - sub[i] for i in range(L)]
        while len(res) > 1 and res[0] == 0:
            res = res[1:]
        a, b = b, res
    return b

H0c = _hermite_coeffs(0)
H4c = _hermite_coeffs(4)
H8c = _hermite_coeffs(8)
H12c = _hermite_coeffs(12)

H0_0 = H0c[-1]
H4_0 = H4c[-1]
H8_0 = H8c[-1]
H12_0 = H12c[-1]

def _align_add(a, b, scale_b):
    L = max(len(a), len(b))
    a = [0] * (L - len(a)) + a
    b = [0] * (L - len(b)) + b
    return [a[i] + scale_b * b[i] for i in range(L)]

H812 = _align_add(H8c, H12c, -H8_0 / H12_0)

L = len(H12c)
H0p = [0] * (L - len(H0c)) + H0c
H4p = [0] * (L - len(H4c)) + H4c
H812p = [0] * (L - len(H812)) + H812

def _g_to_q(gcoeff):
    return gcoeff[2:13]

qH0 = _g_to_q(H0p)
qH4 = _g_to_q(H4p)
qH812 = _g_to_q(H812p)

def _poly_eval(coeffs, x):
    v = 0.0
    for c in coeffs:
        v = v * x + c
    return v

def _build_q(c0, c1, c2):
    q = [0.0] * len(qH0)
    for i in range(len(qH0)):
        q[i] = c0 * qH0[i] + c1 * qH4[i] + c2 * qH812[i]
    return q

def _find_largest_sign_change_root(q):
    try:
        roots = np.roots(q)
    except Exception:
        return None
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10 * max(1.0, abs(rt.real)):
            real_roots.append(rt.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for r in real_roots:
        eps = 1e-7 * max(1.0, abs(r))
        left = _poly_eval(q, r - eps)
        right = _poly_eval(q, r + eps)
        if left * right < 0:
            return r
    return None

def _compute_value(c0, c1, c2):
    lead = c0 * H0p[0] + c1 * H4p[0] + c2 * H812p[0]
    if lead <= 0:
        return None
    q = _build_q(c0, c1, c2)
    r = _find_largest_sign_change_root(q)
    if r is None:
        return None
    return r * r / (2.0 * np.pi)

def _objective(coeffs):
    c0, c1, c2 = coeffs
    val = _compute_value(c0, c1, c2)
    if val is None:
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # 35s safety margin before 210s kill
    
    seed_c0, seed_c1, seed_c2 = -113.0/100, 1.0/25, 1.0/3240
    seed_val = _compute_value(seed_c0, seed_c1, seed_c2)
    if seed_val is None:
        seed_val = 1e10
    
    best_val = seed_val
    best_coeffs = np.array([seed_c0, seed_c1, seed_c2])
    
    # Strategy: Multi-phase search with diverse initializations
    # Phase 1: Global exploration with differential evolution (if time permits)
    # Phase 2: Local refinement from multiple starting points
    # Phase 3: High-precision polish on best result
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [
        [seed_c0, seed_c1, seed_c2],
        [seed_c0 * 0.5, seed_c1 * 2, seed_c2 * 0.5],
        [seed_c0 * 1.5, seed_c1 * 0.5, seed_c2 * 1.5],
        [-0.5, 0.1, 0.0005],
        [-0.2, 0.2, 0.001],
        [-1.0, 0.05, 0.001],
        [-0.8, 0.15, 0.0008],
        [0.0, 0.0, 1.0/10000],
        [-0.1, 0.0, 1.0/10000],
    ]
    
    # Add random starts
    for _ in range(15):
        c0 = rng.uniform(-3.0, 0.5)
        c1 = rng.uniform(0.0, 0.5)
        c2 = rng.uniform(0.0, 0.02)
        starts.append([c0, c1, c2])
    
    # Phase 1: Try differential evolution for global search if time permits
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 1.0), (0.0, 1.0), (0.0, 0.05)]
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x
        except Exception:
            pass
    
    # Phase 2: Local optimization from multiple starting points
    for start in starts:
        if time.time() > deadline - 20:
            break
        
        # Nelder-Mead
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 300}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
        
        # L-BFGS-B with numerical gradient
        if time.time() > deadline - 10:
            break
        try:
            res2 = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_coeffs = res2.x
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        try:
            res3 = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_coeffs = res3.x
        except Exception:
            pass
        
        try:
            res4 = minimize(
                _objective,
                best_coeffs,
                method='COBYLA',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 200}
            )
            if res4.fun < best_val:
                best_val = res4.fun
                best_coeffs = res4.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
