# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 200.0  # Safety margin below the 210s kill

# ---------- Precompute Hermite Polynomial Coefficients ----------
# Physicists' Hermite polynomials:
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
# H_12(x) = 4096x^12 - 147456x^10 + 2903040x^8 - 29030400x^6 + 145152000x^4 - 393216000x^2 + 1032160

_H0_coeffs = np.zeros(13)
_H0_coeffs[0] = 1.0

_H4_coeffs = np.zeros(13)
_H4_coeffs[0] = 12.0
_H4_coeffs[2] = -48.0
_H4_coeffs[4] = 16.0

_H8_coeffs = np.zeros(13)
_H8_coeffs[0] = 40320.0
_H8_coeffs[2] = -67200.0
_H8_coeffs[4] = 26880.0
_H8_coeffs[6] = -4480.0
_H8_coeffs[8] = 256.0

_H12_coeffs = np.zeros(13)
_H12_coeffs[0] = 1032160.0
_H12_coeffs[2] = -393216000.0
_H12_coeffs[4] = 145152000.0
_H12_coeffs[6] = -29030400.0
_H12_coeffs[8] = 2903040.0
_H12_coeffs[10] = -147456.0
_H12_coeffs[12] = 4096.0

# Values at x=0
_H0_0 = _H0_coeffs[0]
_H4_0 = _H4_coeffs[0]
_H8_0 = _H8_coeffs[0]
_H12_0 = _H12_coeffs[0]

def _q_coeffs(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # g(x) coefficients (degree 0 to 12)
    g = c0 * _H0_coeffs + c1 * _H4_coeffs + c2 * _H8_coeffs + c3 * _H12_coeffs
    
    # q(x) = g(x) / x^2. Since g is even, only even powers exist.
    # q(x) = sum_{k=1}^6 g[2k] x^{2k-2}
    q_full = g[2:] 
    return q_full

def _smallest_positive_sign_changing_root(q_coeffs):
    """
    Find the smallest positive real root of the polynomial q_coeffs (low->high) 
    where the polynomial changes sign.
    """
    # Remove leading zeros to ensure correct degree for np.roots
    qc = q_coeffs.copy()
    while len(qc) > 1 and abs(qc[-1]) < 1e-15:
        qc = qc[:-1]
    
    if len(qc) < 2:
        return np.inf
        
    try:
        roots = np.roots(qc)
    except Exception:
        return np.inf
        
    best_r = np.inf
    for rt in roots:
        # Check if real
        if abs(rt.imag) > 1e-10 * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-12:
            continue
            
        # Check sign change.
        # A root is a sign change if the multiplicity is odd.
        # We use the derivative check for simple roots.
        d_coeffs = np.polyder(np.flip(qc))
        d_val = np.polyval(d_coeffs, zr)
        
        if abs(d_val) < 1e-10 * max(1.0, abs(zr)):
            continue
            
        # Verify the root is actually a root
        q_val = np.polyval(np.flip(qc), zr)
        if abs(q_val) > 1e-8 * max(1.0, abs(zr)):
            continue
            
        if zr < best_r:
            best_r = zr
            
    return best_r

def _objective(params):
    """
    Objective function for optimization.
    Returns value = r^2 / (2*pi). Lower is better.
    """
    c0, c1, c2 = params
    try:
        qc = _q_coeffs(c0, c1, c2)
        r = _smallest_positive_sign_changing_root(qc)
        if not np.isfinite(r) or r <= 0:
            return 1e12
        return r * r / (2.0 * np.pi)
    except Exception:
        return 1e12

def _polish_high_precision(x0, max_iter=2000):
    """
    High-precision polish using Nelder-Mead with very tight tolerances.
    """
    try:
        res = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead',
            options={
                'maxiter': max_iter, 
                'xatol': 1e-15, 
                'fatol': 1e-18, 
                'adaptive': True,
                'initial_simplex': None
            }
        )
        if res.fun < 1e10:
            return res.x, res.fun
    except Exception:
        pass
    
    # Try L-BFGS-B as a second opinion
    try:
        res2 = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-15, 'ftol': 1e-18}
        )
        if res2.fun < 1e10:
            return res2.x, res2.fun
    except Exception:
        pass
    
    return x0, _objective(x0)

def construct():
    global _T0
    try:
        seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_val = _objective(seed)
        best_x = seed.copy()
        
        # Multi-start differential evolution with multiple seeds
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
        
        # Try multiple random initializations
        rng = np.random.default_rng(42)
        n_starts = 5
        candidates = [seed]
        
        for i in range(n_starts):
            if time.time() - _T0 > _DEADLINE - 60:
                break
            # Random points in the search space
            x_rand = np.array([
                rng.uniform(bounds[0][0], bounds[0][1]),
                rng.uniform(bounds[1][0], bounds[1][1]),
                rng.uniform(bounds[2][0], bounds[2][1])
            ])
            candidates.append(x_rand)
        
        # Run DE from each candidate with shorter runs
        for i, x_init in enumerate(candidates):
            if time.time() - _T0 > _DEADLINE - 30:
                break
            
            # Adjust bounds to be centered around current best if we have one
            if i == 0:
                de_bounds = bounds
            else:
                # Narrower bounds around best found so far
                center = best_x
                half_width = np.array([5.0, 5.0, 5.0])
                de_bounds = [
                    (max(center[0] - half_width[0], bounds[0][0]), 
                     min(center[0] + half_width[0], bounds[0][1])),
                    (max(center[1] - half_width[1], bounds[1][0]), 
                     min(center[1] + half_width[1], bounds[1][1])),
                    (max(center[2] - half_width[2], bounds[2][0]), 
                     min(center[2] + half_width[2], bounds[2][1]))
                ]
            
            try:
                res = differential_evolution(
                    _objective, 
                    de_bounds, 
                    seed=42 + i, 
                    maxiter=80, 
                    popsize=20, 
                    tol=1e-8,
                    polish=False,
                    init='sobol' if i > 0 else 'latinhypercube'
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Polish the best candidate with high precision
        if time.time() - _T0 < _DEADLINE - 10 and best_val < 1e10:
            polished_x, polished_val = _polish_high_precision(best_x, max_iter=3000)
            if polished_val < best_val:
                best_val = polished_val
                best_x = polished_x
        
        # Additional local search from neighborhood of best
        if time.time() - _T0 < _DEADLINE - 5 and best_val < 1e10:
            # Try small perturbations
            for trial in range(20):
                if time.time() - _T0 > _DEADLINE - 2:
                    break
                perturbation = np.array([
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.1, 0.1)
                ])
                x_trial = best_x + perturbation
                val_trial = _objective(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial
                    # Polish this new best
                    if time.time() - _T0 < _DEADLINE - 5:
                        polished_x2, polished_val2 = _polish_high_precision(best_x, max_iter=1000)
                        if polished_val2 < best_val:
                            best_val = polished_val2
                            best_x = polished_x2
        
        # Final Safety Check
        if not np.isfinite(best_val) or best_val >= 1e10:
            best_x = seed
            best_val = _objective(seed)

        return np.array([float(best_x[0]), float(best_x[1]), float(best_x[2])])
        
    except Exception:
        seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        return np.array([float(seed[0]), float(seed[1]), float(seed[2])])

# EVOLVE-BLOCK-END
