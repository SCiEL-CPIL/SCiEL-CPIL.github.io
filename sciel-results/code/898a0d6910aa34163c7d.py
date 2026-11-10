import time
import numpy as np
from scipy.optimize import minimize_scalar

def _sup_norm_high_precision(coeffs, num_samples=50000, refine=True):
    """
    Compute sup norm with high precision.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    if not refine:
        return float(np.max(mag))
    
    top_k = 30
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    def eval_poly(z):
        res = 0.0
        for c in reversed(coeffs):
            res = res * z + c
        return res

    for idx in peak_indices:
        center = angles[idx]
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(eval_poly(z))
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-14})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except:
            pass
            
    return best_val

def _eval_poly_fast(coeffs, zs):
    """Evaluate polynomial at all points using Horner's method."""
    vals = np.zeros(len(zs), dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return vals

def _local_search_aggressive(coeffs, zs, time_limit=2.0):
    """
    Aggressive local search with single, pair, and triple flips.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    num_samples = len(zs)
    
    current_vals = _eval_poly_fast(current, zs)
    current_score = float(np.max(np.abs(current_vals)))
    
    # Precompute powers for fast flip evaluation
    powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        powers[k] = powers[k-1] * zs
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # Single flips
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            new_vals = current_vals - 2 * old_val * powers[i]
            new_score = float(np.max(np.abs(new_vals)))
            if new_score < current_score - 1e-9:
                current[i] = -old_val
                current_vals = new_vals
                current_score = new_score
                improved = True
        
        # Pair flips
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                vi, vj = current[i], current[j]
                new_vals = current_vals - 2 * vi * powers[i] - 2 * vj * powers[j]
                new_score = float(np.max(np.abs(new_vals)))
                if new_score < current_score - 1e-9:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
                    
    return current, current_score

def _grs_sequence(length=24):
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    deadline = time.time() + 110
    
    num_samples = 4096
    n = 24
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Precompute powers
    powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        powers[k] = powers[k-1] * zs
    
    def fast_score(coeffs):
        vals = np.zeros(num_samples, dtype=complex)
        for k, c in enumerate(coeffs):
            if c == 1:
                vals += powers[k]
            else:
                vals -= powers[k]
        return float(np.max(np.abs(vals)))
    
    def local_search_fast(coeffs, time_limit=1.5):
        current = list(coeffs)
        current_vals = np.zeros(num_samples, dtype=complex)
        for k, c in enumerate(current):
            if c == 1:
                current_vals += powers[k]
            else:
                current_vals -= powers[k]
        current_score = float(np.max(np.abs(current_vals)))
        
        start_time = time.time()
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                old_val = current[i]
                new_vals = current_vals - 2 * old_val * powers[i]
                new_score = float(np.max(np.abs(new_vals)))
                if new_score < current_score - 1e-9:
                    current[i] = -old_val
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
            if time.time() - start_time > time_limit:
                break
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    new_vals = current_vals - 2 * vi * powers[i] - 2 * vj * powers[j]
                    new_score = float(np.max(np.abs(new_vals)))
                    if new_score < current_score - 1e-9:
                        current[i] = -vi
                        current[j] = -vj
                        current_vals = new_vals
                        current_score = new_score
                        improved = True
        return current, current_score
    
    import random
    random.seed(42)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: GRS and variations
    grs = _grs_sequence(24)
    initial_candidates = [
        grs,
        list(reversed(grs)),
        [-c for c in grs],
        [-c for c in reversed(grs)],
    ]
    
    for c in initial_candidates:
        if time.time() > deadline - 40:
            break
        coeffs, score = local_search_fast(c, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)
    
    # Phase 2: Many random restarts
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        c = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = local_search_fast(c, time_limit=0.8)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)
    
    if best_coeffs is None:
        best_coeffs = grs
    
    # Phase 3: High-precision polish
    current_best = list(best_coeffs)
    current_best_hp = _sup_norm_high_precision(current_best, num_samples=30000, refine=True)
    
    for i in range(n):
        if time.time() > deadline - 10:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
        if new_hp < current_best_hp - 1e-9:
            current_best_hp = new_hp
        else:
            current_best[i] = old_val
    
    if time.time() < deadline - 8:
        for i in range(n):
            if time.time() > deadline - 8:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 8:
                    break
                vi, vj = current_best[i], current_best[j]
                current_best[i] = -vi
                current_best[j] = -vj
                new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
                if new_hp < current_best_hp - 1e-9:
                    current_best_hp = new_hp
                else:
                    current_best[i] = vi
                    current_best[j] = vj

    return current_best
