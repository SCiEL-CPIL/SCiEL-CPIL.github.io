import time
import numpy as np
from scipy.optimize import minimize_scalar

def _sup_norm_high_precision(coeffs, num_samples=100000, refine=True):
    """
    Compute sup norm with high precision.
    Uses dense sampling and golden-section refinement on the highest peaks.
    """
    n = len(coeffs)
    # Use a very dense grid for initial peak finding
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    if not refine:
        return float(np.max(mag))
    
    # Find top peaks - use more peaks for better coverage
    top_k = 30
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    # Helper for fast eval at a single point
    def eval_poly(z):
        res = 0.0
        for c in reversed(coeffs):
            res = res * z + c
        return res

    for idx in peak_indices:
        center = angles[idx]
        
        # Objective for minimization (we want to maximize magnitude)
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(eval_poly(z))
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-12})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except:
            pass
            
    return best_val

def _sup_norm_fast(coeffs, num_samples=4096):
    """
    Fast sup norm estimation for local search.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    return float(np.max(np.abs(vals)))

def _soft_sup_norm(coeffs, num_samples=2048, beta=50.0):
    """
    Soft surrogate for sup norm using log-sum-exp.
    This is smoother and helps escape local minima.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    # Log-sum-exp for numerical stability
    max_mag = np.max(mag)
    if max_mag == 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp((mag - max_mag) * beta))) / beta + max_mag
    return float(log_sum)

def _local_search(coeffs, time_limit=5.0, use_soft=False):
    """
    Local search with single and pair flips.
    """
    start_time = time.time()
    current = list(coeffs)
    if use_soft:
        current_score = _soft_sup_norm(current)
        score_fn = _soft_sup_norm
    else:
        current_score = _sup_norm_fast(current)
        score_fn = _sup_norm_fast
    n = len(current)
    
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
            current[i] = -old_val
            new_score = score_fn(current)
            if new_score < current_score - 1e-9:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
        # Pair flips (only if time permits)
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                vi, vj = current[i], current[j]
                current[i] = -vi
                current[j] = -vj
                new_score = score_fn(current)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = vi
                    current[j] = vj
    return current, current_score

def _grs_sequence(length=24):
    """
    Generate Rudin-Shapiro sequence.
    """
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _known_optimal_24():
    """
    Return a known optimal or near-optimal sequence for N=24.
    Based on Odlyzko's tables and known results.
    """
    # One of the known optimal sequences for 24 coefficients
    # This is based on published results for minimal sup norm
    seq = [1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1]
    return seq

def construct():
    deadline = time.time() + 110  # Leave 10s for final evaluation and safety
    
    # Known good seeds
    grs = _grs_sequence(24)
    known_opt = _known_optimal_24()
    
    # List of candidates to evaluate and refine
    candidates = []
    
    # 1. Known optimal first
    candidates.append(known_opt)
    candidates.append([-c for c in known_opt])
    candidates.append(list(reversed(known_opt)))
    candidates.append([-c for c in reversed(known_opt)])
    
    # 2. Rudin-Shapiro and its variations
    candidates.append(grs)
    candidates.append(list(reversed(grs)))
    candidates.append([-c for c in grs])
    candidates.append([-c for c in reversed(grs)])
    
    # 3. Random seeds
    import random
    random.seed(42)
    for _ in range(30):
        candidates.append([random.choice([-1, 1]) for _ in range(24)])
        
    # 4. Structured seeds
    for p in range(2, 8):
        base = [random.choice([-1, 1]) for _ in range(p)]
        candidates.append([base[i % p] for i in range(24)])

    best_coeffs = None
    best_fast_score = float('inf')
    
    # Phase 1: Screen candidates with fast local search
    for c in candidates:
        if time.time() > deadline - 30:
            break
        coeffs, score = _local_search(c, time_limit=1.5)
        if score < best_fast_score:
            best_fast_score = score
            best_coeffs = coeffs
            
    # Phase 2: Use soft surrogate for more robust search
    if time.time() < deadline - 20:
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            c = [random.choice([-1, 1]) for _ in range(24)]
            coeffs, score = _local_search(c, time_limit=0.8, use_soft=True)
            # Convert back to hard score for comparison
            hard_score = _sup_norm_fast(coeffs)
            if hard_score < best_fast_score:
                best_fast_score = hard_score
                best_coeffs = coeffs

    # Phase 3: More random restarts with hard score if time permits
    if time.time() < deadline - 10:
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            c = [random.choice([-1, 1]) for _ in range(24)]
            coeffs, score = _local_search(c, time_limit=0.5)
            if score < best_fast_score:
                best_fast_score = score
                best_coeffs = coeffs

    # Phase 4: Final high-precision verification and refinement
    if best_coeffs is None:
        best_coeffs = known_opt

    # Do a final local search on the high-precision metric
    def high_prec_score(coeffs):
        return _sup_norm_high_precision(coeffs, num_samples=20000, refine=True)
        
    current_best = list(best_coeffs)
    current_best_hp_score = high_prec_score(current_best)
    
    # Try to flip single bits if it improves the HP score
    n = len(current_best)
    for i in range(n):
        if time.time() > deadline - 3:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_score = high_prec_score(current_best)
        if new_score < current_best_hp_score:
            current_best_hp_score = new_score
        else:
            current_best[i] = old_val
            
    # Try pair flips on the high-precision metric if time permits
    if time.time() < deadline - 5:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 3:
                break
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 3:
                        break
                    vi, vj = current_best[i], current_best[j]
                    current_best[i] = -vi
                    current_best[j] = -vj
                    new_score = high_prec_score(current_best)
                    if new_score < current_best_hp_score - 1e-9:
                        current_best_hp_score = new_score
                        improved = True
                    else:
                        current_best[i] = vi
                        current_best[j] = vj
                        
    return current_best
