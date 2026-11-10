import time
import numpy as np
from scipy.optimize import minimize_scalar

def _sup_norm_high_precision(coeffs, num_samples=50000, refine=True):
    """Compute sup norm with high precision using dense sampling + golden-section refinement."""
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

def _grs_sequence(length=24):
    """Generate Rudin-Shapiro sequence."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _eval_decomposed(coeffs, num_samples):
    """
    Evaluate P(z) = A(z) + z^12 * B(z) where A uses first 12 coeffs, B uses last 12.
    Returns complex array of length num_samples.
    """
    n = len(coeffs)
    half = n // 2
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # A(z) = sum_{k=0}^{half-1} c_k * z^k
    # B(z) = sum_{k=0}^{half-1} c_{half+k} * z^k
    # P(z) = A(z) + z^half * B(z)
    
    A = np.zeros(num_samples, dtype=complex)
    for k in range(half):
        A += coeffs[k] * (zs ** k)
    
    B = np.zeros(num_samples, dtype=complex)
    for k in range(half):
        B += coeffs[half + k] * (zs ** k)
    
    zhalf = zs ** half
    return A + zhalf * B

def _local_search_decomposed(coeffs, time_limit=2.0, num_samples=4096):
    """
    Fast local search using decomposition P = A + z^12 * B.
    Precompute A and B contributions for each coefficient.
    Single and pair flips.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    half = n // 2
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Precompute powers for first half: zs^k for k in range(half)
    zs_powers = np.ones((half, num_samples), dtype=complex)
    for k in range(1, half):
        zs_powers[k] = zs_powers[k-1] * zs
    
    zhalf = zs ** half  # z^12
    
    # Precompute contribution of each coefficient
    # For coeff k < half: contribution is c_k * zs^k
    # For coeff k >= half: contribution is c_k * zs^(k-half) * zhalf
    contrib = np.zeros((n, num_samples), dtype=complex)
    for k in range(half):
        contrib[k] = zs_powers[k]
    for k in range(half, n):
        contrib[k] = zs_powers[k - half] * zhalf
    
    # Current evaluation
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * contrib[k]
    
    current_score = float(np.max(np.abs(current_vals)))
    
    def _score(vals):
        return float(np.max(np.abs(vals)))
    
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
            new_vals = current_vals - 2.0 * old_val * contrib[i]
            new_score = _score(new_vals)
            if new_score < current_score - 1e-10:
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
                new_vals = current_vals - 2.0 * vi * contrib[i] - 2.0 * vj * contrib[j]
                new_score = _score(new_vals)
                if new_score < current_score - 1e-10:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
    
    return current, current_score

def _triple_flip_decomposed(coeffs, time_limit=2.0, num_samples=4096):
    """Triple flip search using decomposition."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    half = n // 2
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    zs_powers = np.ones((half, num_samples), dtype=complex)
    for k in range(1, half):
        zs_powers[k] = zs_powers[k-1] * zs
    
    zhalf = zs ** half
    
    contrib = np.zeros((n, num_samples), dtype=complex)
    for k in range(half):
        contrib[k] = zs_powers[k]
    for k in range(half, n):
        contrib[k] = zs_powers[k - half] * zhalf
    
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * contrib[k]
    current_score = float(np.max(np.abs(current_vals)))
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, n):
                if time.time() - start_time > time_limit:
                    break
                for k in range(j+1, n):
                    if time.time() - start_time > time_limit:
                        break
                    vi, vj, vk = current[i], current[j], current[k]
                    new_vals = (current_vals - 2.0*vi*contrib[i] 
                                - 2.0*vj*contrib[j] 
                                - 2.0*vk*contrib[k])
                    new_score = float(np.max(np.abs(new_vals)))
                    if new_score < current_score - 1e-10:
                        current[i] = -vi
                        current[j] = -vj
                        current[k] = -vk
                        current_vals = new_vals
                        current_score = new_score
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return current, current_score

def _soft_local_search_decomposed(coeffs, time_limit=2.0, num_samples=2048, beta=50.0):
    """Local search using soft surrogate with decomposition."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    half = n // 2
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    zs_powers = np.ones((half, num_samples), dtype=complex)
    for k in range(1, half):
        zs_powers[k] = zs_powers[k-1] * zs
    
    zhalf = zs ** half
    
    contrib = np.zeros((n, num_samples), dtype=complex)
    for k in range(half):
        contrib[k] = zs_powers[k]
    for k in range(half, n):
        contrib[k] = zs_powers[k - half] * zhalf
    
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * contrib[k]
    
    def _soft_score(vals):
        mag = np.abs(vals)
        max_mag = np.max(mag)
        if max_mag < 1e-10:
            return 0.0
        scaled = (mag - max_mag) * beta
        scaled = np.clip(scaled, -50, 50)
        return float(max_mag + np.log(np.sum(np.exp(scaled))) / beta)
    
    current_score = _soft_score(current_vals)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            new_vals = current_vals - 2.0 * old_val * contrib[i]
            new_score = _soft_score(new_vals)
            if new_score < current_score - 1e-8:
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
                new_vals = current_vals - 2.0 * vi * contrib[i] - 2.0 * vj * contrib[j]
                new_score = _soft_score(new_vals)
                if new_score < current_score - 1e-8:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
    
    return current, current_score

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    import random
    random.seed(12345)
    np.random.seed(12345)
    
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: Start with GRS and its variations - these are known to be good
    grs = _grs_sequence(n)
    
    # Run local search from GRS first (highest priority)
    coeffs, score = _local_search_decomposed(grs, time_limit=3.0, num_samples=4096)
    best_score = score
    best_coeffs = coeffs
    
    # Also try reversed and negated GRS
    for variant in [list(reversed(grs)), [-x for x in grs], [-x for x in reversed(grs)]]:
        if time.time() > deadline - 60:
            break
        c, s = _local_search_decomposed(variant, time_limit=2.0, num_samples=4096)
        if s < best_score:
            best_score = s
            best_coeffs = c
    
    # Phase 2: Many random restarts with fast local search
    num_random = 80
    for r in range(num_random):
        if time.time() > deadline - 50:
            break
        c = [random.choice([-1, 1]) for _ in range(n)]
        coeffs, score = _local_search_decomposed(c, time_limit=1.5, num_samples=4096)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 3: Soft surrogate search to escape local minima
    if time.time() < deadline - 30:
        # Start from best found
        top_seeds = [best_coeffs]
        # Add some random seeds
        for _ in range(20):
            top_seeds.append([random.choice([-1, 1]) for _ in range(n)])
        
        for c in top_seeds:
            if time.time() > deadline - 30:
                break
            coeffs, soft_score = _soft_local_search_decomposed(c, time_limit=2.0, num_samples=2048, beta=40.0)
            # Evaluate with hard score using higher resolution
            hard_vals = _eval_decomposed(coeffs, 4096)
            hard_score = float(np.max(np.abs(hard_vals)))
            if hard_score < best_score:
                best_score = hard_score
                best_coeffs = coeffs
    
    # Phase 4: More random restarts if time remains
    if time.time() < deadline - 20:
        for r in range(60):
            if time.time() > deadline - 20:
                break
            c = [random.choice([-1, 1]) for _ in range(n)]
            coeffs, score = _local_search_decomposed(c, time_limit=1.0, num_samples=4096)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    if best_coeffs is None:
        best_coeffs = grs
    
    # Phase 5: Triple flip refinement on best
    if time.time() < deadline - 15:
        coeffs, score = _triple_flip_decomposed(best_coeffs, time_limit=8.0, num_samples=4096)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 6: High-precision polish with single and pair flips
    current_best = list(best_coeffs)
    current_best_hp = _sup_norm_high_precision(current_best, num_samples=30000, refine=True)
    
    # Single flips with HP
    for i in range(n):
        if time.time() > deadline - 4:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_hp = _sup_norm_high_precision(current_best, num_samples=20000, refine=True)
        if new_hp < current_best_hp - 1e-9:
            current_best_hp = new_hp
        else:
            current_best[i] = old_val
    
    # Pair flips with HP
    if time.time() < deadline - 3:
        for i in range(n):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    break
                vi, vj = current_best[i], current_best[j]
                current_best[i] = -vi
                current_best[j] = -vj
                new_hp = _sup_norm_high_precision(current_best, num_samples=20000, refine=True)
                if new_hp < current_best_hp - 1e-9:
                    current_best_hp = new_hp
                else:
                    current_best[i] = vi
                    current_best[j] = vj
    
    return current_best
