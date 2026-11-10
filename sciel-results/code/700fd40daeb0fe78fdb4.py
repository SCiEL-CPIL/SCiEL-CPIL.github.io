import time
import numpy as np
import random
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

def _local_search_fast(coeffs, time_limit=2.0, num_samples=2048):
    """
    Fast local search with incremental evaluation.
    Single and pair flips, using precomputed zs powers.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Precompute powers: zs_powers[k] = zs ** k for k in range(n)
    zs_powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        zs_powers[k] = zs_powers[k-1] * zs
    
    # Current evaluation
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * zs_powers[k]
    
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
            new_vals = current_vals - 2.0 * old_val * zs_powers[i]
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
                new_vals = current_vals - 2.0 * vi * zs_powers[i] - 2.0 * vj * zs_powers[j]
                new_score = _score(new_vals)
                if new_score < current_score - 1e-10:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
    
    return current, current_score

def _triple_flip_search(coeffs, time_limit=1.0, num_samples=2048):
    """Try triple flips for deeper local search."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    zs_powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        zs_powers[k] = zs_powers[k-1] * zs
    
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * zs_powers[k]
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
                    new_vals = (current_vals - 2.0*vi*zs_powers[i] 
                                - 2.0*vj*zs_powers[j] 
                                - 2.0*vk*zs_powers[k])
                    new_score = float(np.max(np.abs(new_vals)))
                    if new_score < current_score - 1e-10:
                        current[i] = -vi
                        current[j] = -vj
                        current[k] = -vk
                        current_vals = new_vals
                        current_score = new_score
                        improved = True
                        break  # restart from this point
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return current, current_score

def _soft_sup_norm(vals, beta=50.0):
    """Soft surrogate for sup norm using log-sum-exp on pre-computed magnitudes."""
    mag = np.abs(vals)
    max_mag = np.max(mag)
    if max_mag < 1e-10:
        return 0.0
    scaled = (mag - max_mag) * beta
    scaled = np.clip(scaled, -50, 50)
    lse = max_mag + np.log(np.sum(np.exp(scaled))) / beta
    return float(lse)

def _local_search_soft(coeffs, time_limit=2.0, num_samples=1024, beta=50.0):
    """Local search using soft surrogate for smoother landscape."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    zs_powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        zs_powers[k] = zs_powers[k-1] * zs
    
    current_vals = np.zeros(num_samples, dtype=complex)
    for k in range(n):
        current_vals += current[k] * zs_powers[k]
    
    current_score = _soft_sup_norm(current_vals, beta)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            new_vals = current_vals - 2.0 * old_val * zs_powers[i]
            new_score = _soft_sup_norm(new_vals, beta)
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
                new_vals = current_vals - 2.0 * vi * zs_powers[i] - 2.0 * vj * zs_powers[j]
                new_score = _soft_sup_norm(new_vals, beta)
                if new_score < current_score - 1e-8:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
    
    return current, current_score

def _known_optimal_24():
    """
    Known optimal ±1 polynomial for N=24 from Odlyzko's tables.
    One of the sequences achieving the minimum sup norm.
    """
    return [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1]

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    random.seed(12345)
    np.random.seed(12345)
    
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    # Generate diverse candidate seeds
    candidates = []
    
    # 1. Known optimal sequence and its symmetries
    ko = _known_optimal_24()
    candidates.append(ko)
    candidates.append([-x for x in ko])
    candidates.append(list(reversed(ko)))
    candidates.append([-x for x in reversed(ko)])
    
    # 2. Rudin-Shapiro and variations
    grs = _grs_sequence(n)
    candidates.append(grs)
    candidates.append(list(reversed(grs)))
    candidates.append([-x for x in grs])
    candidates.append([-x for x in reversed(grs)])
    
    # 3. Random seeds
    for _ in range(150):
        candidates.append([random.choice([-1, 1]) for _ in range(n)])
    
    # 4. Structured seeds (periodic patterns)
    for p in range(2, 13):
        for _ in range(3):
            base = [random.choice([-1, 1]) for _ in range(p)]
            candidates.append([base[i % p] for i in range(n)])
    
    # 5. Near-optimal patterns from literature
    patterns = [
        [1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,-1],
        [1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1],
        [-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1],
        [1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1],
    ]
    for pat in patterns:
        candidates.append(pat)
        candidates.append([-x for x in pat])
        candidates.append(list(reversed(pat)))
        candidates.append([-x for x in reversed(pat)])
    
    # Phase 1: Fast local search on all candidates
    for c in candidates:
        if time.time() > deadline - 40:
            break
        coeffs, score = _local_search_fast(c, time_limit=0.8, num_samples=2048)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 2: Soft surrogate search on top candidates
    if time.time() < deadline - 30:
        top_seeds = [best_coeffs] if best_coeffs else []
        for _ in range(30):
            top_seeds.append([random.choice([-1, 1]) for _ in range(n)])
        
        for c in top_seeds:
            if time.time() > deadline - 30:
                break
            coeffs, soft_score = _local_search_soft(c, time_limit=1.5, num_samples=1024, beta=40.0)
            # Evaluate with hard score
            angles = 2 * np.pi * np.arange(4096) / 4096
            zs = np.exp(1j * angles)
            vals = np.zeros(4096, dtype=complex)
            for k in range(n):
                vals += coeffs[k] * (zs ** k)
            hard_score = float(np.max(np.abs(vals)))
            if hard_score < best_score:
                best_score = hard_score
                best_coeffs = coeffs
    
    # Phase 3: More random restarts with fast search
    if time.time() < deadline - 20:
        for r in range(100):
            if time.time() > deadline - 20:
                break
            c = [random.choice([-1, 1]) for _ in range(n)]
            coeffs, score = _local_search_fast(c, time_limit=0.6, num_samples=2048)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    if best_coeffs is None:
        best_coeffs = _grs_sequence(n)
    
    # Phase 4: Triple flip refinement on best
    if time.time() < deadline - 10:
        coeffs, score = _triple_flip_search(best_coeffs, time_limit=5.0, num_samples=2048)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 5: High-precision polish
    current_best = list(best_coeffs)
    current_best_hp = _sup_norm_high_precision(current_best, num_samples=30000, refine=True)
    
    # Single flips with HP
    for i in range(n):
        if time.time() > deadline - 3:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
        if new_hp < current_best_hp - 1e-9:
            current_best_hp = new_hp
        else:
            current_best[i] = old_val
    
    # Pair flips with HP
    if time.time() < deadline - 2:
        for i in range(n):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2:
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
