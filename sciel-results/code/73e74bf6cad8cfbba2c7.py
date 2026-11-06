import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 150.0
_TIME_MARGIN = 15.0

def _time_left():
    return _LIMIT - _TIME_MARGIN - (time.time() - _T0)

def _eval_peak_sq(coeffs, Z, theta, M):
    """Evaluate the maximum |P(z)|^2 on the grid for given coefficients."""
    vals = coeffs @ Z.T
    mag2 = np.abs(vals) ** 2
    peak_sq = np.max(mag2)
    peak_idx = np.argmax(mag2)
    return peak_sq, peak_idx

def _refine_peak(coeffs, peak_val_sq, phase, M_grid=512, iters=30):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: array of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared (for comparison)
    phase: initial phase estimate in radians
    Returns refined peak magnitude squared.
    """
    c_arr = coeffs
    powers = np.arange(_N)
    
    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

    d_phase = 2.0 * math.pi / M_grid
    lo = phase - d_phase
    hi = phase + d_phase
    
    f_lo = f(lo)
    f_hi = f(hi)
    
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    
    for _ in range(iters):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            hi = m2
            f_hi = f2
        else:
            lo = m1
            f_lo = f1
            
    peak2 = max(f_lo, f_hi)
    return max(peak2, peak_val_sq)

def _local_search(coeffs, Z, theta, M, max_flips=3, time_budget=30.0):
    """
    Perform local search with single, double, and triple flips.
    Returns improved coefficients and peak_sq.
    """
    cur_coeffs = coeffs.copy()
    cur_peak_sq, _ = _eval_peak_sq(cur_coeffs, Z, theta, M)
    
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        
        # Try single flips
        for j in range(_N):
            if _time_left() < 1.0:
                return cur_coeffs, cur_peak_sq
            cur_coeffs[j] = -cur_coeffs[j]
            new_peak_sq, new_peak_idx = _eval_peak_sq(cur_coeffs, Z, theta, M)
            if new_peak_sq < cur_peak_sq * 0.999:  # Require meaningful improvement
                new_phase = theta[new_peak_idx]
                refined_sq = _refine_peak(cur_coeffs, new_peak_sq, new_phase, M_grid=M)
                if refined_sq < cur_peak_sq:
                    cur_peak_sq = refined_sq
                    improved = True
                    break
            else:
                cur_coeffs[j] = -cur_coeffs[j]
        
        if improved:
            continue
        
        # Try double flips
        for j in range(_N):
            if _time_left() < 1.0:
                return cur_coeffs, cur_peak_sq
            for k2 in range(j + 1, _N):
                if _time_left() < 0.5:
                    return cur_coeffs, cur_peak_sq
                cur_coeffs[j] = -cur_coeffs[j]
                cur_coeffs[k2] = -cur_coeffs[k2]
                new_peak_sq, new_peak_idx = _eval_peak_sq(cur_coeffs, Z, theta, M)
                if new_peak_sq < cur_peak_sq * 0.999:
                    new_phase = theta[new_peak_idx]
                    refined_sq = _refine_peak(cur_coeffs, new_peak_sq, new_phase, M_grid=M)
                    if refined_sq < cur_peak_sq:
                        cur_peak_sq = refined_sq
                        improved = True
                        break
                cur_coeffs[j] = -cur_coeffs[j]
                cur_coeffs[k2] = -cur_coeffs[k2]
            if improved:
                break
        
        if improved:
            continue
        
        # Try triple flips
        if max_flips >= 3:
            for j in range(_N):
                if _time_left() < 1.0:
                    return cur_coeffs, cur_peak_sq
                for k2 in range(j + 1, _N):
                    if _time_left() < 0.5:
                        return cur_coeffs, cur_peak_sq
                    for l2 in range(k2 + 1, _N):
                        if _time_left() < 0.2:
                            return cur_coeffs, cur_peak_sq
                        cur_coeffs[j] = -cur_coeffs[j]
                        cur_coeffs[k2] = -cur_coeffs[k2]
                        cur_coeffs[l2] = -cur_coeffs[l2]
                        new_peak_sq, new_peak_idx = _eval_peak_sq(cur_coeffs, Z, theta, M)
                        if new_peak_sq < cur_peak_sq * 0.999:
                            new_phase = theta[new_peak_idx]
                            refined_sq = _refine_peak(cur_coeffs, new_peak_sq, new_phase, M_grid=M)
                            if refined_sq < cur_peak_sq:
                                cur_peak_sq = refined_sq
                                improved = True
                                break
                        cur_coeffs[j] = -cur_coeffs[j]
                        cur_coeffs[k2] = -cur_coeffs[k2]
                        cur_coeffs[l2] = -cur_coeffs[l2]
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return cur_coeffs, cur_peak_sq

def _genetic_algorithm(initial_coeffs, Z, theta, M, pop_size=200, generations=100, time_budget=60.0):
    """
    Genetic algorithm to explore the solution space.
    """
    rng = np.random.RandomState(42)
    
    # Initialize population
    population = [initial_coeffs.copy()]
    for _ in range(pop_size - 1):
        c = rng.choice([-1, 1], size=_N).astype(np.float64)
        population.append(c)
    
    # Evaluate initial population
    scores = []
    for c in population:
        peak_sq, _ = _eval_peak_sq(c, Z, theta, M)
        scores.append(peak_sq)
    scores = np.array(scores)
    
    best_idx = np.argmin(scores)
    best_coeffs = population[best_idx].copy()
    best_score = scores[best_idx]
    
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    
    for gen in range(generations):
        if _time_left() < 5.0:
            break
        
        # Tournament selection
        def tournament_select():
            idxs = rng.randint(0, pop_size, size=3)
            return idxs[np.argmin(scores[idxs])]
        
        # Create new population
        new_population = [best_coeffs.copy()]  # Elitism
        
        for _ in range(pop_size - 1):
            if _time_left() < 2.0:
                break
            
            # Selection
            i1 = tournament_select()
            i2 = tournament_select()
            p1 = population[i1]
            p2 = population[i2]
            
            # Uniform crossover
            mask = rng.rand(_N) < 0.5
            child = np.where(mask, p1, p2).astype(np.float64)
            
            # Mutation (flip each bit with probability 0.1)
            mut_mask = rng.rand(_N) < 0.1
            child[mut_mask] = -child[mut_mask]
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for c in new_population:
            peak_sq, _ = _eval_peak_sq(c, Z, theta, M)
            new_scores.append(peak_sq)
        new_scores = np.array(new_scores)
        
        # Update best
        new_best_idx = np.argmin(new_scores)
        if new_scores[new_best_idx] < best_score:
            best_score = new_scores[new_best_idx]
            best_coeffs = new_population[new_best_idx].copy()
        
        population = new_population
        scores = new_scores
    
    return best_coeffs, best_score

def construct():
    global _T0
    _T0 = time.time()
    
    M = 512
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    # Phase 1: Partial brute force to get a good starting point
    B = 10000
    total = 1 << _N
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase = 0.0
    
    # Search a subset of candidates (first ~500k to get a good start)
    max_candidates = 500000
    for start in range(0, min(total, max_candidates), B):
        if _time_left() < 10.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = bits.astype(np.float64) * 2.0 - 1.0
        
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_peak_sq:
            best_peak_sq = min_peak_sq
            best_coeff_idx = start + min_idx
            best_peak_phase = theta[np.argmax(mag2[min_idx])]
    
    # Retrieve best coefficients from brute force
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = bits.astype(np.float64) * 2.0 - 1.0
    
    # Refine the peak
    true_peak_sq = _refine_peak(best_coeffs, best_peak_sq, best_peak_phase, M_grid=M)
    best_peak_sq = true_peak_sq
    
    # Phase 2: Local search from the best candidate
    best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=30.0)
    
    # Phase 3: Genetic algorithm for broader exploration
    if _time_left() > 20.0:
        ga_coeffs, ga_peak_sq = _genetic_algorithm(best_coeffs, Z, theta, M, pop_size=150, generations=50, time_budget=40.0)
        if ga_peak_sq < best_peak_sq:
            best_coeffs = ga_coeffs
            best_peak_sq = ga_peak_sq
            # Local search from GA result
            best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=20.0)
    
    # Final refinement
    _, peak_idx = _eval_peak_sq(best_coeffs, Z, theta, M)
    final_peak_sq = _refine_peak(best_coeffs, best_peak_sq, theta[peak_idx], M_grid=M, iters=40)
    best_peak_sq = final_peak_sq
    
    # Convert back to list of ints
    result = [int(1 if c > 0 else -1) for c in best_coeffs]
    
    return result
