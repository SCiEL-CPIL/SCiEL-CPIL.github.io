import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 150.0

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak(coeffs, M=2048, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _calc_peak_fast(coeffs, M=2048, refine_top=5, refine_iters=20):
    """
    Faster peak calculation for local search iterations.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _local_search(seed, M=2048, max_time=140.0):
    """
    Perform local search to minimize the peak.
    Uses fast peak calculation for speed, with final verification.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M, refine_top=5, refine_iters=20)
    best = list(cur)
    best_peak = cur_peak
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Single flip pass
        for j in range(_N):
            if time.time() - start_time > max_time:
                break
            cur[j] = -cur[j]
            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
                if cur_peak < best_peak - 1e-12:
                    best = list(cur)
                    best_peak = cur_peak
            else:
                cur[j] = -cur[j]
        
        if not improved:
            # Try double flips
            double_improved = False
            for j in range(_N):
                if time.time() - start_time > max_time:
                    break
                for k in range(j + 1, _N):
                    if time.time() - start_time > max_time:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        double_improved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
            
            if not double_improved:
                # Try triple flips if stuck
                triple_improved = False
                for j in range(_N):
                    if time.time() - start_time > max_time:
                        break
                    for k in range(j + 1, _N):
                        if time.time() - start_time > max_time:
                            break
                        for l in range(k + 1, _N):
                            if time.time() - start_time > max_time:
                                break
                            cur[j] = -cur[j]
                            cur[k] = -cur[k]
                            cur[l] = -cur[l]
                            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
                            if p < cur_peak - 1e-12:
                                cur_peak = p
                                triple_improved = True
                                if cur_peak < best_peak - 1e-12:
                                    best = list(cur)
                                    best_peak = cur_peak
                            else:
                                cur[j] = -cur[j]
                                cur[k] = -cur[k]
                                cur[l] = -cur[l]
                
                if not triple_improved:
                    break
                else:
                    improved = True
            else:
                improved = True
        
        # Periodic high-accuracy check
        if time.time() - start_time < max_time - 5.0:
            high_acc_peak = _calc_peak(best, M=4096, refine_top=10, refine_iters=40)
            if high_acc_peak < best_peak:
                best_peak = high_acc_peak
    
    # Final high-accuracy verification
    final_peak = _calc_peak(best, M=8192, refine_top=10, refine_iters=50)
    return best, final_peak

def _genetic_search(pop_size=50, generations=100, max_time=120.0, M=1024):
    """
    Genetic Algorithm for exploring the solution space.
    Fitness is the negative peak magnitude.
    """
    start_time = time.time()
    
    # Initialize population with RS seed and its variations
    rng = np.random.default_rng(42)
    rs_seed = _rs_poly(_N)
    
    population = [list(rs_seed)]
    for _ in range(pop_size - 1):
        # Start with RS and flip a few random positions
        individual = list(rs_seed)
        num_flips = rng.integers(1, 6)
        flip_indices = rng.choice(_N, size=num_flips, replace=False)
        for idx in flip_indices:
            individual[idx] = -individual[idx]
        population.append(individual)
    
    # Also add some purely random individuals
    for _ in range(pop_size // 4):
        population.append(rng.choice([-1, 1], size=_N).tolist())
    
    best_overall = list(population[0])
    best_peak = _calc_peak_fast(population[0], M, refine_top=5, refine_iters=20)
    
    # Evaluate initial population
    peaks = [best_peak]
    for i in range(1, pop_size):
        if time.time() - start_time > max_time:
            break
        p = _calc_peak_fast(population[i], M, refine_top=3, refine_iters=15)
        peaks.append(p)
        if p < best_peak:
            best_peak = p
            best_overall = list(population[i])
    
    # Genetic algorithm loop
    for gen in range(generations):
        if time.time() - start_time > max_time - 10.0:
            break
        
        # Sort by fitness (lower peak is better)
        sorted_indices = np.argsort(peaks)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_peaks = [peaks[i] for i in sorted_indices]
        
        # Elitism: keep top 20%
        elite_count = max(1, pop_size // 5)
        new_population = [list(ind) for ind in sorted_population[:elite_count]]
        new_peaks = sorted_peaks[:elite_count]
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > max_time - 10.0:
                break
            
            # Tournament selection
            def tournament_select():
                candidates = rng.integers(0, pop_size, size=3)
                best_idx = candidates[np.argmin([peaks[i] for i in candidates])]
                return population[best_idx]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: uniform
            mask = rng.integers(0, 2, size=_N)
            child = [parent1[i] if mask[i] == 1 else parent2[i] for i in range(_N)]
            
            # Mutation: flip a few random bits
            mutation_rate = 0.1
            for i in range(_N):
                if rng.random() < mutation_rate:
                    child[i] = -child[i]
            
            # Evaluate child
            p = _calc_peak_fast(child, M, refine_top=3, refine_iters=15)
            new_population.append(child)
            new_peaks.append(p)
            
            if p < best_peak:
                best_peak = p
                best_overall = list(child)
        
        population = new_population
        peaks = new_peaks
    
    return best_overall, best_peak

def _exhaustive_search(M=2048, refine_top=5, refine_iters=20):
    """
    Exhaustive search over equivalence classes accounting for sign and reversal symmetry.
    Fix c[0] = +1 (sign symmetry), then enumerate 2^11 patterns for positions 1-11,
    and for each, check both the pattern and its reversal for positions 12-23.
    """
    start_time = time.time()
    best_coeffs = None
    best_peak = float('inf')
    
    # Precompute for fast evaluation
    powers = np.arange(_N)
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    Z = np.exp(1j * np.outer(theta, powers))
    
    # We fix c[0] = 1. For positions 1-11, we enumerate 2^11 = 2048 patterns.
    # For each pattern, we consider two completions:
    #   1. Direct: c[12:24] = pattern
    #   2. Reversed: c[12:24] = reversed(pattern)
    # This accounts for the reversal symmetry (reversing the entire polynomial
    # maps to another valid polynomial with the same peak).
    
    n_patterns = 1 << 11  # 2048 patterns for positions 1-11
    
    for mask in range(n_patterns):
        if time.time() - start_time > _LIMIT - 15.0:
            break
        
        # Build the first half: c[0] = 1, c[1:12] from mask
        coeffs = [1] * _N
        for bit in range(11):
            if (mask >> bit) & 1:
                coeffs[1 + bit] = -1
        
        # Completion 1: direct
        for bit in range(11):
            if (mask >> bit) & 1:
                coeffs[12 + bit] = -1
        
        c = np.array(coeffs, dtype=complex)
        vals = Z @ c
        mag2 = np.abs(vals) ** 2
        peak_val = np.max(mag2)
        
        if peak_val < best_peak:
            best_peak = peak_val
            best_coeffs = list(coeffs)
        
        # Completion 2: reversed
        coeffs2 = [1] * _N
        for bit in range(11):
            if (mask >> bit) & 1:
                coeffs2[1 + bit] = -1
        for bit in range(11):
            if (mask >> (10 - bit)) & 1:
                coeffs2[12 + bit] = -1
        
        c2 = np.array(coeffs2, dtype=complex)
        vals2 = Z @ c2
        mag2_2 = np.abs(vals2) ** 2
        peak_val2 = np.max(mag2_2)
        
        if peak_val2 < best_peak:
            best_peak = peak_val2
            best_coeffs = list(coeffs2)
    
    if best_coeffs is None:
        best_coeffs = _rs_poly(_N)
        best_peak = _calc_peak_fast(best_coeffs, M, refine_top=5, refine_iters=20)
    
    # Refine the best with golden-section
    refined_peak = _calc_peak(best_coeffs, M=4096, refine_top=10, refine_iters=40)
    return best_coeffs, refined_peak

def construct():
    global _T0
    _T0 = time.time()
    
    # Primary approach: Exhaustive search over equivalence classes
    # This should find the global optimum since we're checking all 2^11 * 2 = 4096
    # distinct polynomials up to sign and reversal symmetry.
    
    best_coeffs, best_peak = _exhaustive_search(M=2048, refine_top=5, refine_iters=20)
    
    # If time remains, try local search to see if we can improve further
    # (though exhaustive search should already be optimal)
    if _time_left() > 20.0:
        res_coeffs, res_peak = _local_search(best_coeffs, M=2048, max_time=min(30.0, _time_left() - 10.0))
        if res_peak < best_peak:
            best_coeffs = res_coeffs
            best_peak = res_peak
    
    # Final high-precision verification
    if _time_left() > 5.0:
        final_peak = _calc_peak(best_coeffs, M=8192, refine_top=10, refine_iters=50)
        best_peak = final_peak
    
    # Ensure coefficients are +1 or -1
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=8192, refine_top=10, refine_iters=50))
