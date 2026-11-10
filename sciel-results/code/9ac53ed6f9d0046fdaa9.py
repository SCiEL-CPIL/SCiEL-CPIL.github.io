import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    def autocorrelation_zeros(coeffs):
        """Use FFT to compute autocorrelation and count near-zero lags."""
        n = len(coeffs)
        # Pad to power of 2 for FFT efficiency
        size = 1
        while size < 2 * n - 1:
            size *= 2
        f = np.fft.rfft(coeffs, n=size)
        ac = np.fft.irfft(f * np.conj(f), n=size)
        # The autocorrelation has 2n-1 meaningful values
        ac = ac[:2*n-1]
        # Count lags where autocorrelation is close to zero (indicating potential cancellation)
        # Use a threshold relative to max magnitude
        max_ac = np.max(np.abs(ac))
        if max_ac == 0:
            return 2*n-1
        threshold = max_ac * 0.01
        zeros = np.count_nonzero(np.abs(ac) < threshold)
        return int(zeros)
    
    best_score = float('inf')
    best_poly = [1] * 12
    
    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)
    
    try_poly([1] * 12)
    
    # STRUCTURAL CHANGE: Use FFT-based spectral analysis to guide search
    # Key insight: polynomials with many autocorrelation zeros have cancellation potential
    # We'll generate candidates, score them by autocorrelation zeros, then refine top ones
    
    random.seed(42)
    
    # Phase 1: Generate diverse candidates and rank by autocorrelation zeros
    candidates = []
    
    # Structured families
    # Family A: Symmetric/antisymmetric patterns
    for k in range(1, 7):
        for s in [1, -1]:
            poly = [0]*12
            for i in range(12):
                if i < k:
                    poly[i] = s
                elif i < 2*k:
                    poly[i] = -s
                elif i < 3*k:
                    poly[i] = s
                else:
                    poly[i] = s * (1 + (i % 2))
            if sum(1 for c in poly if c != 0) == 12 and all(-64 <= c <= 64 for c in poly):
                candidates.append(poly)
    
    # Family B: Periodic with varying amplitudes
    for period in [2, 3, 4, 6]:
        for amp1 in range(1, 5):
            for amp2 in range(1, 5):
                for offset in range(-2, 3):
                    poly = []
                    for i in range(12):
                        phase = (i % period) / period * 2 * np.pi
                        val = int(amp1 * np.cos(phase) + amp2 * np.sin(phase) + offset)
                        if val == 0:
                            val = 1
                        poly.append(val)
                    if all(-64 <= c <= 64 for c in poly) and sum(1 for c in poly if c != 0) == 12:
                        candidates.append(poly)
    
    # Family C: Random with small coefficients
    for _ in range(2000):
        poly = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
        candidates.append(poly)
    
    # Family D: Binary sign patterns with magnitude scaling
    for _ in range(1000):
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.randint(1, 4) for _ in range(12)]
        poly = [signs[i] * mags[i] for i in range(12)]
        candidates.append(poly)
    
    # Score candidates by autocorrelation zeros (proxy for cancellation potential)
    # Higher zeros = more potential for sparse square
    scored_candidates = []
    for c in candidates:
        if len(c) != 12 or sum(1 for x in c if x != 0) != 12:
            continue
        if any(x < -64 or x > 64 for x in c):
            continue
        ac_zeros = autocorrelation_zeros(c)
        # Also compute actual score for top candidates
        scored_candidates.append((ac_zeros, c))
    
    # Sort by autocorrelation zeros (descending) - more zeros = better potential
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Refine top candidates with targeted local search
    def local_search_fft(start, max_iters=500):
        """Local search that uses autocorrelation to guide moves."""
        curr = list(start)
        curr_score = evaluate(curr)
        curr_ac = autocorrelation_zeros(curr)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try changing one coefficient
            pos = random.randint(0, 11)
            old_val = curr[pos]
            
            # Generate candidate values based on other coefficients
            candidate_vals = set()
            for other in range(12):
                if other != pos:
                    v = curr[other]
                    candidate_vals.update([v, -v, 2*v, -2*v, 3*v, -3*v])
            
            # Add random values
            for _ in range(10):
                candidate_vals.add(random.randint(-8, 8))
            
            # Filter
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]
            
            if candidate_vals:
                # Pick value that maximizes autocorrelation zeros
                best_val = old_val
                best_ac = curr_ac
                for v in candidate_vals[:20]:  # Limit evaluations
                    curr[pos] = v
                    ac = autocorrelation_zeros(curr)
                    if ac > best_ac:
                        best_ac = ac
                        best_val = v
                    curr[pos] = old_val
                
                if best_val != old_val:
                    curr[pos] = best_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        curr_ac = best_ac
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val
            
            # Try swapping
            if random.random() < 0.3:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[i], curr[j] = a, b
            
            # Try negating
            if random.random() < 0.2:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                curr[pos] = -old_val
                if curr[pos] != 0:
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val
            
            if not improved and random.random() < 0.05:
                # Perturbation
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-6, 6)
                    if curr[p] == 0:
                        curr[p] = 1
        
        return curr, curr_score
    
    # Refine top 50 candidates
    top_n = min(50, len(scored_candidates))
    for ac_zeros, cand in scored_candidates[:top_n]:
        if time.time() > deadline:
            break
        result, score = local_search_fft(cand, max_iters=300)
        if score < best_score:
            best_score = score
            best_poly = result
    
    # Phase 3: Final refinement of best
    if time.time() < deadline:
        # Exhaustive single-coordinate refinement
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-16, 17):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val
        
        # Second pass with wider range
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-32, 33):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val
    
    return best_poly
