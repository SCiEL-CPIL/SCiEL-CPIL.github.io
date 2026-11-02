"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    sum_len = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    sumset_size = np.sum(conv[:sum_len] > 1e-9)
    
    # Compute difference set using FFT properly
    # U - U: values range from -max_u to max_u
    # Place U shifted by max_u, then autocorrelate
    diff_vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    # Autocorrelation: convolve with reversed version
    diff_vec_rev = np.zeros(fft_len, dtype=np.float64)
    for i in range(fft_len):
        diff_vec_rev[i] = diff_vec[fft_len - 1 - i]
    
    freq_d = fft(diff_vec)
    freq_dr = fft(diff_vec_rev)
    diff_result = ifft(freq_d * freq_dr).real
    
    # Differences d appear at position max_u + d in the result
    # So we look at positions 0 to 2*max_u
    diffset_size = np.sum(np.abs(diff_result[:2*max_u+1]) > 1e-9)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For small sets, use direct computation
    if n <= 40:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _fast_score_approx(U):
    """Fast approximate score for local search guidance."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use sampling for fast approximation
    sample_size = min(n, 20)
    indices = random.sample(range(n), sample_size)
    sampled = [U[i] for i in indices]
    
    sumset_sample = set()
    diffset_sample = set()
    for a in sampled:
        for b in sampled:
            sumset_sample.add(a + b)
            diffset_sample.add(a - b)
    
    # Scale up estimate
    scale_factor = (n * n) / (sample_size * sample_size)
    est_sumset = min(len(sumset_sample) * scale_factor, 2 * max_u + 1)
    est_diffset = min(len(diffset_sample) * scale_factor, 2 * max_u + 1)
    
    if est_sumset <= 0:
        return 0.0
    ratio = est_diffset / est_sumset
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _modular_lattice_set(m, density, max_val):
    """Create a set using modular lattice structure."""
    # Select residues that form a dense sublattice in Z_m
    # Use residues that are multiples of a step, then spread them
    step = max(1, m // (int(density * m) + 1))
    residues = set(range(0, m, step))
    
    # Map residues to actual values by spreading them
    U = set([0])
    for r in residues:
        # Place multiple copies at different scales
        for scale in [1, 2, 4]:
            val = r * scale
            if val <= max_val and val != 0:
                U.add(val)
    
    return sorted(U)

def _local_search(U, max_iters=2000, deadline=None, use_fast_score=True):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200  # search space for swaps
    
    score_func = _fast_score_approx if use_fast_score else _score_set
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try swapping one element
        for i in range(len(U)):
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Try several new values around old_val and random positions
            candidates = set()
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add some random candidates
            for _ in range(15):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = score_func(new_U)
                if s > best_score + 1e-9:
                    # Verify with exact score before accepting
                    exact_s = _score_set(new_U)
                    if exact_s > best_score + 1e-9:
                        best_score = exact_s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 300:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = score_func(new_U)
                    if s > best_score + 1e-9:
                        exact_s = _score_set(new_U)
                        if exact_s > best_score + 1e-9:
                            best_score = exact_s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
            
            if not improved:
                # Try removing an element (if size > 2)
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = score_func(new_U)
                        if s > best_score + 1e-9:
                            exact_s = _score_set(new_U)
                            if exact_s > best_score + 1e-9:
                                best_score = exact_s
                                best_U = new_U[:]
                                U = new_U[:]
                                improved = True
                                break
        
        if not improved:
            # Try a larger random perturbation
            if random.random() < 0.15:
                new_U = U[:]
                # Randomly change a few elements
                for _ in range(min(8, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)  # skip index 0 (value 0)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = score_func(new_U)
                    if s > best_score + 1e-9:
                        exact_s = _score_set(new_U)
                        if exact_s > best_score + 1e-9:
                            best_score = exact_s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
            
            if not improved:
                # Try modular lattice restart
                if random.random() < 0.05:
                    m = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
                    density = random.uniform(0.3, 0.7)
                    max_val = max_u + 100
                    new_U = _modular_lattice_set(m, density, max_val)
                    if new_U and 0 in new_U:
                        s = score_func(new_U)
                        if s > best_score + 1e-9:
                            exact_s = _score_set(new_U)
                            if exact_s > best_score + 1e-9:
                                best_score = exact_s
                                best_U = new_U[:]
                                U = new_U[:]
                                improved = True
            
            if not improved:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 35):
        if time.time() > deadline - 40:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(40):
        if time.time() > deadline - 40:
            break
        n = random.randint(3, 80)
        max_val = random.randint(20, 1000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U[:min(len(U), 30)])
    
    # Powers-based sets
    for p in [2, 3, 5]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 20):
            U.append(p ** i)
            if U[-1] > 50000:
                break
        candidates.append(U)
    
    # Modular lattice sets
    for m in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
        if time.time() > deadline - 40:
            break
        for density in [0.3, 0.5, 0.7]:
            U = _modular_lattice_set(m, density, 5000)
            if U and 0 in U:
                candidates.append(U)
    
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search on best candidate
    if time.time() < deadline - 10:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=deadline - 10, use_fast_score=True)
    
    # Try local search on a few other candidates
    for U in candidates[:12]:
        if time.time() > deadline - 10:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.05:  # only if close
            improved_U, improved_score = _local_search(U, max_iters=3000, deadline=deadline - 10, use_fast_score=True)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish with exact scoring
    if time.time() < deadline - 5:
        final_U, final_score = _local_search(best_U, max_iters=2000, deadline=deadline - 5, use_fast_score=False)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
