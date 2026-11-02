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
    # U + U: values range from 0 to 2*max_u
    sum_len = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT to compute convolution (sumset)
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Count nonzero entries in sumset (values 0 to 2*max_u)
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using autocorrelation via FFT
    # |U-U| = number of distinct values d = a-b for a,b in U
    # Autocorrelation: r[k] = sum_i indicator[i] * indicator[i-k]
    # Using FFT: r = ifft(fft(indicator) * conj(fft(indicator)))
    # r[k] counts pairs (a,b) with a - b = k (for k >= 0 in our indexing)
    # Actually: conv = ifft(freq * conj(freq)).real gives autocorrelation
    # r[d] = number of pairs (a,b) in U x U with a - b = d
    # d ranges from -max_u to max_u
    # In the circular convolution of length fft_len, r[d] for d in [0, fft_len-1]
    # For negative differences, they wrap around: r[fft_len - |d|]
    
    auto = ifft(freq * np.conj(freq)).real
    # auto[k] for k=0..fft_len-1
    # auto[0] = n (pairs with diff 0)
    # auto[k] for k > 0: pairs with a - b = k OR a - b = -(fft_len - k)
    # Since max_u < fft_len/2, differences in [-max_u, max_u] are:
    #   d >= 0: auto[d]
    #   d < 0: auto[fft_len + d]
    
    # Count distinct differences: for each d in [-max_u, max_u], check if auto[d mod fft_len] > 0
    # d=0: auto[0]
    # d=1..max_u: auto[d]
    # d=-1..-max_u: auto[fft_len+d]
    
    # Vectorized: check indices 0, 1, ..., max_u and fft_len-max_u, ..., fft_len-1
    pos_indices = np.arange(0, max_u + 1)
    neg_indices = np.arange(fft_len - max_u, fft_len)
    all_indices = np.concatenate([pos_indices, neg_indices])
    diffset_size = int(np.sum(np.abs(auto[all_indices]) > 1e-9))
    
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

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200  # search space for swaps
    stall_count = 0
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try swapping one element
        for i in range(len(U)):
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Try several new values: local offsets + wider random
            candidates = set()
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add random candidates at various scales
            for _ in range(15):
                scale = random.choice([1, 5, 20, 100])
                candidates.add(random.randint(1, upper_bound + scale))
            # Try replacing with values from a geometric pattern
            for _ in range(5):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    stall_count = 0
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
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        stall_count = 0
                        break
            
            if not improved:
                # Try removing an element (if size > 2)
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            stall_count = 0
                            break
        
        if not improved:
            stall_count += 1
            # Try a larger random perturbation when stalled
            if random.random() < 0.3 or stall_count > 50:
                new_U = U[:]
                # Randomly change a few elements
                for _ in range(min(5, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        stall_count = 0
            
            if not improved and stall_count > 100:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline - 40:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(40):
        if time.time() > deadline - 40:
            break
        n = random.randint(3, 60)
        max_val = random.randint(20, 1000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:min(len(U), 25)])
    
    # Powers-based sets (sparse geometric - good for large diffset)
    for p in [2, 3, 4]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 20):
            val = p ** i
            if val > 50000:
                break
            U.append(val)
        candidates.append(U)
    
    # Mixed-scale sets: combine different magnitudes
    if time.time() < deadline - 40:
        for trial in range(10):
            U = [0]
            # Small cluster
            for i in range(random.randint(2, 5)):
                U.append(random.randint(1, 10))
            # Medium cluster
            for i in range(random.randint(2, 5)):
                U.append(random.randint(20, 200))
            # Large sparse
            for i in range(random.randint(1, 3)):
                U.append(random.randint(500, 5000))
            U = sorted(set(U))
            candidates.append(U)
    
    # Modular structure: elements at positions that maximize differences
    if time.time() < deadline - 40:
        for mod in [7, 11, 13, 17, 23]:
            U = [0]
            for k in range(1, 15):
                U.append(k * mod + (k % 3))  # slight perturbation
            U = sorted(set(U))
            candidates.append(U)
    
    # Sparse random with geometric spacing
    if time.time() < deadline - 40:
        for trial in range(15):
            U = [0]
            current = 1
            for i in range(random.randint(8, 20)):
                current += random.randint(1, current + 5)
                if current > 10000:
                    break
                U.append(current)
            U = sorted(set(U))
            candidates.append(U)
    
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline - 15:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search on best candidate
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=4000, deadline=deadline - 5)
    
    # Try local search on a few other promising candidates
    for U in candidates[:10]:
        if time.time() > deadline - 5:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.03:  # only if close
            improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline - 5)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish: very tight local search on the best
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=deadline - 1)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
