"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

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
    fft_len = _next_pow2(2 * sum_len)
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Threshold to handle floating point errors
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using FFT (autocorrelation approach)
    # Shift U by max_u so all indices are non-negative for autocorrelation
    diff_vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    freq_d = fft(diff_vec)
    # Autocorrelation gives counts of differences
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    # The differences range from -max_u to max_u, so we look at the central part
    # The result is centered at max_u in the shifted domain
    # We want to count non-zero entries in the difference set
    # The difference set D = {a-b | a,b in U}.
    # The autocorrelation R[k] = sum x[i] x[i-k].
    # Here x is the shifted indicator.
    # The support of the difference set corresponds to the non-zero values of R.
    # Since we shifted by max_u, the differences are centered around max_u in the index space?
    # Actually, standard autocorrelation of x gives R[k] = sum x[i+k]x[i].
    # If x[i] is 1 for i in U+max_u, then R[k] counts pairs (u1+max_u, u2+max_u) such that (u1+max_u) - (u2+max_u) = k.
    # So R[k] is non-zero iff k is a difference in U-U.
    # The range of k is -max_u to max_u.
    # In the FFT output, negative indices wrap around.
    # So we need to check indices 0..max_u and fft_len-max_u..fft_len-1.
    
    # Simpler: just count non-zeros in the whole FFT output that correspond to valid differences?
    # No, the FFT output length is fft_len. The valid differences are in range [-max_u, max_u].
    # In the wrapped array, these are indices 0..max_u and fft_len-max_u..fft_len-1.
    
    # Let's just count non-zeros in the relevant slice.
    # Note: R[0] is always n (pairs u-u=0).
    
    # The difference set size is the number of distinct values in {a-b}.
    # This is exactly the number of non-zero entries in the autocorrelation function.
    
    # We need to be careful with floating point precision.
    # Threshold: > 1e-9
    
    # Indices for positive differences: 0 to max_u
    # Indices for negative differences: fft_len - max_u to fft_len - 1
    # But wait, the autocorrelation is symmetric. R[k] = R[-k].
    # So we can just count non-zeros in 0..max_u and multiply by 2, subtract 1 for k=0?
    # No, because R[k] might be 0 for some k in 0..max_u.
    
    # Let's just sum the non-zeros in the range [0, max_u] and [fft_len-max_u, fft_len).
    # Actually, the range of differences is [-max_u, max_u].
    # The FFT result is periodic.
    # Let's extract the central part.
    # Center is at index 0 (for k=0).
    # Positive k: 1..max_u
    # Negative k: fft_len-1 .. fft_len-max_u
    
    diffset_size = 0
    # k=0
    if abs(diff_result[0]) > 1e-9:
        diffset_size += 1
    
    for k in range(1, max_u + 1):
        # Check k and -k
        # diff_result[k] corresponds to shift k
        # diff_result[fft_len-k] corresponds to shift -k
        if abs(diff_result[k]) > 1e-9:
            diffset_size += 1
        if abs(diff_result[fft_len-k]) > 1e-9:
            diffset_size += 1
    
    # It's possible that due to symmetry, we are double counting if we are not careful?
    # No, k and -k are distinct differences unless k=0.
    # The set U-U contains both d and -d.
    # So counting non-zero R[k] for k in [-max_u, max_u] is correct.
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_exact(U):
    """Exact O(n^2) score computation for small sets (high precision)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
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
    
    if n <= 30:
        return _score_set_exact(U)
    else:
        return _score_set_fft(U)

def _golomb_ruler_known(n):
    """Return known optimal Golomb rulers for small n."""
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
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131],
    }
    if n in known:
        return known[n]
    return None

def _greedy_golomb(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = _golomb_ruler_known(n)
    if known:
        return known
    
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

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (move, swap, insert, delete)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 100
    
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        n = len(U)
        
        # 1. Try moving an element
        indices_to_try = list(range(1, n))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            # Try nearby values
            for offset in range(-10, 11):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        # 2. Try swapping two elements
        if not improved and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_U = U[:]
            new_U[i], new_U[j] = new_U[j], new_U[i]
            if len(set(new_U)) == len(new_U):
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve_count = 0
        
        # 3. Try inserting a new element
        if not improved and n < 100:
            nv = random.randint(1, upper_bound)
            if nv not in U:
                new_U = U + [nv]
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve_count = 0
        
        # 4. Try deleting an element (if set is too large or hurts score)
        if not improved and n > 3:
            i = random.randint(1, n - 1)
            new_U = U[:i] + U[i+1:]
            s = _score_set(new_U)
            if s > best_score + 1e-9:
                best_score = s
                best_U = new_U[:]
                U = new_U[:]
                improved = True
                no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
            # Random restart / kick
            if random.random() < 0.1:
                new_U = U[:]
                idx = random.randint(1, len(new_U) - 1)
                new_U[idx] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score - 0.1: # Allow slight degradation to escape local optimum
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0

    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    candidates = []
    
    # 1. Known Golomb Rulers
    for n in range(3, 21):
        gr = _golomb_ruler_known(n)
        if gr:
            candidates.append(gr)
    
    # 2. Greedy Golomb Rulers for larger n
    for n in range(20, 50, 5):
        if time.time() > deadline - 60:
            break
        gr = _greedy_golomb(n)
        candidates.append(gr)
        
    # 3. Random sets with sparse structure
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        n = random.randint(3, 40)
        max_val = random.randint(50, 2000)
        # Bias towards small numbers?
        # Use random.sample to ensure distinctness
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
        
    # 4. Structured sets: Powers
    for base in [2, 3, 4]:
        U = [0]
        for i in range(1, 15):
            val = base ** i
            if val > 5000:
                break
            U.append(val)
        candidates.append(U)
        
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on top candidates
    for s, U in scored_candidates[:15]:
        if time.time() > deadline - 10:
            break
        if U == best_U:
            continue
        # Run local search with remaining time
        improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline - 5)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
            
    # Final polish on the best candidate
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=deadline - 1)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
