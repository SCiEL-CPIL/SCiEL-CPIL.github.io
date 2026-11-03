import time
import random
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - time.time()

def _score_set_fft(U):
    """
    Compute the score for a finite set U containing 0 using FFT.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # Size of FFT needed: max sum is 2*max_u, max diff is max_u - 0 = max_u (but range is -max_u to max_u)
    # For sumset: indices 0 to 2*max_u
    # For diffset: indices -max_u to max_u, which we can map to 0 to 2*max_u by shifting
    # We can compute both using convolutions.
    
    # Create indicator vector
    size = 2 * max_u + 1
    A = np.zeros(size, dtype=np.float64)
    for u in U:
        A[u] = 1.0

    # Sumset: Convolution of A with A
    # The result at index k corresponds to the number of pairs (a,b) such that a+b=k
    # We need the count of k where count > 0
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    A_fft = np.fft.rfft(A, n=fft_size)
    sum_fft = A_fft * A_fft
    sum_conv = np.fft.irfft(sum_fft, n=fft_size)[:2 * max_u + 1]
    
    # Due to floating point errors, values close to 0 should be treated as 0
    # Values should be integers (counts)
    sum_counts = np.round(sum_conv).astype(np.int32)
    sumset_size = np.sum(sum_counts > 0)

    # Diffset: Cross-correlation of A with A
    # We want the number of pairs (a,b) such that a-b = k
    # This is equivalent to convolution of A with reverse(A)
    # Let B be A reversed. Then (A * B)[i] = sum A[j] B[i-j] = sum A[j] A[max_u - (i-j)]
    # This is tricky with boundaries. 
    # Alternative: Compute D[k] = sum_{a,b in U} 1_{a-b=k}
    # This is the autocorrelation of A.
    # Autocorrelation can be computed via FFT: IFFT(|FFT(A)|^2)
    # But we need the full range -max_u to max_u.
    # Let's use the property that diffset size is related to the number of non-zero entries in the autocorrelation.
    
    # Actually, for difference set, we can just use the direct method if n is small, 
    # but FFT is better for larger n.
    # Let's stick to FFT for consistency.
    # The autocorrelation R[k] = sum_x A[x] A[x-k]
    # We want the number of k such that R[k] > 0.
    # The range of k is -max_u to max_u.
    
    # To compute autocorrelation via FFT:
    # 1. Zero pad A to length >= 2*max_u + 1
    # 2. FFT
    # 3. Multiply by conjugate
    # 4. IFFT
    # The result will have the autocorrelation values, but shifted.
    
    # Let's use a simpler approach for diffset if n is not too large, 
    # but since we are optimizing for speed, let's try FFT for both.
    
    # Autocorrelation via FFT
    A_fft_full = np.fft.rfft(A, n=fft_size)
    corr_fft = A_fft_full * np.conj(A_fft_full)
    corr = np.fft.irfft(corr_fft, n=fft_size)
    
    # The correlation values are centered at index 0? No, circular correlation.
    # We need to handle the wrap-around.
    # The linear correlation of two sequences of length L is obtained by:
    # 1. Pad to length >= 2L-1
    # 2. FFT
    # 3. Multiply
    # 4. IFFT
    # 5. Take the first L elements and the last L-1 elements?
    
    # Let's just use the direct set method for diffset if n < 100, 
    # otherwise use a smarter FFT approach or just accept the cost.
    # Actually, for n ~ 50, n^2 = 2500, which is very fast in Python.
    # The bottleneck is the number of evaluations, not the single evaluation cost for small n.
    # For n ~ 100, n^2 = 10000, still fast.
    # So let's use direct set for diffset and FFT for sumset? Or just direct for both if n is small?
    # Let's benchmark mentally: 10000 operations in Python is ~1ms.
    # FFT of size 4096 is ~0.1ms.
    # So FFT is faster for larger n.
    
    # Let's implement a robust FFT-based diffset size calculation.
    # The number of distinct differences is the number of non-zero entries in the autocorrelation sequence.
    
    # Shift the correlation to center it
    # The IFFT result has the correlation values at indices 0, 1, ..., fft_size-1
    # The correlation at lag k is at index k (for k >= 0) and index fft_size - k (for k < 0)
    
    diff_counts = np.zeros(2 * max_u + 1, dtype=np.int32)
    for k in range(-max_u, max_u + 1):
        if k >= 0:
            val = corr[k]
        else:
            val = corr[fft_size + k]
        diff_counts[k + max_u] = int(round(val))
        
    diffset_size = np.sum(diff_counts > 0)

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_direct(U):
    """Compute score using direct set operations. Used for small sets or verification."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

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

def _score_set(U):
    """Choose the best scoring method based on set size."""
    n = len(U)
    if n < 30:
        return _score_set_direct(U)
    else:
        return _score_set_fft(U)

# Known Golomb rulers (shortest length for n marks)
_GOLOMB_RULERS = {
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
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94],
}

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 100
    no_improve_count = 0
    max_no_improve = 50

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_left() < 5:
            break
            
        improved = False

        # 1. Try moving one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            
            # Nearby values
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            
            # Random jumps
            for _ in range(20):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 100
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break

        # 2. Try adding an element
        if not improved and len(U) < 50:
            for _ in range(30):
                new_val = random.randint(1, upper_bound)
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 100
                    improved = True
                    no_improve_count = 0
                    break

        # 3. Try removing an element
        if not improved and len(U) > 3:
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                new_U = U[:i] + U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 100
                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            # Random perturbation to escape local optima
            if random.random() < 0.2:
                new_U = U[:]
                for _ in range(min(3, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 100
                        improved = True
                        no_improve_count = 0

            if no_improve_count > max_no_improve:
                break

    return sorted(set(best_U)), best_score

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140

    random.seed(12345)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # Add known Golomb rulers
    for n, gr in _GOLOMB_RULERS.items():
        candidates.append(gr)
        
    # Add some random sets
    for _ in range(50):
        if _time_left() < 20:
            break
        n = random.randint(5, 25)
        max_val = random.randint(50, 500)
        U = [0] + random.sample(range(1, max_val + 1), min(n - 1, max_val))
        U = sorted(U)
        candidates.append(U)
        
    # Add some structured sets (e.g., powers of 2, primes)
    for _ in range(10):
        if _time_left() < 20:
            break
        base = random.randint(2, 5)
        n = random.randint(5, 15)
        U = [0] + [base ** i for i in range(1, n)]
        if max(U) < 1000:
            candidates.append(U)

    # Evaluate candidates
    for U in candidates:
        if _time_left() < 15:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Local search on the best candidate
    if _time_left() > 10:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=_DEADLINE - 5)

    # Local search on top 5 candidates
    scored = []
    for U in candidates:
        if _time_left() < 10:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        scored.append((s, U))

    scored.sort(key=lambda x: -x[0])
    for s, U in scored[:5]:
        if _time_left() < 8:
            break
        if s < best_score - 0.05:
            break
        improved_U, improved_score = _local_search(U, max_iters=5000, deadline=_DEADLINE - 5)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U

    # Final polish
    if _time_left() > 5:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=_DEADLINE - 2)

    return sorted(set(best_U))
