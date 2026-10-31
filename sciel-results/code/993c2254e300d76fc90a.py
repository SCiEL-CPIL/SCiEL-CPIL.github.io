import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _fft_score(U):
    """
    Fast score calculation using FFT.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    # Sumset range: [0, 2M]. Length 2M+1.
    # Diffset range: [-M, M]. Length 2M+1.
    # To avoid circular convolution, we need L >= 2M+1.
    # Next power of 2 is efficient.
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: A * A
    # The result will be at indices 0..2M.
    # We can use ifft(fft(A) * fft(A))
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    # Count non-zero elements in S[0 : 2*M+1]
    # Due to floating point errors, threshold is needed.
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Autocorrelation of A
    # R[k] = sum_j A[j] A[j+k]
    # This is the inverse FFT of |F_A|^2.
    # The peak is at index 0 (lag 0).
    # Wait, standard autocorrelation via FFT:
    # If we compute ifft(|fft(A)|^2), the result is the circular autocorrelation.
    # For linear autocorrelation, we need zero padding.
    # We already have L >= 2M+1.
    # Let's check the indexing.
    # Let A be length L.
    # C = ifft(|fft(A)|^2).real
    # C[k] corresponds to lag k.
    # Specifically, C[k] = sum_i A[i] A[(i+k) % L].
    # Since A is zero-padded to length L >= 2M+1, and max index is M,
    # for k in [0, M], (i+k) < M+M = 2M < L, so no wrap-around for positive lags.
    # For negative lags, we look at C[L-k].
    # We want the number of distinct differences d = a-b.
    # This is the number of k in [-M, M] such that C[k] > 0 (or C[L-k] > 0 for negative).
    # Note: C[0] is always > 0 (a=a).
    # The difference set U-U includes 0.
    # So we count k in [0, M] where C[k] > 0.1 AND k in [1, M] where C[L-k] > 0.1.
    # Actually, since U contains non-negatives, U-U is symmetric around 0.
    # So |U-U| = 1 + 2 * (number of positive differences).
    # Positive differences are d > 0.
    # C[k] for k > 0 counts pairs (a,b) with a-b = -k? Or a-b=k?
    # Let's verify.
    # C[k] = sum_i A[i] A[i+k].
    # If A[i]=1 and A[i+k]=1, then i and i+k are in U.
    # So i+k - i = k is a difference in U-U.
    # So C[k] > 0 implies k is in U-U.
    # So we just need to count k in [1, M] such that C[k] > 0.1.
    # And add 1 for the zero difference.
    
    F_C = F_A * np.conjugate(F_A)
    C = np.fft.ifft(F_C).real
    
    # Count positive differences
    # k ranges from 1 to M
    pos_diffs = np.sum(C[1:M+1] > 0.1)
    diff_count = 1 + 2 * pos_diffs
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 73, 85],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 73, 85, 99],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 73, 85, 99, 117],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _local_search(U, time_budget=10.0):
    """
    Fast local search: try moving each element by small integers.
    Uses FFT scoring for speed.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _fft_score(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            # Try small shifts
            # Larger shifts might be needed if span is large
            # But for Golomb, small shifts are good.
            for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10, -20, 20]:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                s = _fft_score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        if not improved:
            # Random restart/perturbation if stuck
            if rng.random() < 0.3:
                idx = rng.randint(1, n-1)
                # Larger random jump
                jump = rng.randint(-50, 50)
                new_val = max(1, best_U[idx] + jump)
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = _fft_score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U

    return best_U, best_s

def _greedy_expand(U, time_budget=10.0):
    """
    Try to add new elements to U if it improves the score.
    """
    best_U = sorted(list(U))
    best_s = _fft_score(best_U)
    start = time.time()
    rng = random.Random(123)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        M = max(best_U)
        # Try adding elements in the range [1, M + 100]
        # Sample some candidates
        candidates = list(range(1, min(M + 101, 200)))
        rng.shuffle(candidates)
        
        for c in candidates:
            if c in best_U:
                continue
            new_U = sorted(best_U + [c])
            s = _fft_score(new_U)
            if s > best_s + 1e-10:
                best_s = s
                best_U = new_U
                improved = True
                break
                
        if not improved:
            # Try larger jumps if stuck
            c = M + rng.randint(1, 50)
            if c not in best_U:
                new_U = sorted(best_U + [c])
                s = _fft_score(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers for small n
    for n in range(3, 16):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add a few random variations of small Golomb rulers
    for n in range(3, 12):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(10):
                perturbed = list(base)
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    if _is_golomb(perturbed):
                        candidates.append(perturbed)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates using FFT
    scored = [(_fft_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:15]]
    
    per_set_time = 3.0
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Greedy expansion on the best found
    if _remaining() > 15.0:
        expanded, es = _greedy_expand(best_U, time_budget=min(20.0, _remaining() - 10.0))
        if es > best_s:
            best_s = es
            best_U = expanded
            
    # Final polish on the best found
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(15.0, _remaining() - 5.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
