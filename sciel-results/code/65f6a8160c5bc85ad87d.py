import math
import time
import random
import numpy as np
from scipy.fft import next_fast_len

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
    FFT-based score calculation.
    U is a list of nonnegative integers.
    Returns the score 1 + log2(|U-U|/|U+U|) / log2(2*max(U)+1).
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # For very small sets, exact is faster
    if len(U) <= 30:
        return _score_small(U)

    # Sumset: U + U
    # We need to count the number of distinct values in {a+b | a,b in U}
    # This is equivalent to the support of the convolution of the indicator function of U with itself.
    # Let f be the indicator of U. Then (f * f)[k] = sum_{i+j=k} f[i]f[j].
    # We want the number of k such that (f*f)[k] > 0.
    
    # Diffset: U - U
    # This is {a-b | a,b in U}.
    # Note that U-U is symmetric around 0.
    # The range of U-U is [-M, M].
    # We can compute this by shifting U by M so all values are non-negative,
    # but convolution is for sums.
    # Alternatively, |U-U| is the number of distinct differences.
    # For a set U, the number of distinct differences can be found by:
    # Let g be the indicator of U.
    # The number of representations of d as a-b is sum_{a} g[a]g[a-d].
    # This is the cross-correlation of g with itself.
    # Cross-correlation can be computed via FFT.
    # (g * g_rev)[k] where g_rev[i] = g[-i] (shifted).
    
    # Let's use a unified approach with FFT.
    # Size of array:
    # Sumset range: [0, 2M]. Size 2M+1.
    # Diffset range: [-M, M]. Size 2M+1.
    
    N = next_fast_len(2 * M + 1)
    
    # Indicator vector
    f = np.zeros(N, dtype=np.float64)
    for u in U:
        f[u] = 1.0
    
    # Sumset
    # Convolution f * f
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, N)
    # conv[k] is the number of pairs (a,b) such that a+b=k.
    # We count k where conv[k] > 0.
    # Due to floating point, we check > 0.5 or similar.
    sumset_size = np.sum(conv > 0.5)
    
    # Diffset
    # We need cross-correlation.
    # corr[d] = sum_a f[a] f[a-d].
    # This is equivalent to convolution of f with reversed f.
    # Let g be f reversed and shifted to be non-negative index.
    # Actually, simpler:
    # The number of distinct differences is the support size of the autocorrelation.
    # Autocorrelation R[k] = sum_n f[n] f[n+k].
    # We can compute this using FFT as well.
    # R = ifft(fft(f) * conj(fft(f)))
    # Note: R is real.
    # The indices of R correspond to differences.
    # Specifically, R[k] for k >= 0 gives sum f[n]f[n+k].
    # R[N-k] gives sum f[n]f[n-k].
    # So the distinct differences are the k where R[k] > 0 or R[N-k] > 0.
    # Actually, since U is a set of non-negative integers, f[n] is 0 for n > M.
    # The differences range from -M to M.
    # The autocorrelation computed by FFT has length N.
    # R[0] is sum f[n]^2 = |U|.
    # R[k] for k=1..M is sum f[n]f[n+k] (positive differences).
    # R[N-k] for k=1..M is sum f[n]f[n-k] (negative differences).
    # Since U is a set, f[n] is 0/1.
    # The number of distinct positive differences is count of k in 1..M where R[k] > 0.5.
    # The number of distinct negative differences is count of k in 1..M where R[N-k] > 0.5.
    # Total distinct differences = 1 (for 0) + 2 * (count of positive diffs).
    # Wait, is it always symmetric? Yes, if d is in U-U, then -d is in U-U.
    # So |U-U| = 1 + 2 * |{d > 0 : d in U-U}|.
    
    R = np.fft.irfft(F * np.conj(F), N)
    
    # Count positive differences
    # R[k] for k=1 to M
    pos_diffs = np.sum(R[1:M+1] > 0.5)
    diffset_size = 1 + 2 * pos_diffs
    
    # Calculate score
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio, 2) / denom

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
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _gen_minkowski(U1, U2):
    """Generate Minkowski sum U1 + U2."""
    res = set()
    for a in U1:
        for b in U2:
            res.add(a + b)
    return sorted(list(res))

def _local_search(U, time_budget=10.0):
    """
    Fast local search: try moving each element by small integers.
    Prioritizes reducing max(U) and maintaining Golomb property.
    Uses FFT score for larger sets.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _fft_score(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    # Determine mutation range based on set size
    # Smaller sets can have larger shifts
    if n < 10:
        max_shift = 10
    elif n < 20:
        max_shift = 5
    else:
        max_shift = 3

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
            # Generate random shifts within max_shift
            shifts = list(range(-max_shift, max_shift + 1))
            rng.shuffle(shifts)
            for delta in shifts:
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                # Check Golomb property only for small sets to save time?
                # Actually, Golomb property is not required for the score, just helpful.
                # The score depends on |U-U| and |U+U|.
                # Golomb rulers maximize |U-U| for a given span.
                # So maintaining Golomb is a good heuristic.
                # But for larger sets, it's hard to maintain.
                # Let's just evaluate the score.
                
                s = _fft_score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            # Random restart/perturbation if stuck
            if rng.random() < 0.3:
                # Try adding a new element
                if n < 50:
                    # Pick a random location to insert
                    # Or just shift a random element by a larger amount
                    idx = rng.randint(1, n-1)
                    # Large jump
                    jump = rng.randint(max_shift + 1, max_shift * 2)
                    if rng.random() < 0.5:
                        new_val = best_U[idx] + jump
                    else:
                        new_val = max(1, best_U[idx] - jump)
                    
                    if new_val not in best_U and new_val >= 0:
                        new_U = list(best_U)
                        new_U[idx] = new_val
                        new_U = sorted(new_U)
                        s = _fft_score(new_U)
                        if s > best_s:
                            best_s = s
                            best_U = new_U
                else:
                    # If set is large, try removing an element?
                    # Or just shifting.
                    idx = rng.randint(1, n-1)
                    new_val = best_U[idx] + rng.randint(-max_shift, max_shift)
                    if new_val not in best_U and new_val >= 0:
                        new_U = list(best_U)
                        new_U[idx] = new_val
                        new_U = sorted(new_U)
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
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Generate larger candidates using Minkowski sums of small Golomb rulers
    # e.g., U = G1 + G2
    # This creates sets with structure that might be good.
    g4 = _gen_golomb_optimal(4)
    g5 = _gen_golomb_optimal(5)
    g6 = _gen_golomb_optimal(6)
    
    if g4 and g5:
        candidates.append(_gen_minkowski(g4, g5))
    if g4 and g6:
        candidates.append(_gen_minkowski(g4, g6))
    if g5 and g5:
        candidates.append(_gen_minkowski(g5, g5))
        
    # 3. Add a few random variations of small Golomb rulers
    for n in range(3, 10):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(5):
                perturbed = list(base)
                # Randomly swap two non-zero elements or shift one
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    if _is_golomb(perturbed):
                        candidates.append(perturbed)
                        
    # 4. Randomly generated sets with small max element
    # Try to find sets with high |U-U|/|U+U| ratio
    for _ in range(10):
        # Pick a random size
        size = random.randint(5, 15)
        # Pick a random max element
        max_val = random.randint(size, size * 5)
        # Generate random set
        U = {0}
        for _ in range(size - 1):
            U.add(random.randint(1, max_val))
        U = sorted(list(U))
        candidates.append(U)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_fft_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    # We have ~20-30 candidates. Allocate time carefully.
    top_candidates = [s for _, s in scored[:15]]
    
    # Calculate time per candidate
    # We want to finish in ~100s. 
    # 15 candidates * 3s each = 45s. Safe.
    per_set_time = 3.0
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(15.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
