"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft


def _score_set(U):
    """Compute the score for a finite set U containing 0, using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # For very small sets, direct computation is faster
    if n <= 8:
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

    # FFT-based computation for larger sets
    # Sumset: U + U, values in [0, 2*max_u]
    sum_range = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_range:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_range] > 1e-9))

    # Difference set: U - U, values in [-max_u, max_u]
    # Use autocorrelation: place U at offset max_u, convolve with reversed
    diff_range = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_range:
        fft_len2 *= 2

    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0

    # Autocorrelation via FFT
    freq_d = fft(diff_vec)
    autocorr = ifft(freq_d * np.conj(freq_d)).real

    # The autocorrelation at lag k gives count of pairs with difference k
    # diff_vec has U at positions [max_u, max_u+u] for u in U
    # autocorr[k] = sum_i diff_vec[i] * diff_vec[i-k] (with wraparound handled by padding)
    # We need to count positions where autocorr > 0 in the valid range
    # Valid lags: -max_u to max_u, which in the autocorrelation array are at positions
    # fft_len2 - max_u to fft_len2 + max_u (wrapping)
    # Actually, for a zero-padded signal, autocorr[0] is at index 0,
    # positive lags at 1..max_u, negative lags at fft_len2-max_u..fft_len2-1

    # Count nonzero autocorrelation entries in valid range
    valid_mask = np.zeros(fft_len2, dtype=bool)
    valid_mask[0] = True
    valid_mask[1:max_u+1] = True
    valid_mask[fft_len2-max_u:fft_len2] = True
    diffset_size = int(np.sum(autocorr[valid_mask] > 1e-9))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom


def _golomb_ruler(n):
    """Generate a Golomb ruler of order n."""
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
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }
    if n in known:
        return known[n]
    # Greedy construction
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


def _mian_chowla(n):
    """Generate Mian-Chowla sequence (Sidon sequence) of length n."""
    if n <= 1:
        return [0]
    seq = [0, 1]
    while len(seq) < n:
        # Find smallest candidate that doesn't create duplicate sums
        cand = seq[-1] + 1
        while True:
            valid = True
            sums = set()
            for i in range(len(seq) + 1):
                for j in range(i, len(seq) + 1):
                    if i == len(seq):
                        a = cand
                    else:
                        a = seq[i]
                    if j == len(seq):
                        b = cand
                    else:
                        b = seq[j]
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                seq.append(cand)
                break
            cand += 1
    return seq


def _score_arithmetic_progression(n, d):
    """
    Closed-form score for arithmetic progression {0, d, 2d, ..., (n-1)d}.
    
    By Cauchy-Davenport (and exact computation):
      |U+U| = 2n - 1  (sums range from 0 to 2(n-1)d in steps of d)
      |U-U| = 2n - 1  (diffs range from -(n-1)d to (n-1)d in steps of d)
      max(U) = (n-1)*d
    
    Score = 1 + log(1) / log(2*(n-1)*d + 1) = 1 + 0 = 1.
    
    Wait, that gives exactly 1.0 for any AP. So APs are not optimal.
    The key insight: we need |U-U|/|U+U| > 1 to get score > 1.
    For APs, ratio = 1, so score = 1. We need non-AP structures.
    
    But this tells us the search space: we want sets where differences
    spread out more than sums. Golomb rulers and Sidon sequences achieve this.
    """
    if n <= 1:
        return 1.0
    if d <= 0:
        return 1.0
    # For AP: |U+U| = 2n-1, |U-U| = 2n-1, ratio = 1
    # Score = 1 + log(1)/log(2*(n-1)*d+1) = 1.0
    return 1.0


def _score_golomb_like(n, base_gap):
    """
    Estimate score for a set with approximately linearly growing gaps.
    For a set {0, a1, a2, ..., a_{n-1}} with ai ≈ i*base_gap:
      |U-U| ≈ n^2 (all pairwise differences distinct, Golomb property)
      |U+U| ≈ n^2/2 (sums have some collisions)
      max(U) ≈ (n-1)*base_gap
    
    This is an approximation; use _score_set for exact value.
    """
    if n <= 1:
        return 1.0
    # Rough estimate
    diff_est = n * (n - 1) / 2 + 1  # ~n^2/2 distinct diffs
    sum_est = n * (n + 1) / 2  # ~n^2/2 distinct sums (with 0)
    max_est = (n - 1) * base_gap
    if max_est <= 0:
        return 1.0
    ratio = diff_est / sum_est
    if ratio <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_est + 1)


def _local_search(U, max_iters=3000, deadline=None):
    """Improve set U via local search with FFT scoring."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # Try moving one element (not 0)
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try nearby values and some random ones
            candidates = set()
            for offset in range(-30, 31, 1):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
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
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 200
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try adding an element
            if len(U) < 100:
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
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            # Try removing an element (if size > 3)
            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

        if not improved:
            break

    return sorted(set(best_U)), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # Key insight from Cauchy-Davenport: APs give score exactly 1.0.
    # We need structures where |U-U| > |U+U|.
    # Golomb rulers maximize distinct differences.
    # Sidon/Mian-Chowla sequences control sum collisions.
    
    # Golomb rulers of various sizes - these have all pairwise diffs distinct
    for n in range(3, 20):
        if time.time() > deadline:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)

    # Mian-Chowla sequences - these minimize sum collisions
    for n in range(3, 20):
        if time.time() > deadline:
            break
        mc = _mian_chowla(n)
        candidates.append(mc)

    # Structured sparse sets: superincreasing-like spacing
    # Key: make gaps grow fast enough that most differences are unique
    random.seed(42)
    for trial in range(40):
        if time.time() > deadline:
            break
        n = random.randint(5, 25)
        # Start with 0, then add elements with rapidly increasing gaps
        U = [0]
        current = 1
        for i in range(n - 1):
            # Gaps grow superlinearly to ensure most diffs are unique
            gap = random.randint(2, 4) * (i + 1) ** random.randint(1, 2)
            current += gap
            U.append(current)
        candidates.append(sorted(U))

    # Random sparse sets with controlled density
    for trial in range(40):
        if time.time() > deadline:
            break
        n = random.randint(5, 30)
        # Choose max_val to balance |U-U|/|U+U| vs log(2*max+1)
        # We want ratio > 1, so need enough spread
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        candidates.append(U)

    # Hybrid: start from Golomb ruler, perturb slightly
    for n in [8, 10, 12, 15]:
        if time.time() > deadline:
            break
        gr = _golomb_ruler(n)
        # Try a few perturbations
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = gr[:]
            idx = random.randint(1, len(perturbed) - 1)
            offset = random.randint(-3, 3)
            new_val = perturbed[idx] + offset
            if new_val > 0 and new_val not in perturbed:
                perturbed[idx] = new_val
                perturbed.sort()
                candidates.append(perturbed)

    # Evaluate all candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    scored_candidates.sort(key=lambda x: -x[0])

    # Local search on best candidate
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=deadline)

    # Try local search on top several good candidates for diversity
    for s, U in scored_candidates[:8]:
        if time.time() > deadline:
            break
        if sorted(set(U)) == best_U:
            continue
        if s > best_score - 0.1:  # allow slightly worse starting points
            improved_U, improved_score = _local_search(U, max_iters=1500, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
