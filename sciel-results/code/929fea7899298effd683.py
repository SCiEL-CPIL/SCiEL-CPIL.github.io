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
    if n <= 10:
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
    diff_range = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_range:
        fft_len2 *= 2

    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0

    freq_d = fft(diff_vec)
    autocorr = ifft(freq_d * np.conj(freq_d)).real

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

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are distinct)."""
    diffs = set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _compress_golomb(U, max_steps=1000):
    """Try to reduce the span of a Golomb ruler by moving elements left."""
    U = sorted(set(U))
    if not _is_golomb(U):
        return U
    improved = True
    while improved:
        improved = False
        # Try moving each element (except 0) to the left
        for i in range(len(U) - 1, 0, -1):
            original = U[i]
            for new_val in range(original - 1, -1, -1):
                if new_val == 0 and i != 0:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(U):
                    continue
                if _is_golomb(new_U):
                    U = new_U
                    improved = True
                    break
            if improved:
                break
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Known optimal Golomb rulers (span minimized)
    known_rulers = {
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

    # Generate candidates
    candidates = []
    
    # Use known rulers
    for n in range(1, 13):
        if n in known_rulers:
            candidates.append(known_rulers[n][:])

    # Generate greedy rulers for larger n
    for n in range(13, 20):
        if time.time() > deadline:
            break
        U = [0]
        for i in range(1, n):
            pos = 1
            while True:
                U.append(pos)
                if _is_golomb(U):
                    break
                U.pop()
                pos += 1
        candidates.append(U)

    # Evaluate and compress
    for U in candidates:
        if time.time() > deadline:
            break
        # Compress to reduce M
        U_comp = _compress_golomb(U)
        s = _score_set(U_comp)
        if s > best_score:
            best_score = s
            best_U = U_comp[:]

    # Local search on the best candidate
    if time.time() < deadline:
        # Try to further optimize by random moves
        U = best_U[:]
        max_u = max(U)
        for _ in range(500):
            if time.time() > deadline:
                break
            # Pick a random element (not 0)
            i = random.randint(1, len(U) - 1)
            old_val = U[i]
            # Try moving it
            for offset in range(-10, 11):
                new_val = old_val + offset
                if new_val <= 0:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(U):
                    continue
                if _is_golomb(new_U):
                    s = _score_set(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        break

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
