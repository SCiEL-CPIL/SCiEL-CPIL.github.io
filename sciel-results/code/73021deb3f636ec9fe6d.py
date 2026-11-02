"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

# ---------------------------------------------------------------------------
# Scoring (FFT-based, exact for the sizes we use)
# ---------------------------------------------------------------------------
def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)

    # Sumset: autocorrelation of indicator (convolution with itself)
    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:2 * max_u + 1] > 1e-9))

    # Difference set: count nonzero autocorrelation lags in [-max_u, max_u]
    autocorr = ifft(freq * np.conj(freq)).real
    # Vectorized count: lags 1..max_u and their negatives
    if max_u >= 1:
        pos_lags = autocorr[1:max_u + 1] > 1e-9
        neg_lags = autocorr[fft_len - max_u:fft_len - 0][::-1] > 1e-9
        diffset_size = 1 + int(np.sum(pos_lags)) + int(np.sum(neg_lags))
    else:
        diffset_size = 1

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

    if n <= 25:
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
    return _score_set_fft(U)


# ---------------------------------------------------------------------------
# Seed families
# ---------------------------------------------------------------------------
def _golomb_ruler(n):
    """Known (near-)optimal Golomb rulers: unique pairwise differences."""
    known = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    }
    if n in known:
        return known[n][:]
    if n <= 1:
        return [0]
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(cand)):
                for k in range(j + 1, len(cand)):
                    d = cand[k] - cand[j]
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


def _sidon_ruler(n):
    """Greedy Sidon (B2) set: few distinct pairwise sums, differences near-maximal."""
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            sums = set()
            valid = True
            for i in range(len(cand)):
                for j in range(i, len(cand)):
                    s = cand[i] + cand[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


# ---------------------------------------------------------------------------
# Local search (gap-based moves + add/remove + perturbation)
# ---------------------------------------------------------------------------
def _local_search(U, max_iters=5000, deadline=None, polish=False):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200
    tol = 1e-12 if polish else 1e-9
    move_range = 80 if polish else 40
    n_rand = 20 if polish else 12
    max_size = 70 if polish else 60

    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # 1) swap / move one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            for offset in range(-move_range, move_range + 1):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(n_rand):
                candidates.add(random.randint(1, upper_bound))
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + tol:
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
            # 2) add an element
            if len(U) < max_size:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            # 3) remove an element
            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

        if not improved:
            # 4) larger random perturbation to escape local optima
            if random.random() < 0.2:
                new_U = U[:]
                for _ in range(min(4, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
            if not improved:
                break

    return sorted(set(best_U)), best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before the 150s limit

    random.seed(42)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # --- Sidon / B2 sets: directly minimize the sumset (the key refinement) ---
    for n in range(3, 22):
        if time.time() > deadline - 70:
            break
        candidates.append(_sidon_ruler(n))

    # --- Golomb rulers: maximize the difference set ---
    for n in range(3, 22):
        if time.time() > deadline - 70:
            break
        candidates.append(_golomb_ruler(n))

    # --- Structured: arithmetic progressions with various bases ---
    for base in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline - 70:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:min(len(U), 25)])

    # --- Powers-based sets ---
    for p in [2, 3]:
        if time.time() > deadline - 70:
            break
        U = [0]
        for i in range(1, 15):
            U.append(p ** i)
            if U[-1] > 20000:
                break
        candidates.append(U)

    # --- Mixed structured: powers with small offsets ---
    for p in [2, 3]:
        if time.time() > deadline - 70:
            break
        U = [0]
        for i in range(1, 12):
            val = p ** i + random.randint(-5, 5)
            if val > 0 and val not in U:
                U.append(val)
        candidates.append(sorted(U))

    # --- Random sets ---
    for trial in range(30):
        if time.time() > deadline - 70:
            break
        n = random.randint(3, 40)
        max_val = random.randint(20, 600)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # --- Quasi-random: golden ratio based ---
    for n in range(5, 25):
        if time.time() > deadline - 70:
            break
        U = [0]
        for i in range(1, n):
            val = int((i * (1 + 5 ** 0.5) / 2) % 1 * 200) + 1
            if val not in U:
                U.append(val)
        candidates.append(sorted(U))

    # --- Fibonacci-based ---
    for n in range(5, 20):
        if time.time() > deadline - 70:
            break
        U = [0]
        fib = [1, 1]
        for i in range(2, 30):
            fib.append(fib[-1] + fib[-2])
        for i in range(1, min(n, len(fib))):
            if fib[i] not in U:
                U.append(fib[i])
        candidates.append(sorted(U))

    # Evaluate all candidates, keep the top few for polishing
    scored = []
    for U in candidates:
        if time.time() > deadline - 40:
            break
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    scored.sort(key=lambda x: -x[0])

    # Polish the best candidate first, then the next-best distinct ones
    seen = []
    for s, U in scored:
        if time.time() > deadline - 15:
            break
        key = tuple(U)
        if key in seen:
            continue
        seen.append(key)
        if s < best_score - 0.03:
            continue
        iters = 6000 if len(seen) == 1 else 2000
        improved_U, improved_score = _local_search(U, max_iters=iters, deadline=deadline - 15)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U

    # Final high-precision polish pass on the best result
    if time.time() < deadline - 5 and len(best_U) > 2:
        polished_U, polished_score = _local_search(best_U, max_iters=5000, deadline=deadline - 3, polish=True)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
