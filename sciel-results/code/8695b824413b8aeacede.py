"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations
import numpy as np
from numpy.fft import fft, ifft


def _fft_len(n):
    """Smallest power of two >= n (fast for numpy FFT)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _score_set_fft(U):
    """Score a finite set U (must contain 0) via FFT convolution.

    score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
    """
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    sum_len = 2 * max_u + 1
    L = _fft_len(2 * sum_len)

    ind = np.zeros(L, dtype=np.float64)
    ind[:max_u + 1][U] = 1.0          # indicator over [0, max_u]

    f = fft(ind)
    conv = ifft(f * f).real
    sumset_size = int(np.count_nonzero(conv[:sum_len] > 1e-6))

    # U - U: convolve indicator with its reversal.
    rev = np.zeros(L, dtype=np.float64)
    rev[:max_u + 1] = ind[:max_u + 1][::-1]
    diff_conv = ifft(f * fft(rev)).real
    # diff value d in [-max_u, max_u] sits at index d + max_u
    diffset_size = int(np.count_nonzero(diff_conv[max_u:max_u + 2 * max_u + 1] > 1e-6))

    if sumset_size == 0 or diffset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0.0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0.0:
        return 1.0
    return 1.0 + log_ratio / log_denom


def _score_set(U):
    """Dispatch: exact O(n^2) for small sets, FFT for larger ones."""
    U = sorted(set(U))
    if len(U) <= 40 or max(U) <= 400:
        U_set = set(U)
        n = len(U_set)
        if n == 0:
            return 0.0
        max_u = max(U_set)
        if max_u == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        if not sumset:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0.0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0.0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    return _score_set_fft(U)


def _golomb_ruler(n):
    """Greedy Golomb ruler (unique positive differences)."""
    known = {
        1: [0], 2: [0, 1], 3: [0, 1, 3], 4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 12], 6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22], 8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36], 10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            diffs = set()
            ok = True
            for j in range(len(cand)):
                for k in range(j + 1, len(cand)):
                    d = cand[k] - cand[j]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _search_small_sets(max_n=6, max_val=40, deadline=None):
    """Exhaustive search over small sets (fast exact scorer)."""
    best_score = 0.0
    best_U = [0, 1, 3]
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
            U = [0] + list(combo)
            U.sort()
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    return best_U, best_score


def _dual_grid_candidates(deadline=None):
    """Structured candidates: CRT dual-grid APs, hybrid AP+tail, Golomb, random."""
    cands = []
    random.seed(12345)

    # 1) Dual-grid: multiples of L = lcm(m1, m2) up to a cap (sparse AP, includes 0).
    for m1 in range(1, 41):
        if deadline and time.time() > deadline:
            return cands
        for m2 in range(1, 41):
            L = m1 // math.gcd(m1, m2) * m2
            for cap in (2 * L, 3 * L, 4 * L, 5 * L):
                if L > 0 and cap <= 20000:
                    k = cap // L
                    if k >= 2:
                        cands.append([L * i for i in range(k + 1)])

    # 2) Hybrid: short AP base + sparse tail at large gaps.
    for k in range(3, 12):
        base = list(range(k))
        for m in range(1, 31):
            for t in range(1, 4):
                tail = [m * (k + j) for j in range(1, t + 1)]
                U = sorted(set(base + tail))
                if len(U) > 2:
                    cands.append(U)

    # 3) Golomb rulers (known to keep difference set large).
    for n in range(3, 13):
        cands.append(_golomb_ruler(n))

    # 4) Random sets over a variety of sizes/ranges.
    for _ in range(60):
        if deadline and time.time() > deadline:
            return cands
        n = random.randint(3, 18)
        max_val = random.choice([20, 50, 120, 300, 800, 2000, 6000, 15000])
        if max_val >= n - 1:
            U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
            cands.append(U)

    # 5) Pure sparse APs with a few sizes.
    for m in range(2, 60):
        for k in range(3, 12):
            cands.append([m * i for i in range(k)])
    return cands


def _local_search(U, max_iters=600, deadline=None):
    """Improve set U via move/add/remove local search (FFT-backed scorer)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + max(20, max_u // 3)

    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # Move one element (skip 0).
        for i in range(len(U)):
            if improved:
                break
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
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
                    improved = True
                    break

        if improved:
            continue

        # Add a new element.
        for new_val in range(1, upper_bound + 1):
            if new_val in best_U:
                continue
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break

        if improved:
            continue

        # Remove an element (keep 0, keep size > 3).
        if len(best_U) > 4:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i + 1:]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break

        if not improved:
            break

    return sorted(set(best_U)), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # 1) Exhaustive small-set search (fast exact scorer, broad range).
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=40, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass

    # 2) Structured candidates (dual-grid, hybrid, Golomb, random) via fast FFT scorer.
    candidates = _dual_grid_candidates(deadline=deadline)
    scored = []
    for U in candidates:
        if time.time() > deadline:
            break
        try:
            s = _score_set(U)
        except Exception:
            continue
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    # 3) Local search from the global best.
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=600, deadline=deadline)

    # 4) Local search from the other top candidates (diversity).
    scored.sort(key=lambda x: -x[0])
    for s, U in scored:
        if time.time() > deadline - 5:
            break
        if s < best_score - 0.02:
            break
        if U == best_U:
            continue
        improved_U, improved_score = _local_search(U, max_iters=250, deadline=deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U

    # 5) Final polish pass on the best set.
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iters=300, deadline=deadline)

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
