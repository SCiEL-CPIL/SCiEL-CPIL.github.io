"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0 using NumPy for speed."""
    U_set = sorted(set(U))
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = U_set[-1]
    if max_u == 0:
        return 1.0

    U_arr = np.array(U_set, dtype=np.int64)

    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]

    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))

    if sumset_size == 0:
        return 0.0

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0

    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)

    if log_denom == 0:
        return 1.0

    return 1.0 + log_ratio / log_denom

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed on larger sets."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # Sumset via convolution of indicator with itself
    sum_len = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = np.fft.fft(indicator)
    conv = np.fft.ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 1e-6))

    # Difference set: U - U. Use reverse indicator.
    rev_indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev_indicator[fft_len - 1 - u] = 1.0
    freq_rev = np.fft.fft(rev_indicator)
    diff_conv = np.fft.ifft(freq * freq_rev).real
    # Difference values d = a - b range [-max_u, max_u]; in convolution they sit
    # at index (fft_len - 1 - max_u) + (a - b) + max_u ... we just count nonzero
    # over the full range and subtract the guaranteed zero at 0 handled naturally.
    diffset_size = int(np.sum(np.abs(diff_conv) > 1e-6))

    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score(U):
    """Dispatch: NumPy for small sets, FFT for larger ones."""
    n = len(set(U))
    if n <= 40:
        return _score_set(U)
    return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n. Known optimal for small n, greedy for larger."""
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

    ruler = [0]
    for i in range(1, n):
        pos = ruler[-1] + 1
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

def _local_search(U, deadline, max_window=None):
    """
    Systematic greedy coordinate-ascent local search (refinement from inspiration 1).
    At each pass tries every single-element swap to every value in a window,
    every add, and every remove; accepts the single best improvement.
    """
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    if max_window is None:
        max_window = max_u + 30

    no_improve = 0
    while time.time() < deadline:
        improved = False
        U = best_U
        current = best_score

        # --- Swap: replace one element with any value in window ---
        for i in range(len(U)):
            old_val = U[i]
            for new_val in range(0, max_window + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                if 0 not in new_U:
                    continue
                s = _score(new_U)
                if s > current + 1e-12:
                    current = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        if improved:
            no_improve = 0
            continue

        # --- Add: insert any value in window ---
        if len(U) < 45:
            for new_val in range(0, max_window + 1):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = _score(new_U)
                if s > current + 1e-12:
                    current = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

        # --- Remove: drop any non-zero element ---
        if len(U) > 2:
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                new_U = U[:i] + U[i+1:]
                s = _score(new_U)
                if s > current + 1e-12:
                    current = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

        # No improvement this pass
        no_improve += 1
        # Widen window slightly to escape local minima, then give up if stuck
        if no_improve >= 3:
            break
        max_window = max_window + 10

    best_score = _score(best_U)
    return sorted(set(best_U)), best_score

def _simulated_annealing(U_init, deadline, max_iters=50000):
    """
    Improve set U using Simulated Annealing (from current program).
    Moves: Swap an element, Add an element, Remove an element.
    """
    U = sorted(set(U_init))
    current_score = _score(U)
    best_U = U[:]
    best_score = current_score

    T = 0.5
    cooling_rate = 0.9995
    max_val_limit = 200

    for i in range(max_iters):
        if time.time() > deadline:
            break

        U_new = U[:]
        move_type = random.choice(['swap', 'add', 'remove'])

        if move_type == 'swap':
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                if random.random() < 0.7:
                    new_val = max(1, min(max_val_limit, old_val + random.randint(-10, 10)))
                else:
                    new_val = random.randint(1, max_val_limit)
                if new_val != old_val and new_val not in U:
                    U_new[idx] = new_val
                    U_new = sorted(set(U_new))
                    if len(U_new) != len(U):
                        continue
                else:
                    continue
        elif move_type == 'add':
            if len(U) < 15:
                new_val = random.randint(1, max_val_limit)
                if new_val not in U:
                    U_new.append(new_val)
                    U_new = sorted(U_new)
                else:
                    continue
        elif move_type == 'remove':
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U_new.pop(idx)

        if not U_new or 0 not in U_new:
            continue

        new_score = _score(U_new)

        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            U = U_new
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]

        T *= cooling_rate
        if T < 1e-4:
            T = 1e-4

    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    best_U = [0, 1, 3]
    best_score = _score(best_U)

    # 1. Exhaustive search for small n (bitmask-friendly ranges)
    try:
        for n in range(3, 7):
            max_val = 50 if n <= 5 else 40
            for combo in combinations(range(1, max_val + 1), n - 1):
                if time.time() > deadline - 30:
                    break
                U = [0] + list(combo)
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
            if time.time() > deadline - 30:
                break
    except Exception:
        pass

    # 2. Heuristic candidates
    candidates = []

    for n in range(3, 15):
        if time.time() > deadline - 20:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)

    for k in range(3, 10):
        for M in range(k, 100):
            if time.time() > deadline - 20:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)

    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        n = random.randint(4, 12)
        max_val = random.randint(50, 200)
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        candidates.append(U)

    for U in candidates:
        if time.time() > deadline - 10:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # 3. Simulated Annealing on the best candidate
    if time.time() < deadline - 20:
        sa_deadline = min(deadline - 15, time.time() + 50)
        best_U, best_score = _simulated_annealing(best_U, sa_deadline, max_iters=100000)

    # 4. Systematic local-search polish (refinement from inspiration 1)
    if time.time() < deadline - 5:
        ls_deadline = min(deadline - 2, time.time() + 40)
        best_U, best_score = _local_search(best_U, ls_deadline)

    # 5. If time remains, polish a couple of other strong candidates
    remaining_time = deadline - time.time()
    if remaining_time > 8:
        other_candidates = [U for U in candidates if U != best_U]
        # Score and sort cheaply (top few only)
        scored = []
        for U in other_candidates[:20]:
            if time.time() > deadline - 5:
                break
            scored.append((_score(U), U))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, U in scored[:3]:
            if time.time() > deadline - 5:
                break
            ls_deadline = min(deadline - 2, time.time() + (remaining_time / 3))
            candidate_U, candidate_score = _local_search(U, ls_deadline)
            if candidate_score > best_score:
                best_score = candidate_score
                best_U = candidate_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
