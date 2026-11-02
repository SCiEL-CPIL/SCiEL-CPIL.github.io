"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    num_sums = len(sum_set)
    num_diffs = len(diff_set)
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _golomb_seed(n, max_val):
    """Greedy Golomb ruler: pick n elements with distinct pairwise differences."""
    U = [0]
    diffs = set()
    for _ in range(1, n):
        best_v = None
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_v = v
                break
        if best_v is None:
            break
        U.append(best_v)
        for a in U:
            diffs.add(abs(best_v - a))
    return U

def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False

        # Try adding a new element
        candidates = []
        for delta in range(1, 40):
            candidates.append(max_val + delta)
        for delta in range(1, 25):
            candidates.append(max_val * 2 + delta)
        # Some interior values
        for v in range(1, max_val + 1):
            if v not in current_U:
                candidates.append(v)
        random.shuffle(candidates)

        for new_val in candidates[:15]:
            if new_val <= 0:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                break

        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:8]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        break

        if not improved and step > 50:
            if random.random() < 0.05:
                # Perturb: replace a random element
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    new_val = random.randint(1, max_val * 2)
                    if new_val not in current_U:
                        current_U[idx] = new_val
                        current_U.sort()
                        current_score = _score_set(current_U)
            else:
                break

    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Try geometric progressions
    for base in range(2, 30):
        if _elapsed() > _DEADLINE - 40:
            break
        for n in range(2, 12):
            if _elapsed() > _DEADLINE - 40:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= base
                U.append(val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # Try small sets {0,1,a,b}, {0,1,a,b,c}
    for a in range(2, 25):
        if _elapsed() > _DEADLINE - 30:
            break
        for b in range(a + 1, 150):
            if _elapsed() > _DEADLINE - 30:
                break
            U = [0, 1, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    for a in range(2, 15):
        if _elapsed() > _DEADLINE - 20:
            break
        for b in range(a + 1, 80):
            if _elapsed() > _DEADLINE - 20:
                break
            for c in range(b + 1, 300):
                if _elapsed() > _DEADLINE - 20:
                    break
                U = [0, 1, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # Try Golomb rulers as seeds
    for n in range(4, 10):
        if _elapsed() > _DEADLINE - 20:
            break
        for max_val in [n * n, n * n + n, n * n * 2]:
            if _elapsed() > _DEADLINE - 20:
                break
            U = _golomb_seed(n, max_val)
            if len(U) >= n - 1:
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # Local search from best
    if _elapsed() < _DEADLINE - 5:
        U, score = _local_search(best_U, max_steps=1500)
        if score > best_score:
            best_score = score
            best_U = U

    # Local search from Golomb seeds
    for n in [5, 6, 7]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = _golomb_seed(n, n * n + n)
        if len(U) >= 4:
            U2, score2 = _local_search(U, max_steps=800)
            if score2 > best_score:
                best_score = score2
                best_U = U2

    # Local search from geometric seeds
    for base in [2, 3, 4, 5]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = [0]
        val = 1
        for i in range(1, 6):
            val *= base
            U.append(val)
        U2, score2 = _local_search(U, max_steps=800)
        if score2 > best_score:
            best_score = score2
            best_U = U2

    return sorted(best_U)
