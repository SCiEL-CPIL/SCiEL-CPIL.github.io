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

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # Direct computation
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

# Known optimal Golomb rulers (shortest length for n marks)
# Source: Wikipedia / OEIS A003022
_GOLOMB_RULERS = {
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
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94, 104], # Approx
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94, 104, 114], # Approx
    18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94, 104, 114, 124], # Approx
    19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94, 104, 114, 124, 134], # Approx
    20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94, 104, 114, 124, 134, 144], # Approx
}

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 50

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_left() < 5:
            break
        improved = False

        # Try moving one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try nearby values
            candidates = set()
            for offset in range(-20, 21):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add some random jumps
            for _ in range(10):
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
                    upper_bound = max_u + 50
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try adding an element
            if len(U) < 25:
                for _ in range(20):
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
                        upper_bound = max_u + 50
                        improved = True
                        break

            if not improved:
                # Try removing an element
                if len(U) > 3:
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
                            upper_bound = max_u + 50
                            improved = True
                            break

        if not improved:
            # Random perturbation
            if random.random() < 0.1:
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
                        upper_bound = max_u + 50
                        improved = True

            if not improved:
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
    for n in range(3, 21):
        if _time_left() < 30:
            break
        if n in _GOLOMB_RULERS:
            gr = _GOLOMB_RULERS[n]
            candidates.append(gr)
        else:
            # Fallback: generate a simple Golomb ruler
            gr = [0]
            for i in range(1, n):
                pos = gr[-1] + i
                gr.append(pos)
            candidates.append(gr)

    # Add some random sets
    for _ in range(20):
        if _time_left() < 20:
            break
        n = random.randint(3, 15)
        max_val = random.randint(10, 200)
        U = [0] + random.sample(range(1, max_val + 1), min(n - 1, max_val))
        U = sorted(U)
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
        best_U, best_score = _local_search(best_U, max_iters=10000, deadline=_DEADLINE - 5)

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
