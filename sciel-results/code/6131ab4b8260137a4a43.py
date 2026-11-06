import time
import random

def _count_progression_diffs(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    S_list = list(S)
    diffs = set()
    for a in S_list:
        for b in S_list:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    random.seed(12345)
    N = 32
    MAXV = 512

    # Initial set: random
    current = set(random.sample(range(1, MAXV + 1), N))
    current_score = _count_progression_diffs(current)
    best = set(current)
    best_score = current_score

    # Simulated annealing
    T = 30.0
    T_min = 0.01
    cooling = 0.9995

    while time.time() < deadline - 20.0 and T > T_min:
        # Generate a move: swap one element for a random new one
        elem = random.choice(list(current))
        new_elem = random.randint(1, MAXV)
        if new_elem in current:
            continue
        current.remove(elem)
        current.add(new_elem)
        new_score = _count_progression_diffs(current)
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-9)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            # reject
            current.remove(new_elem)
            current.add(elem)
        T *= cooling

    # Local search polish on best
    # Reduced trials per element to speed up each iteration
    for _ in range(50000):
        if time.time() > deadline - 2.0:
            break
        improved = False
        # try all single swaps
        best_swap = None
        best_swap_score = best_score
        blist = list(best)
        for i, elem in enumerate(blist):
            # try replacing elem with 1 random candidate
            cand = random.randint(1, MAXV)
            if cand in best:
                continue
            test = set(best)
            test.remove(elem)
            test.add(cand)
            sc = _count_progression_diffs(test)
            if sc > best_swap_score:
                best_swap_score = sc
                best_swap = (elem, cand)
        if best_swap is not None:
            elem, cand = best_swap
            best.remove(elem)
            best.add(cand)
            best_score = best_swap_score
            improved = True
        if not improved:
            # try a random perturbation to escape
            elem = random.choice(list(best))
            cand = random.randint(1, MAXV)
            if cand not in best:
                best.remove(elem)
                best.add(cand)
                best_score = _count_progression_diffs(best)

    return list(best)
