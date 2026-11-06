"""Optimized solver: maximize |S| subset of [1,256] with (a*b)%(a+b)!=0 for all distinct a,b."""
import time
import random
from itertools import combinations

DEADLINE = 58.0  # safety margin before 60s hard kill

def _bad(a, b):
    return (a * b) % (a + b) == 0

def _greedy_build():
    """Greedy: add numbers in ascending order if they don't conflict with existing set."""
    S = []
    for x in range(1, 257):
        ok = True
        for y in S:
            if _bad(x, y):
                ok = False
                break
        if ok:
            S.append(x)
    return set(S)

def _greedy_build_desc():
    """Greedy descending."""
    S = []
    for x in range(256, 0, -1):
        ok = True
        for y in S:
            if _bad(x, y):
                ok = False
                break
        if ok:
            S.append(x)
    return set(S)

def _greedy_build_random(seed):
    """Greedy in random order."""
    rng = random.Random(seed)
    order = list(range(1, 257))
    rng.shuffle(order)
    S = []
    for x in order:
        ok = True
        for y in S:
            if _bad(x, y):
                ok = False
                break
        if ok:
            S.append(x)
    return set(S)

def _count_conflicts(x, S):
    """Count how many elements in S conflict with x."""
    cnt = 0
    for y in S:
        if _bad(x, y):
            cnt += 1
    return cnt

def _local_search(S, deadline, rng):
    """Local search with add, remove, and swap moves. Returns best set found."""
    S = set(S)
    best = len(S)
    best_set = set(S)
    start = time.time()
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline:
        improved = False
        # Try add moves: for each x not in S, if conflicts==0, add it
        # We do a random subset of candidates for speed
        candidates = [x for x in range(1, 257) if x not in S]
        if candidates:
            # Try a random sample of candidates
            sample_size = min(len(candidates), 50)
            sample = rng.sample(candidates, sample_size)
            for x in sample:
                if _count_conflicts(x, S) == 0:
                    S.add(x)
                    improved = True
                    if len(S) > best:
                        best = len(S)
                        best_set = set(S)
                        no_improve = 0
                    break  # restart after one add

        if not improved:
            # Try swap moves: remove one, add one
            s_list = list(S)
            if s_list and candidates:
                # Random swap attempt
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    a = rng.choice(s_list)
                    x = rng.choice(candidates)
                    # Check if removing a and adding x is beneficial
                    # After removing a, can we add x?
                    new_conflicts = 0
                    for y in S:
                        if y != a and _bad(x, y):
                            new_conflicts += 1
                    if new_conflicts == 0:
                        S.remove(a)
                        S.add(x)
                        # Now try to add more
                        for x2 in candidates:
                            if x2 not in S and _count_conflicts(x2, S) == 0:
                                S.add(x2)
                                if len(S) > best:
                                    best = len(S)
                                    best_set = set(S)
                                    no_improve = 0
                                break
                        improved = True
                        break

        if not improved:
            # Try remove-then-add: remove an element that has many conflicts, then add
            if s_list:
                # Find element with most conflicts (expensive, do sparsely)
                if no_improve % 10 == 0:
                    max_conf = -1
                    worst_elem = None
                    for a in s_list:
                        c = 0
                        for y in S:
                            if y != a and _bad(a, y):
                                c += 1
                        if c > max_conf:
                            max_conf = c
                            worst_elem = a
                    if worst_elem is not None and max_conf > 0:
                        S.remove(worst_elem)
                        # Try to add candidates
                        for x in candidates:
                            if x not in S and _count_conflicts(x, S) == 0:
                                S.add(x)
                                if len(S) > best:
                                    best = len(S)
                                    best_set = set(S)
                                    no_improve = 0
                                break
                        improved = True

        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0

    return best_set

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(42)

    # Build initial candidates from different greedy strategies
    candidates = []
    candidates.append(_greedy_build())
    candidates.append(_greedy_build_desc())
    for i in range(20):
        candidates.append(_greedy_build_random(i))

    # Pick the best initial candidate
    best_init = max(candidates, key=len)

    # Local search from best initial
    result = _local_search(best_init, deadline, rng)

    # If time remains, try local search from other good candidates
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Sort candidates by size descending, try next best
        candidates_sorted = sorted(candidates, key=len, reverse=True)
        for cand in candidates_sorted[1:]:
            if time.time() >= deadline - 2.0:
                break
            if len(cand) + 1 < len(result):
                continue  # can't beat current best even with +1
            t_end = min(deadline - 1.0, time.time() + max(1.0, remaining * 0.15))
            local = _local_search(cand, t_end, rng)
            if len(local) > len(result):
                result = local
            remaining = deadline - time.time()
            if remaining < 3.0:
                break

    return sorted(result)
