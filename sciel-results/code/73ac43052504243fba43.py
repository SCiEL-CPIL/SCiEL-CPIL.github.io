"""Rewritten program: local search with 1-for-2 and 2-for-1 swaps plus random restarts."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_DENOMINATORS = list(range(2, 49))
_N = len(_DENOMINATORS)
_RECIPS = [Fraction(1, d) for d in _DENOMINATORS]

def _sum_of(mask):
    """Compute the sum of reciprocals for the given bitmask."""
    s = Fraction()
    for i in range(_N):
        if mask & (1 << i):
            s += _RECIPS[i]
    return s

def _local_search(mask, deadline):
    """
    Perform local search starting from mask.
    Tries 1-for-1, 1-for-2, and 2-for-1 swaps.
    Returns (best_mask, best_sum).
    """
    best_mask = mask
    best_sum = _sum_of(mask)
    # Ensure strictly below 1
    if best_sum >= 1:
        return None, None

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break

        # 1-for-1 swap: remove i, add j
        for i in range(_N):
            if not (mask & (1 << i)):
                continue
            for j in range(_N):
                if (mask & (1 << j)):
                    continue
                new_mask = mask ^ (1 << i) ^ (1 << j)
                new_sum = best_sum - _RECIPS[i] + _RECIPS[j]
                if new_sum < 1 and new_sum > best_sum:
                    best_mask = new_mask
                    best_sum = new_sum
                    improved = True

        if improved:
            continue

        # 1-for-2 swap: remove i, add j and k
        for i in range(_N):
            if not (mask & (1 << i)):
                continue
            for j in range(_N):
                if (mask & (1 << j)):
                    continue
                for k in range(j + 1, _N):
                    if (mask & (1 << k)):
                        continue
                    new_mask = mask ^ (1 << i) ^ (1 << j) ^ (1 << k)
                    new_sum = best_sum - _RECIPS[i] + _RECIPS[j] + _RECIPS[k]
                    if new_sum < 1 and new_sum > best_sum:
                        best_mask = new_mask
                        best_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # 2-for-1 swap: remove i and j, add k
        for i in range(_N):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, _N):
                if not (mask & (1 << j)):
                    continue
                for k in range(_N):
                    if (mask & (1 << k)):
                        continue
                    new_mask = mask ^ (1 << i) ^ (1 << j) ^ (1 << k)
                    new_sum = best_sum - _RECIPS[i] - _RECIPS[j] + _RECIPS[k]
                    if new_sum < 1 and new_sum > best_sum:
                        best_mask = new_mask
                        best_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    if best_sum < 1:
        return best_mask, best_sum
    return None, None

def _greedy_start(deadline):
    """
    Greedy construction: add reciprocals in order that keeps sum < 1,
    but try to maximize sum by considering all candidates.
    """
    mask = 0
    s = Fraction()
    # Try adding in descending order of value (i.e., ascending denominator)
    # but with a twist: consider all, pick the one that gets closest to 1 without exceeding
    remaining = list(range(_N))
    while remaining:
        if time.time() > deadline:
            break
        best_idx = -1
        best_sum = s
        for idx in remaining:
            new_s = s + _RECIPS[idx]
            if new_s < 1 and new_s > best_sum:
                best_sum = new_s
                best_idx = idx
        if best_idx == -1:
            break
        mask |= (1 << best_idx)
        s = best_sum
        remaining.remove(best_idx)
    return mask, s

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s to be safe

    best_mask = 0
    best_sum = Fraction()

    # First, do a greedy start
    mask, s = _greedy_start(deadline)
    if s < 1 and s > best_sum:
        best_mask = mask
        best_sum = s

    # Local search from greedy start
    lm, ls = _local_search(mask, deadline)
    if lm is not None and ls > best_sum:
        best_mask = lm
        best_sum = ls

    # Random restarts
    random.seed(42)
    while time.time() < deadline:
        # Random subset
        rmask = 0
        rs = Fraction()
        for i in range(_N):
            if random.random() < 0.5:
                rmask |= (1 << i)
                rs += _RECIPS[i]
        if rs >= 1:
            continue
        if rs > best_sum:
            best_mask = rmask
            best_sum = rs

        # Local search from random start
        lm, ls = _local_search(rmask, deadline)
        if lm is not None and ls > best_sum:
            best_mask = lm
            best_sum = ls

    # Convert mask to indices
    result = [i for i in range(_N) if best_mask & (1 << i)]
    if not result:
        # Fallback: just pick index 0 (denominator 2)
        result = [0]
    return result
# EVOLVE-BLOCK-END
