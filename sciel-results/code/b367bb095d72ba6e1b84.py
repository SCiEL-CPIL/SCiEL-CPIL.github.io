"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math
from math import gcd as _gcd
import sys
import random
sys.setrecursionlimit(100000)

def _build_coprime_masks():
    """Build coprime bitmask for each node 0..510 (representing values 1..511)."""
    n = 511
    coprime_mask = [0] * n
    for i in range(n):
        vi = i + 1
        mask = 0
        for j in range(n):
            if j == i:
                continue
            if _gcd(vi, j + 1) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask
    return coprime_mask

def _greedy_build(coprime_mask, weights, n, order):
    """Build a solution greedily following the given order."""
    chosen = set()
    total = 0.0
    for v in order:
        ok = True
        for c in chosen:
            if not (coprime_mask[v] >> c) & 1:
                ok = False
                break
        if ok:
            chosen.add(v)
            total += weights[v]
    return total, chosen

def _local_search(coprime_mask, weights, n, start_set, deadline):
    """Hill-climbing local search with add, swap, and 2-swap moves."""
    cur_set = set(start_set)
    cur_w = sum(weights[v] for v in cur_set)
    improved = True
    while improved and _time.time() < deadline:
        improved = False
        # Try adding each unchosen node
        for v in range(n):
            if v in cur_set:
                continue
            ok = True
            for c in cur_set:
                if not (coprime_mask[v] >> c) & 1:
                    ok = False
                    break
            if ok:
                cur_set.add(v)
                cur_w += weights[v]
                improved = True
                break
        # Try swapping: remove one, add one better
        if not improved:
            for rem in list(cur_set):
                new_w = cur_w - weights[rem]
                # Find best replacement
                best_add = -1
                best_gain = 0.0
                for v in range(n):
                    if v in cur_set or v == rem:
                        continue
                    ok = True
                    for c in cur_set:
                        if c == rem:
                            continue
                        if not (coprime_mask[v] >> c) & 1:
                            ok = False
                            break
                    if ok:
                        gain = weights[v] - weights[rem]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = v
                if best_add >= 0:
                    cur_set.discard(rem)
                    cur_set.add(best_add)
                    cur_w += best_gain
                    improved = True
                    break
        # Try 2-swap: remove two, add two
        if not improved and len(cur_set) >= 2:
            rem_list = list(cur_set)
            for i in range(len(rem_list)):
                for j in range(i+1, len(rem_list)):
                    r1, r2 = rem_list[i], rem_list[j]
                    base_w = cur_w - weights[r1] - weights[r2]
                    # Find best pair to add
                    best_pair = None
                    best_gain = 0.0
                    for v1 in range(n):
                        if v1 in cur_set:
                            continue
                        ok1 = True
                        for c in cur_set:
                            if c == r1 or c == r2:
                                continue
                            if not (coprime_mask[v1] >> c) & 1:
                                ok1 = False
                                break
                        if not ok1:
                            continue
                        for v2 in range(n):
                            if v2 in cur_set or v2 == v1:
                                continue
                            ok2 = True
                            for c in cur_set:
                                if c == r1 or c == r2:
                                    continue
                                if not (coprime_mask[v2] >> c) & 1:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            if not (coprime_mask[v1] >> v2) & 1:
                                continue
                            gain = weights[v1] + weights[v2] - weights[r1] - weights[r2]
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (v1, v2)
                    if best_pair is not None:
                        cur_set.discard(r1)
                        cur_set.discard(r2)
                        cur_set.add(best_pair[0])
                        cur_set.add(best_pair[1])
                        cur_w += best_gain
                        improved = True
                        break
                if improved:
                    break
    return cur_w, cur_set

def construct():
    start_time = _time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n = 511
    coprime_mask = _build_coprime_masks()
    weights = [1.0 / (512 - (i + 1)) for i in range(n)]
    
    best_w = 0.0
    best_s = set()
    
    # Multiple greedy starts with different orderings
    rng = random.Random(12345)
    
    orderings = []
    # Weight descending (highest contribution first)
    orderings.append(sorted(range(n), key=lambda x: -weights[x]))
    # Weight descending with random perturbation
    for _ in range(10):
        orderings.append(sorted(range(n), key=lambda x: (-weights[x] + rng.random()*0.01, x)))
    # Random orderings
    for _ in range(5):
        orderings.append(list(range(n)))
        rng.shuffle(orderings[-1])
    
    for order in orderings:
        if _time.time() > deadline:
            break
        w, s = _greedy_build(coprime_mask, weights, n, order)
        if w > best_w:
            best_w = w
            best_s = set(s)
    
    # Local search from best solution
    lw, ls = _local_search(coprime_mask, weights, n, best_s, deadline)
    if lw > best_w:
        best_w = lw
        best_s = ls
    
    # More random restarts + local search if time permits
    for attempt in range(30):
        if _time.time() > deadline:
            break
        # Randomized greedy
        order = list(range(n))
        rng.shuffle(order)
        w, s = _greedy_build(coprime_mask, weights, n, order)
        # Local search
        lw2, ls2 = _local_search(coprime_mask, weights, n, s, deadline)
        if lw2 > best_w:
            best_w = lw2
            best_s = ls2
    
    # Final polish: try to add any remaining nodes
    final_set = set(best_s)
    for v in range(n):
        if v in final_set:
            continue
        ok = True
        for c in final_set:
            if not (coprime_mask[v] >> c) & 1:
                ok = False
                break
        if ok:
            final_set.add(v)
    
    if sum(weights[v] for v in final_set) > best_w:
        best_s = final_set
    
    return [v + 1 for v in sorted(best_s)]
# EVOLVE-BLOCK-END
