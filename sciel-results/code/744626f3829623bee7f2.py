"""Improved program: cardinality-indexed subset-sum DP to maximize set size."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill


def _build_dp_state(S):
    """Build sums_by_k DP table for set S."""
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _check_add_valid(sums_by_k, candidate):
    """
    Check if adding candidate to the set (with DP state sums_by_k) keeps validity.
    Only checks new sums involving candidate against existing sums and each other.
    """
    n = len(sums_by_k) - 1
    # New sums at cardinality k: {s + candidate for s in sums_by_k[k-1]}
    # These must not collide with:
    # 1. Any existing sums_by_j[j] for j != k
    # 2. New sums at other cardinalities
    
    # Collect all new sums by cardinality
    new_sums_by_k = {}
    for k in range(1, n + 2):
        if k - 1 < len(sums_by_k) and sums_by_k[k - 1]:
            ns = set(s + candidate for s in sums_by_k[k - 1])
            if ns:
                new_sums_by_k[k] = ns
    
    # Check 1: new sums at cardinality k don't collide with existing sums at cardinality j != k
    for k, ns in new_sums_by_k.items():
        for j in range(len(sums_by_k)):
            if j == k:
                continue
            if sums_by_k[j] and ns & sums_by_k[j]:
                return False
    
    # Check 2: new sums at different cardinalities don't collide with each other
    new_ks = list(new_sums_by_k.keys())
    for i in range(len(new_ks)):
        for j in range(i + 1, len(new_ks)):
            if new_sums_by_k[new_ks[i]] & new_sums_by_k[new_ks[j]]:
                return False
    
    return True


def _add_to_dp(sums_by_k, candidate):
    """Update DP state to include candidate. Returns new sums_by_k."""
    n = len(sums_by_k) - 1
    new_sums_by_k = [set() for _ in range(n + 2)]
    new_sums_by_k[0].add(0)
    for k in range(1, n + 2):
        new_sums_by_k[k] = set(sums_by_k[k]) if k <= n else set()
        if k - 1 <= n and sums_by_k[k - 1]:
            for s in sums_by_k[k - 1]:
                new_sums_by_k[k].add(s + candidate)
    return new_sums_by_k


def _greedy_build(candidates, deadline):
    """Greedily build a valid set from candidates list, maintaining DP state."""
    current = []
    sums_by_k = [set()]
    sums_by_k[0].add(0)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        if _check_add_valid(sums_by_k, x):
            current.append(x)
            sums_by_k = _add_to_dp(sums_by_k, x)
    return current, sums_by_k


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Deterministic ascending
    candidates = list(range(1, 257))
    current, _ = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Deterministic descending
    candidates = list(range(256, 0, -1))
    current, _ = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Random restarts with various biases
    random.seed(42)
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Various randomization strategies
        if attempt % 4 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif attempt % 4 == 1:
            # Bias toward smaller values
            candidates = list(range(1, 129))
            random.shuffle(candidates)
            candidates += list(range(129, 257))
            random.shuffle(candidates)
        elif attempt % 4 == 2:
            # Mix of small and large
            small = list(range(1, 65))
            large = list(range(192, 257))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
        else:
            # Power-of-2 biased (helps with cardinality separation)
            p2 = [2**i for i in range(1, 9)]  # 2,4,8,...,128
            others = [x for x in range(1, 257) if x not in p2]
            random.shuffle(others)
            candidates = p2 + others
            random.shuffle(candidates)
        
        current, _ = _greedy_build(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # If we've found a decent set, try to extend it
        if best_size >= 8 and best_size < 60:
            best_dp = _build_dp_state(best_set)
            remaining = [x for x in range(1, 257) if x not in best_set]
            random.shuffle(remaining)
            improved = best_set[:]
            improved_dp = [set(s) for s in best_dp]
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(improved) >= 60:
                    break
                if _check_add_valid(improved_dp, x):
                    improved.append(x)
                    improved_dp = _add_to_dp(improved_dp, x)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    return list(best_set)
# EVOLVE-BLOCK-END
