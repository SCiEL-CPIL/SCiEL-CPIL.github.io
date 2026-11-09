import time
import random
from math import lcm
from functools import reduce
from itertools import combinations

# Universe is [1, 24]
UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

def subset_sum_check(target, weights):
    """Check if target can be formed as a sum of a subset of weights."""
    if target < 0:
        return False
    if target == 0:
        return True
    if not weights:
        return False
    if sum(weights) < target:
        return False
    if min(weights) > target:
        return False
    
    # Use a set-based subset sum with pruning
    # Sort weights descending for better pruning
    ws = sorted(weights, reverse=True)
    
    # Try all subsets using bit manipulation if len <= 20, else use DP set
    n = len(ws)
    if n <= 20:
        # Meet in the middle or brute force
        # Brute force with early termination
        def dfs(idx, remaining):
            if remaining == 0:
                return True
            if idx == n:
                return False
            if remaining < 0:
                return False
            # Prune: if remaining > sum of rest
            if remaining > sum(ws[idx:]):
                return False
            # Try including ws[idx]
            if dfs(idx + 1, remaining - ws[idx]):
                return True
            # Try not including
            if dfs(idx + 1, remaining):
                return True
            return False
        return dfs(0, target)
    else:
        # Use a set of achievable sums, but cap size
        achievable = {0}
        for w in ws:
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            achievable.update(new_sums)
            if target in achievable:
                return True
            # Cap to prevent explosion
            if len(achievable) > 100000:
                break
        return target in achievable

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = L // x
        others = [L // y for y in S if y != x]
        if subset_sum_check(target, others):
            return False
    return True

def construct():
    deadline = time.time() + 85  # 85s to leave safety margin
    
    best = list(range(13, 25))  # Fallback: size 12
    if not is_valid(best):
        best = [24]
    
    # Try greedy construction with different orderings
    random.seed(42)
    
    # Strategy 1: Greedy add in various orders
    orders = [
        sorted(UNIVERSE, reverse=True),  # Large first
        sorted(UNIVERSE),                # Small first
        list(reversed(UNIVERSE)),
    ]
    # Add some random permutations
    for _ in range(5):
        perm = UNIVERSE[:]
        random.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        if len(S) > len(best):
            best = S
    
    # Strategy 2: Local search - try to add elements to best
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    improved = True
    
    # Strategy 3: Try removing one element and adding two different ones
    for _ in range(20):
        if time.time() > deadline:
            break
        S = best[:]
        random.shuffle(S)
        # Try removing one element
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_no_i = S[:i] + S[i+1:]
            # Try adding elements not in S_no_i
            for x in UNIVERSE:
                if time.time() > deadline:
                    break
                if x not in S_no_i:
                    candidate = S_no_i + [x]
                    if len(candidate) > len(best) and is_valid(candidate):
                        best = candidate
                        S = best[:]
                        break
            if len(best) > len(S_no_i):
                break
    
    return best
