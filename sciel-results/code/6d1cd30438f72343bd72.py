import time
import random
from fractions import Fraction

def _subset_sum_exists(values, target):
    if not values:
        return False
    values = sorted(values, reverse=True)
    n = len(values)
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
    if suffix[0] < target:
        return False
    memo = {}
    def dfs(idx, current):
        if current == target:
            return True
        if idx >= n or current > target:
            return False
        if current + suffix[idx] < target:
            return False
        key = (idx, current)
        if key in memo:
            return memo[key]
        # Include
        if dfs(idx + 1, current + values[idx]):
            memo[key] = True
            return True
        # Exclude
        if dfs(idx + 1, current):
            memo[key] = True
            return True
        memo[key] = False
        return False
    return dfs(0, Fraction(0))

def _is_valid(S):
    S = list(set(S))
    for x in S:
        target = Fraction(1, x)
        others = [Fraction(1, y) for y in S if y != x]
        # Quick check: if sum of all others < target, no subset can reach target
        total = sum(others)
        if total < target:
            continue
        if _subset_sum_exists(others, target):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_set = set(range(13, 25))
    best_size = 12
    
    def try_order(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            candidate_set = S_set | {x}
            if _is_valid(candidate_set):
                S.append(x)
                S_set.add(x)
        # Greedy completion
        excluded = [x for x in range(1, 25) if x not in S_set]
        changed = True
        while changed:
            changed = False
            for x in excluded[:]:
                if x in S_set:
                    continue
                candidate_set = S_set | {x}
                if _is_valid(candidate_set):
                    S_set.add(x)
                    excluded.remove(x)
                    changed = True
        return S_set
    
    # Deterministic orders
    orders = [
        list(range(24, 0, -1)),
        list(range(13, 25)) + list(range(12, 0, -1)),
        list(range(13, 25)) + list(range(1, 13)),
        list(range(1, 25)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = try_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Random orders
    rng = random.Random(12345)
    while time.time() < deadline:
        order = list(range(1, 25))
        rng.shuffle(order)
        S = try_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return sorted(best_set)
