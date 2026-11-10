"""Optimized solver for no-dividing-3-sum task using precomputed conflict pairs and fast checks."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

_N = 192
_MAX_ALLOWED = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin

def _elapsed():
    return time.time() - _START_TIME

# Precompute conflict structure:
# For each x in 1..192, and for each pair (a,b) with a<b from 1..192,
# we want to know if there exists c (c != a, c != b) such that (a+b+c) % x == 0.
# But that's too large. Instead, we precompute for each x a set of "forbidden pairs" 
# in the sense: if we have selected set S, and we want to add x, we need no triple from S
# summing to 0 mod x. But that's dynamic.
#
# Better approach: Precompute for each x, a list of "conflict pairs" (a,b) such that
# there exists some c (c != a, c != b) in [1,192] with (a+b+c) % x == 0.
# Actually, that's almost always true for small x.
#
# Let's think differently. For the validity check of adding x to set S:
# We need: no 3 distinct elements a,b,c in S such that (a+b+c) % x == 0.
# This is equivalent to: for each pair (a,b) in S, the residue (-(a+b)) % x must not
# be present in S \ {a,b}.
#
# For swapping: similar but with S minus the removed element.
#
# Let's precompute for each x, a mapping from residue r to the count of numbers in [1,192]
# with that residue mod x. But we need per-set counts.
#
# Key insight: Precompute for each x, a bitset or array of residues for all numbers 1..192.
# Then for a given set S, we can quickly compute residue counts.

# Precompute residues for each x
_residues = {}
for x in range(1, _N + 1):
    _residues[x] = [v % x for v in range(1, _N + 1)]

def _is_valid_addition(S_set, x):
    """Check if adding x to valid set S_set keeps it valid."""
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    if n >= 3:
        res_count = defaultdict(int)
        for v in S_list:
            res_count[v % x] += 1
        
        for i in range(n):
            a = S_list[i]
            a_res = a % x
            for j in range(i + 1, n):
                b = S_list[j]
                target = (-(a + b)) % x
                cnt = res_count.get(target, 0)
                if a_res == target:
                    cnt -= 1
                if b % x == target:
                    cnt -= 1
                if cnt > 0:
                    return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for a in S_list:
        others = [v for v in S_list if v != a]
        m = len(others)
        if m < 2:
            continue
        
        res_count = defaultdict(int)
        for v in others:
            res_count[v % a] += 1
        
        target = (-x) % a
        for i in range(m):
            b = others[i]
            b_res = b % a
            needed = (target - b_res) % a
            cnt = res_count.get(needed, 0)
            if b_res == needed:
                cnt -= 1
            if cnt > 0:
                return False
    
    return True

def _is_valid_swap(S_set, remove_val, add_val):
    """Check if swapping remove_val with add_val in valid set S_set keeps it valid."""
    if add_val in S_set:
        return False
    if remove_val not in S_set:
        return False
    
    S_without = S_set - {remove_val}
    S_list = list(S_without)
    n = len(S_list)
    
    # Check 1: add_val does not divide sum of any 3 in S_without
    if n >= 3:
        res_count = defaultdict(int)
        for v in S_list:
            res_count[v % add_val] += 1
        
        for i in range(n):
            a = S_list[i]
            a_res = a % add_val
            for j in range(i + 1, n):
                b = S_list[j]
                target = (-(a + b)) % add_val
                cnt = res_count.get(target, 0)
                if a_res == target:
                    cnt -= 1
                if b % add_val == target:
                    cnt -= 1
                if cnt > 0:
                    return False
    
    # Check 2: for each a in S_without, a does not divide sum of add_val + 2 others
    for a in S_list:
        others = [v for v in S_list if v != a]
        m = len(others)
        if m < 2:
            continue
        
        res_count = defaultdict(int)
        for v in others:
            res_count[v % a] += 1
        
        target = (-add_val) % a
        for i in range(m):
            b = others[i]
            b_res = b % a
            needed = (target - b_res) % a
            cnt = res_count.get(needed, 0)
            if b_res == needed:
                cnt -= 1
            if cnt > 0:
                return False
    
    return True

def _greedy(order=None):
    """Greedy construction with given order."""
    selected = []
    S_set = set()
    if order is None:
        order = list(range(1, _N + 1))
    
    for x in order:
        if _elapsed() > 50.0:
            break
        if _is_valid_addition(S_set, x):
            selected.append(x)
            S_set.add(x)
    
    return selected

def _local_search(S, time_budget=5.0):
    """Local search with swaps and insertions."""
    S = list(S)
    S_set = set(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and _elapsed() < 50.0:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX_ALLOWED:
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if _elapsed() > 50.0:
                    return best
                if _is_valid_addition(S_set, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            
            for a in to_remove[:50]:
                if _elapsed() > 50.0:
                    return best
                for x in candidates[:50]:
                    if _elapsed() > 50.0:
                        return best
                    if _is_valid_swap(S_set, a, x):
                        S.remove(a)
                        S.append(x)
                        S_set.remove(a)
                        S_set.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        if not improved:
            # Perturb: remove 1-3 random elements
            if len(S) > 3:
                k = random.randint(1, 3)
                to_remove = random.sample(S, k)
                for v in to_remove:
                    S.remove(v)
                    S_set.remove(v)
    
    return best

def _multi_start_greedy(num_starts=15, time_budget=15.0):
    """Multiple greedy starts with different orderings."""
    best = []
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() - start_time > time_budget or _elapsed() > 45.0:
            break
        
        # Different ordering strategies
        if i % 6 == 0:
            order = list(range(_N, 0, -1))  # Descending
        elif i % 6 == 1:
            order = list(range(1, _N + 1))  # Ascending
        elif i % 6 == 2:
            order = list(range(1, _N + 1))
            random.shuffle(order)
        elif i % 6 == 3:
            # Bias toward larger numbers
            order = list(range(_N // 2, _N + 1)) + list(range(1, _N // 2 + 1))
            random.shuffle(order[:_N // 2])
        elif i % 6 == 4:
            # Weighted random: prefer mid-to-large range
            order = list(range(64, _N + 1)) + list(range(1, 64))
            random.shuffle(order[:len(order)//2])
        else:
            # Random with slight bias
            order = list(range(1, _N + 1))
            random.shuffle(order)
        
        S = _greedy(order)
        if len(S) > len(best):
            best = S
    
    return best

def construct():
    """Main entry point."""
    # Phase 1: Multi-start greedy
    best = _multi_start_greedy(num_starts=15, time_budget=12.0)
    
    # Phase 2: Local search
    if _elapsed() < 45.0:
        S2 = _local_search(best, time_budget=15.0)
        if len(S2) > len(best):
            best = S2
    
    # Phase 3: Another local search pass if time permits
    if _elapsed() < 38.0:
        S3 = _local_search(best, time_budget=12.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 4: Final polish - try to add any remaining valid elements systematically
    if _elapsed() < 30.0:
        S_set = set(best)
        all_vals = list(range(1, _N + 1))
        random.shuffle(all_vals)
        for x in all_vals:
            if _elapsed() > 30.0:
                break
            if x not in S_set and _is_valid_addition(S_set, x):
                best.append(x)
                S_set.add(x)
    
    # Phase 5: One more local search if we gained elements
    if _elapsed() < 20.0 and len(best) > 40:
        S4 = _local_search(best, time_budget=15.0)
        if len(S4) > len(best):
            best = S4
    
    return best
# EVOLVE-BLOCK-END
