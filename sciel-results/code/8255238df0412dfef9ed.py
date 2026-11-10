"""Optimized solver for no-dividing-3-sum task using fast validity checks and multi-start search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

_N = 192
_MAX_ALLOWED = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin

def _elapsed():
    return time.time() - _START_TIME

def _is_valid(S):
    """Check if set S is valid: no element divides sum of any 3 distinct others."""
    S_list = list(S)
    n = len(S_list)
    for i, x in enumerate(S_list):
        # Check if x divides sum of any 3 distinct others
        others = [v for j, v in enumerate(S_list) if j != i]
        m = len(others)
        if m < 3:
            continue
        # For each pair (b, c) in others, check if there exists d in others 
        # (d != b, d != c) such that (b + c + d) % x == 0
        # i.e., d % x == (-(b+c)) % x
        # Count how many d in others satisfy this, ensuring d != b and d != c
        # Use a counter for residues of others
        res_count = defaultdict(int)
        for v in others:
            res_count[v % x] += 1
        
        for a_idx in range(m):
            b = others[a_idx]
            for c_idx in range(a_idx + 1, m):
                c = others[c_idx]
                target = (-(b + c)) % x
                # How many elements in others have residue target?
                cnt = res_count.get(target, 0)
                # Subtract b and c if they have residue target
                if b % x == target:
                    cnt -= 1
                if c % x == target:
                    cnt -= 1
                if cnt > 0:
                    return False
    return True

def _is_valid_addition(S_set, x):
    """Check if adding x to valid set S_set keeps it valid. Fast version."""
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    # Need: no 3 distinct elements a,b,c in S such that (a+b+c) % x == 0
    if n >= 3:
        # Count residues
        res_count = defaultdict(int)
        for v in S_list:
            res_count[v % x] += 1
        
        # For each pair (a, b), check if there's a c != a, c != b with c % x == (-(a+b)) % x
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                target = (-(a + b)) % x
                cnt = res_count.get(target, 0)
                if a % x == target:
                    cnt -= 1
                if b % x == target:
                    cnt -= 1
                if cnt > 0:
                    return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for a in S_list:
        # Need: no 2 distinct elements b, c in S (b != a, c != a, b != c) 
        # such that (x + b + c) % a == 0
        # i.e., (b + c) % a == (-x) % a
        others = [v for v in S_list if v != a]
        m = len(others)
        if m < 2:
            continue
        
        res_count = defaultdict(int)
        for v in others:
            res_count[v % a] += 1
        
        for i in range(m):
            b = others[i]
            for j in range(i + 1, m):
                c = others[j]
                target = (-(x)) % a
                # We need (b + c) % a == target
                # But we're iterating pairs, so just check directly
                if (b + c) % a == target:
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
            for j in range(i + 1, n):
                b = S_list[j]
                target = (-(a + b)) % add_val
                cnt = res_count.get(target, 0)
                if a % add_val == target:
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
        
        for i in range(m):
            b = others[i]
            for j in range(i + 1, m):
                c = others[j]
                if (add_val + b + c) % a == 0:
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
            for x in candidates[:50]:
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
            
            for a in to_remove[:30]:
                if _elapsed() > 50.0:
                    return best
                for x in candidates[:30]:
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
            # Perturb: remove 1-2 random elements
            if len(S) > 3:
                k = random.randint(1, 2)
                to_remove = random.sample(S, k)
                for v in to_remove:
                    S.remove(v)
                    S_set.remove(v)
    
    return best

def _multi_start_greedy(num_starts=10, time_budget=15.0):
    """Multiple greedy starts with different orderings."""
    best = []
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() - start_time > time_budget or _elapsed() > 45.0:
            break
        
        # Different ordering strategies
        if i % 4 == 0:
            order = list(range(_N, 0, -1))  # Descending
        elif i % 4 == 1:
            order = list(range(1, _N + 1))  # Ascending
        elif i % 4 == 2:
            order = list(range(1, _N + 1))
            random.shuffle(order)
        else:
            # Bias toward larger numbers
            order = list(range(_N // 2, _N + 1)) + list(range(1, _N // 2 + 1))
            random.shuffle(order[:_N // 2])
        
        S = _greedy(order)
        if len(S) > len(best):
            best = S
    
    return best

def construct():
    """Main entry point."""
    # Phase 1: Multi-start greedy
    best = _multi_start_greedy(num_starts=8, time_budget=10.0)
    
    # Phase 2: Local search
    if _elapsed() < 45.0:
        S2 = _local_search(best, time_budget=15.0)
        if len(S2) > len(best):
            best = S2
    
    # Phase 3: Another local search pass if time permits
    if _elapsed() < 40.0:
        S3 = _local_search(best, time_budget=10.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 4: Final polish - try to add any remaining valid elements
    if _elapsed() < 35.0:
        S_set = set(best)
        all_vals = list(range(1, _N + 1))
        random.shuffle(all_vals)
        for x in all_vals:
            if _elapsed() > 35.0:
                break
            if x not in S_set and _is_valid_addition(S_set, x):
                best.append(x)
                S_set.add(x)
    
    return best
# EVOLVE-BLOCK-END
