import time
import random
from math import lcm

_START = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START)

# Precompute LCM of 1..24
_L = 1
for i in range(2, 25):
    _L = lcm(_L, i)

_WEIGHTS = {k: _L // k for k in range(1, 25)}
_ALL_ELEMENTS = list(range(1, 25))

def _is_valid(S):
    """
    Check if S is reciprocal-subsum-free.
    Uses meet-in-the-middle for subset sum check.
    """
    if not S:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    for i, k in enumerate(S_list):
        target = _WEIGHTS[k]
        others = [_WEIGHTS[j] for j in S_list if j != k]
        
        if not others:
            continue
            
        # Split others into two halves
        mid = len(others) // 2
        A = others[:mid]
        B = others[mid:]
        
        # Generate subset sums for A
        sums_A = {0}
        for w in A:
            new_sums = set()
            for s in sums_A:
                ns = s + w
                if ns <= target: # Prune
                    new_sums.add(ns)
            sums_A |= new_sums
            
        # If target is in sums_A, it's a sum of a subset of A (which is subset of others)
        if target in sums_A:
            return False
            
        # Generate subset sums for B
        sums_B = {0}
        for w in B:
            new_sums = set()
            for s in sums_B:
                ns = s + w
                if ns <= target: # Prune
                    new_sums.add(ns)
            sums_B |= new_sums
            
        # Check if any a in sums_A and b in sums_B sum to target
        # Sort sums_B for binary search or use set intersection
        # Since sizes are small (2^10 = 1024), we can just iterate
        # But set difference/intersection is faster in Python?
        # Let's just iterate sums_A and check if target - a is in sums_B
        # To speed up, convert sums_B to a set (it already is)
        for a in sums_A:
            if (target - a) in sums_B:
                return False
                
    return True

def _greedy_fill(base_set, order):
    """
    Greedily add elements from 'order' to base_set if valid.
    """
    S = set(base_set)
    for k in order:
        if k in S:
            continue
        if _is_valid(S | {k}):
            S.add(k)
    return S

def _local_search(S):
    """
    Local search to maximize size.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    while _time_remaining() > 2.0:
        improved = False
        
        # 1. Try to add an element
        # Shuffle candidates to avoid bias
        candidates = [k for k in _ALL_ELEMENTS if k not in S]
        random.shuffle(candidates)
        
        added = False
        for k in candidates:
            if _time_remaining() < 2.0:
                break
            if _is_valid(S | {k}):
                S.add(k)
                improved = True
                added = True
                break # Restart search with new S
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            continue # Restart loop to try adding more
        
        # 2. No add possible. Try swaps.
        # Try to swap i in S with j not in S
        # We want to maintain size, but potentially enable future adds.
        # Or, if we are stuck, maybe a swap allows a different configuration.
        
        # Strategy: Try random swaps
        # Limit number of swaps to avoid infinite loops on same state?
        # We can track visited states if needed, but with 24 elements, 2^24 is large.
        # Just try a few random swaps.
        
        swaps_tried = 0
        max_swaps = 50
        swap_made = False
        
        while swaps_tried < max_swaps and _time_remaining() > 2.0:
            swaps_tried += 1
            if not S:
                break
            i = random.choice(list(S))
            candidates_out = [k for k in _ALL_ELEMENTS if k not in S]
            if not candidates_out:
                break
            j = random.choice(candidates_out)
            
            S_new = S - {i} | {j}
            if _is_valid(S_new):
                S = S_new
                swap_made = True
                break # Restart search after a successful swap
                
        if swap_made:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            continue
            
        # 3. If no swap worked, we are stuck in a local optimum.
        # Perturb: Remove a random element to allow different paths?
        # Or just restart with a new random greedy?
        # Let's try a "kick": remove 1-2 random elements.
        if len(S) > 1:
            # Remove 1 random element
            i = random.choice(list(S))
            S.remove(i)
            # This reduces size, but might allow adding 2 different ones later?
            # Or just restart entirely?
            # Given the time limit, restarting with a new greedy is safer.
            break
        else:
            break
            
    return best_S

def construct():
    if _time_remaining() < 1.0:
        return list(range(13, 25))
        
    best_set = set()
    best_size = 0
    
    # Run multiple random restarts
    while _time_remaining() > 5.0:
        # Random order for greedy
        order = _ALL_ELEMENTS[:]
        random.shuffle(order)
        
        # Greedy fill
        S = _greedy_fill(set(), order)
        
        # Local search
        S_local = _local_search(S)
        
        if len(S_local) > best_size:
            best_size = len(S_local)
            best_set = S_local
            
        # If we found a very large set, maybe stop early?
        # Max possible is 24.
        if best_size >= 20: # Just a heuristic threshold
             # Continue a bit more to see if we can hit 21+
             pass
             
    if not best_set:
        return [13]
        
    return list(best_set)

_RESULT = construct()

def construct():
    return _RESULT
