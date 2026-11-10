import time
import random
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _can_form_target(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals.
    vals is a list of integers.
    Uses meet-in-the-middle for efficiency.
    """
    if target == 0:
        return True
    if not vals:
        return False
    if target > sum(vals):
        return False
    
    n = len(vals)
    half = n // 2
    left = vals[:half]
    right = vals[half:]
    
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
        
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
        
    for s in left_sums:
        if target - s in right_sums:
            return True
    return False

def _is_valid_set(S):
    """
    Check if no element's reciprocal is the sum of reciprocals of a subset of the others.
    S is a list of distinct integers.
    """
    n = len(S)
    if n <= 1:
        return True
        
    for i, x in enumerate(S):
        others = S[:i] + S[i+1:]
        if not others:
            continue
            
        try:
            L = reduce(_lcm, others, 1)
        except ZeroDivisionError:
            continue
            
        if L % x != 0:
            continue
            
        target = L // x
        if target == 0:
            continue
            
        vals = [L // y for y in others]
        
        if _can_form_target(target, vals):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin
    
    # Elements in descending order
    elems = list(range(24, 0, -1))
    
    best_set = []
    best_size = 0
    
    # Initial greedy to get a good lower bound for pruning
    # Try adding elements in descending order if valid
    greedy_set = []
    for x in elems:
        if _is_valid_set(greedy_set + [x]):
            greedy_set.append(x)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
        
    # Also try ascending order for greedy
    greedy_set_asc = []
    for x in reversed(elems):
        if _is_valid_set(greedy_set_asc + [x]):
            greedy_set_asc.append(x)
    if len(greedy_set_asc) > best_size:
        best_set = greedy_set_asc
        best_size = len(greedy_set_asc)

    def search(selected, idx):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Upper bound pruning
        if len(selected) + (len(elems) - idx) <= best_size:
            return
            
        if idx == len(elems):
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
            
        x = elems[idx]
        
        # Option 1: Include x
        # Check validity of selected + [x]
        if _is_valid_set(selected + [x]):
            search(selected + [x], idx + 1)
            
        # Option 2: Exclude x
        search(selected, idx + 1)
        
    search([], 0)
    
    # If time remains, try random restarts to see if we missed something? 
    # B&B should be exhaustive if it finishes. 
    # If it times out, it might not be exhaustive.
    # But with N=24 and strong pruning, it should be fine.
    
    if not best_set:
        return [24]
        
    return best_set
