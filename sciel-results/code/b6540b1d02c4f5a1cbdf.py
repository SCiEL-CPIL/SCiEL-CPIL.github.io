import random
from itertools import combinations
from math import gcd
import time

# Precompute bad triples
def compute_bad_triples():
    bad = set()
    n = 128
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, n + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad.add((a, b, c))
    return bad

BAD_TRIPLES = compute_bad_triples()

# Precompute GCD table for faster lookups
N = 128
GCD_TABLE = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        GCD_TABLE[i][j] = g
        GCD_TABLE[j][i] = g

def can_add(x, current_set):
    """
    Check if x can be added to current_set without forming a bad triple.
    current_set is a set of ints.
    """
    # Iterate over all pairs in current_set
    for u, v in combinations(current_set, 2):
        triple = tuple(sorted((x, u, v)))
        if triple in BAD_TRIPLES:
            return False
    return True

def greedy_build(order):
    """
    Build a set by iterating through 'order' and adding if possible.
    """
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s, deadline):
    """
    Try to improve set s by adding single elements or swapping.
    Returns the improved set.
    """
    improved = True
    while improved:
        improved = False
        
        # Time check
        if time.time() > deadline:
            break
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, s):
                s.add(x)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to swap: remove one, add some
        # We iterate through elements to remove, try to fill the gap
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                break
                
            s_without_u = s - {u}
            new_s = set(s_without_u)
            
            # Try to add elements from the complement of the original s
            # Note: u is in s, so it's not in the complement.
            # We want to see if we can add > 0 elements to new_s
            # such that len(new_s) > len(s).
            # Since we removed 1, we need to add at least 2 to improve size,
            # OR add 1 element that is NOT u (which is impossible since u is not in complement)
            # Wait, if we add 1 element from complement, new_s has size len(s)-1+1 = len(s).
            # So we need to add at least 2 elements to strictly improve.
            # Or we can just let the greedy run and check size at the end.
            
            addable = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(addable)
            
            for x in addable:
                if can_add(x, new_s):
                    new_s.add(x)
            
            if len(new_s) > len(s):
                s = new_s
                improved = True
                break
                
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, N + 1))
    s = greedy_build(initial_order)
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # Random restarts
    while time.time() < deadline - 5:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline - 5)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    # Final local search on best set if time permits
    if time.time() < deadline:
        s = best_set.copy()
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()

    if not best_set:
        return [1]
        
    return list(best_set)
