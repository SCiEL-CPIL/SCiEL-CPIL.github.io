import random
from itertools import combinations
from math import gcd
import time

# Precompute conflict graph: for each element x, store set of elements y such that
# there exists z where {x,y,z} forms a bad triple
def compute_conflicts():
    conflicts = {i: set() for i in range(1, 129)}
    n = 128
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, n + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # a,b,c form a bad triple
                    conflicts[a].add(b)
                    conflicts[a].add(c)
                    conflicts[b].add(a)
                    conflicts[b].add(c)
                    conflicts[c].add(a)
                    conflicts[c].add(b)
    return conflicts

CONFLICTS = compute_conflicts()

def can_add_fast(x, current_set):
    """
    Check if x can be added to current_set without forming a bad triple.
    Uses precomputed conflicts for faster lookup.
    """
    # x conflicts with y if there exists z in current_set such that {x,y,z} is bad
    # This is equivalent to: y is in conflicts[x] AND there exists z in current_set 
    # such that {x,y,z} is a bad triple.
    # Actually, the conflict graph approach needs refinement.
    # Let's use a different approach: precompute for each pair (u,v), which x values 
    # would form a bad triple with u and v.
    pass

# Precompute: for each pair (u,v), the set of x such that {u,v,x} is a bad triple
def compute_pair_conflicts():
    pair_conflicts = {}
    n = 128
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            conflicting_x = set()
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    conflicting_x.add(c)
            if conflicting_x:
                pair_conflicts[(a, b)] = conflicting_x
                pair_conflicts[(b, a)] = conflicting_x
    return pair_conflicts

PAIR_CONFLICTS = compute_pair_conflicts()

def can_add(x, current_set):
    """
    Check if x can be added to current_set without forming a bad triple.
    """
    for u, v in combinations(current_set, 2):
        # Check if x conflicts with pair (u,v)
        key = (u, v) if u < v else (v, u)
        if key in PAIR_CONFLICTS and x in PAIR_CONFLICTS[key]:
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
    while improved and time.time() < deadline:
        improved = False
        # 1. Try to add any single element
        candidates = [x for x in range(1, 129) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, s):
                s.add(x)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to swap: remove one, add some
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            s_without_u = s - {u}
            new_s = set(s_without_u)
            
            addable = [x for x in range(1, 129) if x not in s]
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
    deadline = start_time + 55.0  # 5 second margin
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, 129))
    s = greedy_build(initial_order)
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # Random restarts
    while time.time() < deadline:
        random_order = list(range(1, 129))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    return list(best_set)
