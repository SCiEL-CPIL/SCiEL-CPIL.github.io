"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0

def _precompute_bad_triples():
    """
    Precompute a structure to quickly check if adding x to S creates a bad triple.
    bad_with[a][b] is a set of c such that (a,b,c) is a bad triple.
    """
    bad_with = [None] * 129
    for i in range(129):
        bad_with[i] = [None] * 129
        for j in range(129):
            bad_with[i][j] = set()
            
    for a in range(1, 129):
        for b in range(a + 1, 129):
            g_ab = math.gcd(a, b)
            for c in range(b + 1, 129):
                g_ac = math.gcd(a, c)
                if g_ac != g_ab:
                    continue
                g_bc = math.gcd(b, c)
                if g_bc == g_ab:
                    bad_with[a][b].add(c)
                    bad_with[a][c].add(b)
                    bad_with[b][c].add(a)
                    bad_with[b][a].add(c)
                    bad_with[c][a].add(b)
                    bad_with[c][b].add(a)
                    
    return bad_with

def _count_conflicts(bad_with):
    """Count how many bad triples each vertex is part of."""
    counts = [0] * 129
    for a in range(1, 129):
        for b in range(a + 1, 129):
            for c in bad_with[a][b]:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
    return counts

def _greedy_add(order, bad_with):
    """
    Greedily add elements from order to the set, maintaining validity.
    """
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        if (time.time() - _START_TIME) > _DEADLINE:
            break
            
        can_add = True
        # Check if x forms a bad triple with any pair in S
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if x in bad_with[a][b]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.append(x)
            S_set.add(x)
            
    return S

def construct():
    """
    Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal.
    """
    random.seed(42)
    
    # 1. Precompute conflict structure
    bad_with = _precompute_bad_triples()
    counts = _count_conflicts(bad_with)
    
    # 2. Initial Greedy with structured order (low conflict first)
    order_struct = sorted(range(1, 129), key=lambda x: counts[x])
    S_best = _greedy_add(order_struct, bad_with)
    best_size = len(S_best)
    
    # 3. Randomized Restart
    # Each greedy pass is fast (~0.05s), so we can do many.
    while (time.time() - _START_TIME) < _DEADLINE - 2.0:
        order_rand = list(range(1, 129))
        random.shuffle(order_rand)
        S_rand = _greedy_add(order_rand, bad_with)
        if len(S_rand) > best_size:
            S_best = S_rand
            best_size = len(S_rand)
            
    return S_best
# EVOLVE-BLOCK-END
