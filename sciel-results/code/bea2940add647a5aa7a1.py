import random
from itertools import combinations
from math import lcm

# Precompute bad triples
# conflicts[x] is a set of tuples (a, b) such that {a, b, x} is a bad triple
conflicts = {i: set() for i in range(1, 129)}

def is_bad_triple(a, b, c):
    if a == b or b == c or a == c:
        return False
    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    return l_ab == l_ac == l_bc

# Precompute conflicts
for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            if is_bad_triple(a, b, c):
                conflicts[a].add((b, c))
                conflicts[b].add((a, c))
                conflicts[c].add((a, b))

def construct():
    best_set = set()
    
    # Strategy: Randomized Greedy Independent Set
    # 1. Randomize order of elements
    # 2. Iterate and add element if it doesn't conflict with existing set
    # 3. Repeat many times to find the largest such set
    
    # 2000 iterations is fast: 2000 * 128 * (avg conflicts ~10) = ~2.5M ops
    for _ in range(2000):
        S = set()
        order = list(range(1, 129))
        random.shuffle(order)
        
        for x in order:
            # Check if x can be added to S
            # x conflicts if there exists (a, b) in conflicts[x] such that a in S and b in S
            can_add = True
            for a, b in conflicts[x]:
                if a in S and b in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
    return list(best_set)
