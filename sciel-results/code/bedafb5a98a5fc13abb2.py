import random
import time
from math import gcd

def construct():
    N = 128
    # Precompute conflicts
    # conflicts[x] is a set of tuples (u, v) with u < v
    # such that {x, u, v} is a "bad" triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # {a, b, c} is a bad triple
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))
    
    def can_add(x, S):
        # Check if x can be added to S
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    best_set = set()
    best_size = 0
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Use a fixed seed for reproducibility in debugging, but random for actual run?
    # The evaluator runs it once. Randomness is good for finding global optimum.
    
    while time.time() < deadline:
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy Phase
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Improvement Phase
        # Try to add any remaining elements
        # We iterate through all numbers not in S
        # If we add one, it might block others, so we just do one pass.
        # The order of checking in improvement phase doesn't strictly matter for validity,
        # but might affect which maximal set we find.
        # Since we restart, we will find various maximal sets.
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return list(best_set)
