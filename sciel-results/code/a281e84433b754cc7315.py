"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute bad_pairs for each x
    # bad_pairs[x] is a set of frozensets {u, v} such that {u, v, x} is a bad triple
    # i.e., gcd(u,v) == gcd(u,x) == gcd(v,x)
    bad_pairs = [set() for _ in range(N + 1)]
    
    # Optimization: Precompute gcd matrix
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    for x in range(1, N + 1):
        for u in range(1, N + 1):
            if u == x:
                continue
            gux = gcd_matrix[u][x]
            for v in range(u + 1, N + 1):
                if v == x:
                    continue
                guv = gcd_matrix[u][v]
                gvx = gcd_matrix[v][x]
                if guv == gux == gvx:
                    bad_pairs[x].add(frozenset((u, v)))
                    
    # Greedy construction with restarts
    best_set = set()
    max_size = 0
    
    # Try a few specific orderings first
    # 1. Random
    # 2. Sorted by len(bad_pairs[x]) ascending (least constrained first)
    # 3. Sorted by len(bad_pairs[x]) descending (most constrained first)
    
    orderings = []
    # Generate a few random permutations
    for _ in range(10):
        perm = numbers[:]
        random.shuffle(perm)
        orderings.append(perm)
        
    # Sorted by constraint count
    by_constraint_asc = sorted(numbers, key=lambda x: len(bad_pairs[x]))
    by_constraint_desc = sorted(numbers, key=lambda x: -len(bad_pairs[x]))
    orderings.append(by_constraint_asc)
    orderings.append(by_constraint_desc)
    
    def run_greedy(order):
        S = set()
        pairs_in_S = set()
        
        for x in order:
            # Check if x conflicts with any pair in S
            # Conflict if there exists u,v in S such that {u,v} is in bad_pairs[x]
            # Equivalent: any pair in S is in bad_pairs[x]
            if not (pairs_in_S & bad_pairs[x]):
                # Add x
                S.add(x)
                # Update pairs_in_S
                for u in S:
                    if u != x:
                        pairs_in_S.add(frozenset((u, x)))
        return S

    # Run with predefined orderings
    for order in orderings:
        if time.time() > deadline:
            break
        S = run_greedy(order)
        if len(S) > max_size:
            max_size = len(S)
            best_set = S

    # Run random restarts until deadline
    restarts = 0
    while time.time() < deadline:
        perm = numbers[:]
        random.shuffle(perm)
        S = run_greedy(perm)
        if len(S) > max_size:
            max_size = len(S)
            best_set = S
        restarts += 1
        # If we've done a lot of restarts without improvement, we might be stuck, 
        # but given the small N, we can keep going or try local search if time permits.
        # For now, just keep restarting.
        
    return list(best_set)

# EVOLVE-BLOCK-END
