"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    Returns a list of tuples (a,b,c) and a dict mapping each element to 
    the set of pairs that form a bad triple with it.
    """
    n = max_n
    # Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
                
    bad_triples = []
    # bad_with[x] will store a set of tuples (a,b) such that {x,a,b} is bad
    # We store pairs (min(a,b), max(a,b)) to keep it consistent
    bad_with = {i: set() for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    # Add to bad_with for each element
                    bad_with[a].add((b, c))
                    bad_with[b].add((a, c))
                    bad_with[c].add((a, b))
                    
    return bad_triples, bad_with

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    max_n = 128
    bad_triples, bad_with = _precompute_bad_triples(max_n)
    
    # If no bad triples, return all
    if not bad_triples:
        return list(range(1, max_n + 1))
        
    best_set = set()
    
    # Min-conflict greedy strategy
    # We will try multiple times with slight randomization in tie-breaking
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        remaining = set(range(1, max_n + 1))
        selected = []
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Calculate cost for each candidate in remaining
            # Cost = number of pairs (a,b) in bad_with[x] such that a,b in remaining
            min_cost = float('inf')
            best_candidate = None
            candidates_with_min_cost = []
            
            for x in remaining:
                cost = 0
                # bad_with[x] contains pairs (a,b)
                for a, b in bad_with[x]:
                    if a in remaining and b in remaining:
                        cost += 1
                        if cost >= min_cost:
                            break
                if cost < min_cost:
                    min_cost = cost
                    best_candidate = x
                    candidates_with_min_cost = [x]
                elif cost == min_cost:
                    candidates_with_min_cost.append(x)
                    
            # Pick randomly from candidates with min cost to add diversity
            if candidates_with_min_cost:
                x = random.choice(candidates_with_min_cost)
            else:
                x = best_candidate
                
            if x is not None:
                selected.append(x)
                remaining.remove(x)
                
        if len(selected) > len(best_set):
            best_set = set(selected)
            
    # Local search improvement: Try to add elements from the complement
    # This is a simple hill-climbing step
    current_set = set(best_set)
    complement = set(range(1, max_n + 1)) - current_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(complement):
            # Check if x can be added
            # x can be added if no bad triple is formed with two existing elements
            # We need to check all pairs in current_set
            # This is O(N^2) per check. N=128. 128^2 * 128 ~ 2M ops.
            # Might be slow if done for all x in loop.
            # Optimization: Pre-check using bad_with?
            # x is valid if for all (a,b) in bad_with[x], NOT (a in current_set AND b in current_set)
            valid = True
            for a, b in bad_with[x]:
                if a in current_set and b in current_set:
                    valid = False
                    break
            if valid:
                current_set.add(x)
                complement.remove(x)
                improved = True
                # Restart loop as set changed
                break
                
    return list(current_set)
# EVOLVE-BLOCK-END
