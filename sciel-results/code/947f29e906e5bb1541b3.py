import math
import time
import random

def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for the 90s limit
    
    # Precompute conflict information
    # For each x in [2, 256], precompute for each y in [2, 256] whether x | y*z for some z
    # Actually, let's precompute the "conflict" relation more carefully.
    
    # For a set S, the condition is: for all distinct a, b, c in S, a ∤ b*c
    # This is a 3-uniform hypergraph constraint.
    
    # Let's use a different approach: greedy construction with smart ordering
    # and local search.
    
    # Key insight: if we sort numbers by their "divisibility power" (how many
    # products they divide), we can greedily add the safest numbers first.
    
    # For each x, compute how many pairs (y, z) with y,z in [2,256], y!=z, 
    # satisfy x | y*z. This is a measure of how "conflict-prone" x is.
    
    def conflict_score(x, S_set):
        """Count how many pairs (y,z) in S_set, y!=z, y!=x, z!=x, satisfy x | y*z"""
        count = 0
        for y in S_set:
            if y == x:
                continue
            for z in S_set:
                if z == x or z == y:
                    continue
                if (y * z) % x == 0:
                    count += 1
        return count
    
    # Phase 1: Greedy construction
    # Start with the largest numbers (they're least likely to divide products)
    S = []
    S_set = set()
    
    # Try adding numbers from 256 down to 2
    for x in range(256, 1, -1):
        if time.time() > deadline - 5:
            break
            
        # Check if adding x to S maintains the constraint
        # Need to check: for all distinct a, b in S (a,b != x), x ∤ a*b
        # And for all distinct a in S (a != x), a ∤ x*b and a ∤ b*x for all b in S, b != a, b != x
        # Actually, the condition is symmetric in the sense that we need to check all triples.
        
        valid = True
        S_list = list(S_set)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                # Triple (x, a, b): x | a*b?
                if (a * b) % x == 0:
                    valid = False
                    break
                # Triple (a, x, b): a | x*b?
                if (x * b) % a == 0:
                    valid = False
                    break
                # Triple (b, x, a): b | x*a?
                if (x * a) % b == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
    
    # Phase 2: Local search - try to swap elements
    # Try removing one element and adding another
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        S_list = list(S_set)
        
        for i, a in enumerate(S_list):
            if time.time() > deadline - 10:
                break
            
            # Try removing a
            S_temp = S_list[:i] + S_list[i+1:]
            S_temp_set = set(S_temp)
            
            # Try adding each candidate not in S_temp
            for x in range(256, 1, -1):
                if x in S_temp_set:
                    continue
                
                # Check if S_temp U {x} is valid
                valid = True
                for j, b in enumerate(S_temp):
                    for k, c in enumerate(S_temp):
                        if j == k:
                            continue
                        if (b * c) % x == 0:
                            valid = False
                            break
                        if (x * c) % b == 0:
                            valid = False
                            break
                        if (x * b) % c == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S = S_temp + [x]
                    S_set = set(S)
                    improved = True
                    break
        
        if improved:
            break
    
    # Phase 3: Try to add more elements that were skipped
    for x in range(256, 1, -1):
        if time.time() > deadline - 5:
            break
        if x in S_set:
            continue
        
        valid = True
        S_list = list(S_set)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                if (a * b) % x == 0:
                    valid = False
                    break
                if (x * b) % a == 0:
                    valid = False
                    break
                if (x * a) % b == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
    
    return S
