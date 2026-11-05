import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    # Candidates are 2..256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute incompatibility: incompat[x] is a set of y such that lcm(x, y) <= 256
    # Note: x, y are values.
    # We will use a dictionary mapping value to set of incompatible values.
    incompat = {x: set() for x in candidates}
    for i, x in enumerate(candidates):
        for y in candidates[i+1:]:
            # Calculate lcm
            g = gcd(x, y)
            l = x // g * y
            if l <= 256:
                incompat[x].add(y)
                incompat[y].add(x)
                
    # Helper to check if a value v is compatible with all elements in a set S
    # S is a set of values
    def is_compatible_with_set(v, S):
        for w in S:
            if w == v:
                continue
            if w in incompat[v]:
                return False
        return True
        
    # Helper to calculate sum of reciprocals
    def calc_sum(S):
        return sum(1.0 / x for x in S)
        
    # Initial Solution: Greedy by weight (smallest x first)
    S = set()
    # Sort by x ascending (highest weight)
    for x in candidates:
        if is_compatible_with_set(x, S):
            S.add(x)
            
    best_S = set(S)
    best_score = calc_sum(best_S)
    
    # Local Search
    # We will try to find 1-for-1 swaps that improve the score.
    # Move: remove u from S, add v to S.
    # Improvement if 1/v > 1/u => v < u.
    # Condition: v is compatible with S \ {u}.
    
    # To speed up, we can pre-sort candidates by value.
    # For a given u, we only care about v < u not in S.
    
    while time.time() < deadline:
        improved = False
        best_move = None
        best_gain = 0
        
        # Try all u in S
        for u in list(S):
            # We want to find v < u, v not in S, such that v compatible with S \ {u}
            # Iterate v from 2 to u-1
            # Optimization: If v is incompatible with any w in S\{u}, skip.
            # We can check compatibility against S\{u}.
            
            # To make this faster, we can precompute for each v the list of u's it conflicts with?
            # Or just iterate. |S| is small.
            
            # Let's iterate v in range(2, u)
            for v in range(2, u):
                if v in S:
                    continue
                # Check if v is compatible with S \ {u}
                # This is the expensive part.
                # We can break early if incompatible.
                compatible = True
                for w in S:
                    if w == u:
                        continue
                    if w in incompat[v]:
                        compatible = False
                        break
                if compatible:
                    gain = (1.0 / v) - (1.0 / u)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (u, v)
                        
        if best_move:
            u, v = best_move
            S.remove(u)
            S.add(v)
            current_score = calc_sum(S)
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            improved = True
            
        if not improved:
            # Perturbation
            # Strategy: Remove k random elements, then greedily re-add best compatible elements.
            # k = 2 or 3
            k = random.randint(1, 3)
            to_remove = random.sample(list(S), min(k, len(S)))
            for u in to_remove:
                S.remove(u)
                
            # Greedily re-add from candidates not in S, sorted by weight
            # Candidates not in S
            remaining = [x for x in candidates if x not in S]
            # Sort by x ascending
            remaining.sort()
            for x in remaining:
                if is_compatible_with_set(x, S):
                    S.add(x)
                    
            current_score = calc_sum(S)
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
                
            # If we are running out of time, break
            if time.time() > deadline - 5.0:
                break

    return list(best_S)
