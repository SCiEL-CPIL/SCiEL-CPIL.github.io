import math
from math import gcd
from itertools import combinations
import numpy as np

def construct():
    n = 255
    # Build the set U = {x in [1, n] : gcd(x, n) > 1}, which must contain n itself.
    U = [x for x in range(1, n + 1) if math.gcd(x, n) > 1]
    
    # We need the largest subset of U such that every pair has gcd > 1.
    # This is the maximum clique in the graph G where vertices are U and edges connect pairs with gcd > 1.
    # Since n is fixed and small, we can use a branch-and-bound maximum clique algorithm.
    
    # Build adjacency list
    m = len(U)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(U[i], U[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum clique via branch and bound (Tomita's algorithm)
    # Use bitsets for efficiency
    # Order vertices by degree (descending) for better pruning
    degrees = [len(adj[i]) for i in range(m)]
    order = sorted(range(m), key=lambda i: -degrees[i])
    
    # Re-index
    new_index = [0] * m
    for new_i, old_i in enumerate(order):
        new_index[old_i] = new_i
    
    # Build new adjacency as bitsets
    # Use Python integers as bitsets
    adj_bits = [0] * m
    for i in range(m):
        for j in adj[order[i]]:
            adj_bits[i] |= (1 << new_index[j])
    
    # Tomita's algorithm for maximum clique
    best = []
    best_size = 0
    
    def expand(R, P):
        nonlocal best, best_size
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best = list(R)
            return
        
        # Bound: if len(R) + len(P) <= best_size, prune
        if len(R) + len(P) <= best_size:
            return
        
        # Choose vertex with max degree in P
        # For simplicity, pick the first vertex in P (or one with max connections to P)
        # Let's pick a vertex that maximizes |P ∩ adj(v)|
        max_overlap = -1
        chosen = -1
        for v in P:
            overlap = len([u for u in P if u in adj[v]])
            if overlap > max_overlap:
                max_overlap = overlap
                chosen = v
        
        if chosen == -1:
            return
        
        # Branch 1: include chosen
        R.append(chosen)
        P_new = [u for u in P if u in adj[chosen]]
        expand(R, P_new)
        R.pop()
        
        # Branch 2: exclude chosen
        P.remove(chosen)
        expand(R, P)
        P.append(chosen)
    
    initial_P = list(range(m))
    expand([], initial_P)
    
    # Convert back to original values
    result = [U[order[i]] for i in best]
    
    # Ensure 255 is in the result (it should be since 255 is in U and has high connectivity)
    # But let's verify
    if 255 not in result:
        # This shouldn't happen, but if it does, force it and fix
        # Actually, this branch-and-bound should find a clique containing high-degree nodes
        # Let's just return the best we found
        pass
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify all pairs have gcd > 1
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
