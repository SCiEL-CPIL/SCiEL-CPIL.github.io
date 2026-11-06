import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 290
    
    # Helper to check if a subset is a clique or independent set
    def is_clique(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def is_independent(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if (adj[u] >> v) & 1:
                    return False
        return True

    def find_offending_set(adj, n, size=5):
        # Find a clique or independent set of size 'size'
        # Returns the set or None
        # Optimization: Use random sampling of subsets if n is large
        # But n=42 is small enough for some brute force or efficient search?
        # C(42, 5) is ~ 850k. Checking all is feasible per step?
        # 850k * 10 checks = 8.5M ops. Maybe too slow for every step.
        
        # Better: Randomly sample subsets.
        for _ in range(1000):
            subset = random.sample(range(n), size)
            if is_clique(adj, subset):
                return subset
            if is_independent(adj, subset):
                return subset
        return None

    def count_offending(adj, n, size=5):
        # Estimate the number of offending sets
        # This is expensive. We'll just rely on finding one.
        pass

    def solve_for_n(n_target):
        # Try to find a valid graph on n_target vertices
        # using simulated annealing / local search
        max_iters = 50000
        temp = 1.0
        cooling = 0.9995
        
        # Initialize random graph
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_adj = None
        best_score = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Check if current is valid
            # We can do a quick check
            valid = True
            # Instead of checking all C(n,5), we check random samples
            # If we find one, it's invalid.
            # If we don't find one after many samples, it's likely valid.
            # But for correctness in the return, we MUST verify fully.
            
            # For the search loop, we just want to minimize "badness".
            # Let's define badness as: 1 if offending set found in random sample, 0 otherwise?
            # No, that's noisy.
            
            # Strategy: 
            # 1. Find an offending set. If none found in K samples, assume valid (tentatively).
            # 2. If found, flip an edge in it.
            
            offending = find_offending_set(adj, n_target)
            
            if offending is None:
                # Likely valid. Verify fully.
                # Full verification: Check all C(n,5) cliques and indep sets.
                # C(42,5) ~ 850k. 
                # Loop 850k times, check 10 pairs. ~8.5M ops.
                # In Python, this might take ~1-2 seconds.
                # We can afford a few full checks.
                
                is_valid = True
                for comb in combinations(range(n_target), 5):
                    if is_clique(adj, comb) or is_independent(adj, comb):
                        is_valid = False
                        break
                
                if is_valid:
                    # Convert adj to edges
                    edges = []
                    for i in range(n_target):
                        for j in range(i+1, n_target):
                            if (adj[i] >> j) & 1:
                                edges.append((i, j))
                    return n_target, edges
            
            # If not valid (or offending found), perform a move
            if offending:
                # Pick a random edge in the offending set to flip
                u, v = random.sample(offending, 2)
                # Flip edge (u,v)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # No offending set found in random sample, but not fully verified?
                # Or maybe we just flip a random edge to explore.
                u, v = random.sample(range(n_target), 2)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            temp *= cooling
            
        return None

    # First, try to find a 42-vertex graph
    # We can try multiple restarts if time permits
    result = solve_for_n(42)
    if result:
        return result
        
    # If 42 fails, try 41
    result = solve_for_n(41)
    if result:
        return result
        
    # Fallback to Paley graph on 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj_mask = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
