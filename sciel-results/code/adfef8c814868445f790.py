import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_valid_full(n, adj):
        """Check if graph has no regular induced subgraph on 6+ vertices."""
        # We must check all k >= 6
        # Optimization: If no 6-subset is regular, it's still possible a 7-subset is.
        # But empirically, avoiding 6-subsets is the hard part.
        # Let's check 6 first, then 7..n if time permits or if n is small enough.
        
        # Check k=6
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            d0 = (adj[subset[0]] & mask).bit_count()
            regular = True
            for v in subset[1:]:
                if (adj[v] & mask).bit_count() != d0:
                    regular = False
                    break
            if regular:
                return False
        
        # Check k=7..n (only if n is small enough to be feasible, or always for correctness)
        # For n=21, C(21,7) ~ 116k, C(21,8) ~ 203k... sum is manageable if optimized.
        for k in range(7, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                d0 = (adj[subset[0]] & mask).bit_count()
                regular = True
                for v in subset[1:]:
                    if (adj[v] & mask).bit_count() != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    def count_violations_6(n, adj, subsets_6):
        """Count number of regular 6-subsets."""
        count = 0
        for subset in subsets_6:
            mask = 0
            for v in subset:
                mask |= (1 << v)
            d0 = (adj[subset[0]] & mask).bit_count()
            regular = True
            for v in subset[1:]:
                if (adj[v] & mask).bit_count() != d0:
                    regular = False
                    break
            if regular:
                count += 1
        return count

    def greedy_search(n, time_limit):
        """
        Systematic greedy search:
        1. Start with random graph.
        2. Precompute all 6-subsets.
        3. Loop:
           a. Find the edge flip that minimizes count_violations_6.
           b. If no improvement, try random flips or restart.
           c. Check full validity periodically.
        """
        subsets_6 = list(combinations(range(n), 6))
        
        # Initialize random graph
        adj = [0] * n
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.add((i, j))
        
        current_cost = count_violations_6(n, adj, subsets_6)
        
        best_adj = adj[:]
        best_cost = current_cost
        best_edges = edges.copy()
        
        # Precompute list of all potential edges to flip
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        no_improvement_count = 0
        max_no_improvement = 20 # Restart if stuck
        
        while time.time() < time_limit:
            if no_improvement_count >= max_no_improvement:
                # Random restart
                adj = [0] * n
                edges = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            edges.add((i, j))
                current_cost = count_violations_6(n, adj, subsets_6)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    best_edges = edges.copy()
                no_improvement_count = 0
                continue
            
            # Find best edge flip
            best_flip_edge = None
            min_cost = current_cost
            
            # Sample a subset of edges to flip to save time? 
            # For n=21, C(21,2)=210 edges. 210 * 38760 (subsets) * 6 (degree checks) ~ 48M ops.
            # This is too slow in Python per iteration.
            # We need a faster heuristic or sampling.
            
            # Strategy: Randomly sample 20 edges to flip, pick the best one.
            sampled_edges = random.sample(all_edges, min(20, len(all_edges)))
            
            for u, v in sampled_edges:
                # Flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_cost = count_violations_6(n, adj, subsets_6)
                
                # Unflip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_flip_edge = (u, v)
            
            if best_flip_edge is not None:
                u, v = best_flip_edge
                # Apply flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                if (u, v) in edges:
                    edges.remove((u, v))
                else:
                    edges.add((u, v))
                
                current_cost = min_cost
                no_improvement_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    best_edges = edges.copy()
                    
                    # Check full validity if cost is low
                    if best_cost == 0:
                        if is_valid_full(n, best_adj):
                            return best_adj, list(best_edges)
            else:
                no_improvement_count += 1
                
        # Return best found
        if best_cost == 0 and is_valid_full(n, best_adj):
            return best_adj, list(best_edges)
        return None, None

    # Try n=21, then 20
    for n in [21, 20]:
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 7 + n)
            adj, edges = greedy_search(n, deadline - 5)
            if adj is not None:
                return {'n': n, 'edges': edges}
        
        # If greedy failed, try a quick local search as fallback?
        # Or just move to next n
        
    # Fallback: return a valid small graph
    return {'n': 5, 'edges': []}
