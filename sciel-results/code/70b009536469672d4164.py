import time
import random
import sys

def _elapsed():
    return time.time() - _start_time

_start_time = time.time()
_DEADLINE = 115.0 # 5s margin

def _has_triangle(adj, n):
    # adj is list of bitsets (int)
    for i in range(n):
        # Check if any neighbor of i is connected to another neighbor
        # This is slow. Better: iterate edges.
        pass
    # We will maintain triangle-freeness during construction.

def _max_independent_set_size(adj, n):
    # Branch and bound
    # adj[i] is bitset of neighbors of i
    # We want max set S such that no edges in S.
    # This is max clique in complement.
    
    # Use bitset operations
    best = 0
    # Simple BnB
    def bnb(candidates, size):
        nonlocal best
        if not candidates:
            if size > best:
                best = size
            return
        if size + bin(candidates).count('1') <= best:
            return
        # Pick a vertex
        # Heuristic: pick vertex with highest degree in induced subgraph?
        # Or just lowest index.
        v = (candidates & -candidates).bit_length() - 1
        # Remove v from candidates
        new_cand = candidates & ~(1 << v)
        # Branch 1: include v
        bnb(new_cand & ~adj[v], size + 1)
        # Branch 2: exclude v
        bnb(new_cand, size)
        
    bnb((1 << n) - 1, 0)
    return best

def construct():
    global _start_time
    _start_time = time.time()
    
    best_n = 26
    best_edges = None
    
    # Seed graph
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    # Verify seed
    # If valid, best_n = 26
    
    for target_n in [40, 39]:
        if _elapsed() > _DEADLINE:
            break
            
        found = False
        attempts = 0
        max_attempts = 500
        
        while not found and attempts < max_attempts and _elapsed() < _DEADLINE:
            attempts += 1
            n = target_n
            # Initialize random graph
            # Use bitsets
            adj = [0] * n
            
            # Start with random edges
            p = 0.4
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        # Check if adding (i,j) creates triangle
                        # Triangle if common neighbor
                        common = adj[i] & adj[j]
                        if common == 0:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Now we have a triangle-free graph.
            # Try to maximize edges (greedy)
            # Iterate through all pairs, add if possible
            # Repeat until no change
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not (adj[i] & (1 << j)):
                            # Check triangle
                            if (adj[i] & adj[j]) == 0:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                changed = True
                if _elapsed() > _DEADLINE:
                    break
            
            if _elapsed() > _DEADLINE:
                break
                
            # Check alpha
            alpha = _max_independent_set_size(adj, n)
            
            if alpha <= 9:
                # Found valid graph
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                found = True
                break
                
    if best_edges is None:
        # Fallback to seed
        return {'n': 26, 'edges': edges_seed}
    else:
        return {'n': best_n, 'edges': best_edges}
