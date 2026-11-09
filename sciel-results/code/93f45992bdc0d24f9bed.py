"""Triangle-free diameter-four augmentation via greedy construction and local search."""
# EVOLVE-BLOCK-START

import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _diameter(added_edges, base_adj):
    """Calculate diameter of the graph formed by base edges + added edges. Returns inf if disconnected."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in base_adj:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 for d in dist):
            return float('inf')
        max_d = max(max_d, max(dist))
    return max_d

def _is_valid(added_edges, base_adj):
    """Check if the set of added edges results in a valid graph."""
    # 1. Check for duplicate edges or self-loops (though we generate carefully)
    # 2. Check triangle-free (bipartite constraint: even-odd)
    for u, v in added_edges:
        if u % 2 == v % 2:
            return False
    
    # 3. Check diameter
    d = _diameter(added_edges, base_adj)
    return d <= 4

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 # Safety margin
    
    n = P['n']
    base_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    
    # Generate all valid candidate edges (even-odd, not in base)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if u % 2 != v % 2 and (min(u,v), max(u,v)) not in base_edges:
                candidates.append((u, v))
    
    # Strategy: Greedy Add + Local Remove
    # 1. Start with a large random subset or all candidates?
    #    Adding all candidates makes diameter 1 or 2, which is valid.
    #    Then we remove edges to minimize count.
    #    However, removing too many might break diameter <= 4.
    
    # Let's try a few random initial sets of size ~15-20 and try to shrink them.
    # Or start with a known valid large set (e.g. all edges connecting to center) and shrink.
    
    best_edges = None
    best_len = float('inf')
    
    # Initial guess: Hub and spoke at 12 (current solution)
    # This is a valid starting point of size 10.
    initial_guess = [
        (12, 1), (12, 3), (12, 5), (12, 7), (12, 9),
        (12, 15), (12, 17), (12, 19), (12, 21), (12, 23)
    ]
    
    if _is_valid(initial_guess, base_edges):
        best_edges = initial_guess
        best_len = len(initial_guess)
    
    # Local Search: Try to remove edges from best_edges
    # If removal maintains validity, keep it.
    # Repeat until no more edges can be removed.
    
    current_edges = list(best_edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(current_edges)
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            edge_to_remove = current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            if _is_valid(test_edges, base_edges):
                current_edges = test_edges
                best_edges = current_edges
                best_len = len(current_edges)
                improved = True
                break # Restart loop with new list
    
    # Try a few random restarts to see if we can find a smaller set
    # Start with a random valid set? Hard to generate.
    # Instead, start with a superset of the best solution and try to remove.
    # Or, generate a random set of size 12, check validity, if valid, try to reduce.
    
    for _ in range(50): # Limit restarts
        if time.time() > deadline:
            break
        
        # Generate a random candidate set
        # We want a set that is likely valid. 
        # Picking random edges might result in disconnected or high diameter.
        # Better: Start with the best solution found so far, add 2 random edges, then reduce?
        # Or: Start with a random set of 15 edges, check if valid. If yes, reduce.
        
        # Let's try: Take best_edges, add 2 random candidates not in it, then try to reduce.
        if len(candidates) > len(best_edges) + 2:
            available = [e for e in candidates if e not in set(best_edges)]
            if len(available) < 2:
                continue
            add1, add2 = random.sample(available, 2)
            test_edges = list(best_edges) + [add1, add2]
            
            # Reduce this new set
            reduced_edges = list(test_edges)
            improved = True
            while improved:
                improved = False
                random.shuffle(reduced_edges)
                for i in range(len(reduced_edges)):
                    edge_to_remove = reduced_edges[i]
                    temp_edges = reduced_edges[:i] + reduced_edges[i+1:]
                    if _is_valid(temp_edges, base_edges):
                        reduced_edges = temp_edges
                        improved = True
                        break
            
            if len(reduced_edges) < best_len:
                best_edges = reduced_edges
                best_len = len(reduced_edges)

    # Final validation of the best found
    if best_edges is None:
        # Fallback to the hardcoded solution if search failed
        return [
            [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
            [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
        ]
    
    # Convert tuples to lists for return
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
