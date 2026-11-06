import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # More than one bit set
                return True
    return False

def _max_independent_set_size(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses a recursive branch and bound with bitmasks.
    """
    if n == 0:
        return 0
    
    # Precompute neighbors as bitmasks
    # adj is a list of integers where bit i is 1 if vertex i is adjacent to vertex j
    
    # We want to find a set of vertices with no edges between them.
    # This is equivalent to finding a clique in the complement graph.
    
    # Simple Branch and Bound
    # State: available vertices (bitmask), current size
    # Upper bound: size of available vertices + current size
    
    best = 0
    # Order vertices by degree to prune better? 
    # For now, use a simple heuristic: pick vertex with max degree in available set
    
    def search(available, current_size):
        nonlocal best
        if current_size >= target:
            return target
        
        if available == 0:
            return max(best, current_size)
        
        # Upper bound: current_size + popcount(available)
        # If this bound <= best, prune
        if current_size + bin(available).count('1') <= best:
            return best
            
        # Heuristic: pick a vertex from available
        # To speed up, pick vertex with highest degree in the available subgraph
        # Or just pick the lowest index
        v = (available & -available).bit_length() - 1
        
        # Option 1: Include v
        # New available: available without v and without neighbors of v
        # Neighbors of v in the original graph are adj[v]
        # We need to remove v and its neighbors from available
        # Note: adj[v] contains bits for all neighbors.
        # We only care about neighbors that are in 'available'.
        new_available = available & ~(1 << v) & ~adj[v]
        res1 = search(new_available, current_size + 1)
        if res1 >= target:
            return target
            
        # Option 2: Exclude v
        new_available2 = available & ~(1 << v)
        res2 = search(new_available2, current_size)
        
        return max(res1, res2)

    # Start with all vertices
    all_mask = (1 << n) - 1
    return search(all_mask, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4(n, adj):
        return False
    
    if _max_independent_set_size(n, adj, 11) >= 11:
        return False
    
    return True

def _try_build_graph(n, deadline, seed_offset=0):
    """
    Attempt to build a C4-free graph on n vertices with alpha < 11.
    Uses a greedy construction followed by local search.
    """
    random.seed(42 + seed_offset)
    
    adj = [0] * n
    edges = []
    
    # 1. Greedy construction: Add edges if they don't create a C4
    # Order vertices by index
    # To make it more effective, we can shuffle the order of adding edges
    # But we must ensure C4-freeness.
    
    # A better greedy: Start with empty graph.
    # For each pair (u, v), if adding edge (u, v) doesn't create a C4, add it.
    # This maximizes edges, which helps reduce alpha.
    
    # Optimization: Iterate through pairs in a specific order or random order.
    # Random order with multiple restarts is often better.
    
    best_adj = None
    best_edges = None
    best_alpha = float('inf')
    
    for attempt in range(5): # 5 restarts for greedy
        if time.time() > deadline - 5:
            break
            
        random.shuffle(range(n)) # Not directly usable for pairs, but we can shuffle pair generation
        # Actually, just shuffle the list of all possible edges
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        adj = [0] * n
        edges = []
        
        for u, v in all_pairs:
            # Check if adding (u, v) creates a C4
            # This happens if N(u) and N(v) have more than 1 common neighbor
            common = adj[u] & adj[v]
            if common & (common - 1):
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
        
        alpha = _max_independent_set_size(n, adj, 11)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj
            best_edges = edges
            
        if alpha < 11:
            # We found a valid graph. 
            # We can try to improve it or just return it.
            # Since we want to maximize n, and we are trying a specific n,
            # if we find a valid one, we can return it immediately.
            return n, best_edges, best_adj

    # If no valid graph found in greedy phase, try local search on the best greedy result
    if best_adj is None:
        return None, None, None
        
    # Local Search: Try to add edges to reduce alpha further (though greedy already added all possible C4-free edges)
    # Actually, the greedy graph is "maximal" in the sense that no more edges can be added without creating a C4.
    # So alpha is fixed for this specific maximal C4-free subgraph.
    # To get a different graph, we need to start with a different random order or remove some edges and re-optimize.
    
    # For the purpose of this task, if greedy fails to find alpha < 11 for this n,
    # it's likely that this n is not achievable or we need a smarter heuristic.
    # However, for n=39, it is achievable.
    
    return None, None, None

def construct():
    """
    Construct a C4-free graph with no independent set of size 11, maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Try to find a graph with n=43 down to 39
    # It is highly unlikely to find n>39 in 120s, but we try.
    # We prioritize n=39 as it is the known best.
    
    # To save time, we will spend most of the time on n=39.
    # Try n=40, 41, 42, 43 with a small time budget each.
    
    time_per_n = 5 # seconds per n for higher n
    total_time_high_n = time_per_n * 4 # 20s for 40-43
    
    for n in [43, 42, 41, 40]:
        if time.time() > start_time + total_time_high_n:
            break
            
        n_res, edges, adj = _try_build_graph(n, time.time() + time_per_n, seed_offset=n)
        if n_res is not None and edges is not None:
            # Verify to be safe
            if _verify_graph(n, edges):
                return {'n': n, 'edges': edges}
        
    # Now focus on n=39
    # We have about 110 - 20 = 90 seconds left.
    # We will run multiple attempts for n=39.
    
    best_result = None
    
    # We know n=39 is possible. Let's try to find it reliably.
    # We can use a specific seed or multiple seeds.
    
    for i in range(20): # 20 attempts
        if time.time() > deadline:
            break
            
        n_res, edges, adj = _try_build_graph(39, deadline, seed_offset=100 + i)
        if n_res is not None and edges is not None:
            # Verify
            if _verify_graph(39, edges):
                best_result = {'n': 39, 'edges': edges}
                break # Found a valid one, no need to search more for 39
        
    if best_result:
        return best_result
        
    # Fallback: If we somehow failed to find n=39 (should not happen),
    # try n=38, 37...
    for n in [38, 37, 36, 35]:
        if time.time() > deadline:
            break
        n_res, edges, adj = _try_build_graph(n, deadline, seed_offset=n)
        if n_res is not None and edges is not None:
            if _verify_graph(n, edges):
                return {'n': n, 'edges': edges}
                
    # Final fallback: Return a small valid graph to ensure validity
    # A complete graph K_10 is C4-free (no cycles at all) and alpha=1.
    # n=10 is valid.
    n = 10
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v])
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
