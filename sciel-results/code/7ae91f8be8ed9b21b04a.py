import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to maximize n. Try from 43 down to 30.
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random constructions for this n
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            if attempts > 500: # Cap attempts per n to avoid getting stuck
                break
                
            graph = build_maximal_c4_free(n)
            if is_valid_independent_number(graph, n):
                # Convert to required format
                edges = []
                for u in range(n):
                    for v in graph[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
    # Fallback: Return the known 30-vertex construction if all else fails
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def build_maximal_c4_free(n):
    """
    Builds a maximal C4-free graph on n vertices by adding random edges 
    that do not create a C4.
    """
    adj = [set() for _ in range(n)]
    
    # Generate all possible edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    
    random.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u, v) creates a C4
        # This happens if u and v have more than one common neighbor
        common = adj[u].intersection(adj[v])
        if len(common) == 0:
            adj[u].add(v)
            adj[v].add(u)
            
    return adj

def is_valid_independent_number(adj, n):
    """
    Checks if the graph has an independent set of size 11.
    Returns True if NO independent set of size 11 exists (i.e., alpha < 11).
    """
    # First, a quick greedy check
    if greedy_independent_size(adj, n) >= 11:
        return False
        
    # If greedy didn't find one, do a more thorough check using backtracking
    # We only need to find one set of size 11 to invalidate.
    return not exists_independent_set_of_size(adj, 11)

def greedy_independent_size(adj, n):
    """
    Greedy algorithm to find a large independent set.
    """
    remaining = set(range(n))
    independent = set()
    
    # Sort vertices by degree (ascending) to pick low-degree vertices first?
    # Or just random order?
    # A common heuristic: pick vertex with min degree, add to independent, remove it and its neighbors.
    while remaining:
        # Find vertex with minimum degree in the induced subgraph
        min_v = -1
        min_deg = float('inf')
        for v in remaining:
            deg = len(adj[v].intersection(remaining))
            if deg < min_deg:
                min_deg = deg
                min_v = v
        if min_v == -1:
            break
        independent.add(min_v)
        # Remove min_v and its neighbors from remaining
        neighbors = adj[min_v].intersection(remaining)
        remaining.discard(min_v)
        remaining.difference_update(neighbors)
        
    return len(independent)

def exists_independent_set_of_size(adj, k):
    """
    Backtracking search for an independent set of size k.
    """
    # Use a simple recursive backtracking
    # To make it faster, we can use bitmasks or sets.
    # Given n <= 43, sets are fine.
    
    def backtrack(current_set, available):
        if len(current_set) == k:
            return True
        if not available:
            return False
            
        # Heuristic: if |available| + |current_set| < k, fail
        if len(available) + len(current_set) < k:
            return False
            
        # Pick a vertex from available
        # Try vertices with lower degree first?
        # For simplicity, just pick the first one
        v = next(iter(available))
        
        # Option 1: Include v
        new_available = available - {v} - adj[v]
        if backtrack(current_set | {v}, new_available):
            return True
            
        # Option 2: Exclude v
        available.discard(v)
        if backtrack(current_set, available):
            return True
            
        # Restore v for backtracking correctness if we were modifying in place
        # But we passed 'available' as a set, and sets are mutable.
        # We should pass copies or be careful.
        # Let's rewrite to pass immutable or copies to be safe.
        
    # Better implementation:
    def bt(available, current_size):
        if current_size == k:
            return True
        if not available:
            return False
        if len(available) + current_size < k:
            return False
            
        # Pick a vertex
        v = next(iter(available))
        
        # Include v
        new_avail = available - {v} - adj[v]
        if bt(new_avail, current_size + 1):
            return True
            
        # Exclude v
        new_avail_excl = available - {v}
        if bt(new_avail_excl, current_size):
            return True
            
        return False

    return bt(set(range(n)), 0)
