import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill

    def popcount(x):
        return bin(x).count('1')

    def verify_graph(adj, n):
        """
        Verify that no subset of size >= 6 induces a regular subgraph.
        Returns True if valid, False otherwise.
        Uses bit operations for speed.
        """
        # Precompute adjacency masks
        # adj[i] is a bitmask of neighbors of vertex i
        
        # Check all subsets of size 6 to n
        # For efficiency, check size 6 first (most likely to find violations)
        vertices = list(range(n))
        
        for k in range(6, n + 1):
            for comb in itertools.combinations(vertices, k):
                mask = 0
                for u in comb:
                    mask |= (1 << u)
                
                # Check if induced subgraph is regular
                # All vertices in the subset must have the same degree within the subset
                first_deg = -1
                is_regular = True
                for u in comb:
                    deg = popcount(adj[u] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return False
        
        return True

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    # Pre-computed valid graph on 20 vertices
    # This is a known construction: the complement of the Petersen graph blown up, or similar
    # Let me construct a specific valid graph
    
    # Strategy: Use a graph where every 6-vertex subset has non-regular induced subgraph
    # One approach: use a graph with high girth and specific properties
    
    # Let's try a specific construction:
    # Take vertices 0..19. 
    # Connect i to i+1, i+2, i+3, i+4, i+5 mod 20 (circular 5-regular)
    # But this might have regular induced subgraphs.
    
    # Better: Use a known valid graph. Since graphs on 20 vertices are known to exist,
    # let me try to find one via quick local search with verification.
    
    def find_valid_graph(n, time_budget):
        """Find a valid graph on n vertices within time_budget."""
        deadline = start_time + time_budget
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.45:  # Slightly sparse
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Quick verification
            if verify_graph(adj, n):
                return adj
            
            # Local search: flip edges to fix violations
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Find a violation
                violation_found = False
                vertices = list(range(n))
                
                # Check size 6 subsets (most common violation)
                for comb in itertools.combinations(vertices, 6):
                    mask = 0
                    for u in comb:
                        mask |= (1 << u)
                    
                    first_deg = -1
                    is_regular = True
                    for u in comb:
                        deg = popcount(adj[u] & mask)
                        if first_deg == -1:
                            first_deg = deg
                        elif deg != first_deg:
                            is_regular = False
                            break
                    
                    if is_regular:
                        violation_found = True
                        # Flip a random edge within this subset
                        u, v = random.sample(list(comb), 2)
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        break
                
                if not violation_found:
                    # Check larger subsets quickly
                    valid = True
                    for k in range(7, n + 1):
                        for comb in itertools.combinations(vertices, k):
                            mask = 0
                            for u in comb:
                                mask |= (1 << u)
                            
                            first_deg = -1
                            is_regular = True
                            for u in comb:
                                deg = popcount(adj[u] & mask)
                                if first_deg == -1:
                                    first_deg = deg
                                elif deg != first_deg:
                                    is_regular = False
                                    break
                            
                            if is_regular:
                                valid = False
                                # Flip an edge in this subset
                                u, v = random.sample(list(comb), 2)
                                if (adj[u] >> v) & 1:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                else:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return adj
        
        return None

    # Try to find a valid graph on 21 vertices first
    adj21 = find_valid_graph(21, 60.0)
    if adj21 is not None:
        return {'n': 21, 'edges': get_edges_from_adj(adj21, 21)}
    
    # Try to find a valid graph on 20 vertices
    adj20 = find_valid_graph(20, 80.0)
    if adj20 is not None:
        return {'n': 20, 'edges': get_edges_from_adj(adj20, 20)}
    
    # Fallback: Return a small valid graph
    # Any graph on 5 vertices is valid (no 6-vertex subset exists)
    n = 5
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': n, 'edges': get_edges_from_adj(adj, n)}
