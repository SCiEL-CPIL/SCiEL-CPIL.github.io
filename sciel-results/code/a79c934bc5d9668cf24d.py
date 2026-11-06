import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        # Check min degree
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def solve_for_n(n, min_deg):
        """
        Find a C4-free graph on n vertices with min degree >= min_deg.
        Uses random initialization and local search repair.
        """
        best_adj = None
        best_score = -1
        
        # Number of attempts
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 104729 + 1)
            
            # Initialize with random graph
            # Target degree ~ min_deg + 1
            p = (min_deg + 1.0) / (n - 1)
            adj = [0] * n
            deg = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        deg[i] += 1
                        deg[j] += 1
            
            # Repair phase
            # We maintain a list of "bad" pairs (C4 violations)
            # and a list of "low" degree vertices
            
            improved = True
            max_iter = 2000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Find C4 violations
                # Optimization: Only check pairs that might have changed?
                # For simplicity, full check, but optimized with bitmasks.
                violating_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if bin(common).count('1') > 1:
                            violating_pairs.append((i, j))
                
                # Check min degree
                low_verts = [i for i in range(n) if deg[i] < min_deg]
                
                if not violating_pairs and not low_verts:
                    # Success
                    if check_valid(adj, n, min_deg):
                        return adj
                    break
                
                if violating_pairs:
                    # Fix a random C4 violation
                    i, j = random.choice(violating_pairs)
                    common = adj[i] & adj[j]
                    # Get list of common neighbors
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove edges to reduce common neighbors to 1
                    # We need to remove at least len(common_list) - 1 edges
                    # Prefer removing edges incident to the vertex with higher degree
                    # or random
                    
                    # Sort common neighbors by degree (descending) to remove from higher degree vertices first?
                    # Actually, removing edge (i, w) reduces deg[i] and deg[w].
                    # Removing edge (j, w) reduces deg[j] and deg[w].
                    
                    # We need to keep 1 common neighbor.
                    # Let's keep the one with lowest degree impact?
                    # Or just remove random ones until 1 left.
                    
                    # Shuffle to randomize which one to keep
                    random.shuffle(common_list)
                    # Keep the first one, remove the rest
                    keep_w = common_list[0]
                    for w in common_list[1:]:
                        # Decide whether to remove (i, w) or (j, w)
                        if deg[i] > deg[j]:
                            # Remove (i, w)
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg[i] -= 1
                            deg[w] -= 1
                        elif deg[j] > deg[i]:
                            # Remove (j, w)
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg[j] -= 1
                            deg[w] -= 1
                        else:
                            if random.random() < 0.5:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg[i] -= 1
                                deg[w] -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg[j] -= 1
                                deg[w] -= 1
                else:
                    # No C4s, but some low degree vertices
                    if not low_verts:
                        break
                    u = random.choice(low_verts)
                    # Find a non-neighbor v such that N(u) & N(v) == 0
                    candidates = []
                    for v in range(n):
                        if v == u:
                            continue
                        if not (adj[u] & (1 << v)):
                            if (adj[u] & adj[v]) == 0:
                                candidates.append(v)
                    
                    if candidates:
                        # Prefer candidates with low degree to boost them too?
                        # Or random?
                        v = random.choice(candidates)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                    else:
                        # No candidate, try to "move" an edge?
                        # This is complex. Just break or try another attempt.
                        pass
        
        return None

    # Try n from 47 down to 43
    # Based on analysis, 47 is likely impossible for delta 8.
    # 46 requires delta 7.
    # 45 requires delta 6.
    
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
        
        # If we fail for n, try n-1

    # Fallback: Return a valid small graph
    # For n=40, min_deg=1.
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
