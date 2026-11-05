"""Construct a C4-free graph with no independent set of size 11."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with 10 disjoint triangles (seed)
    n = 30
    adj = [0] * n  # bitmask of neighbors
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a,b],[a,c],[b,c]]
        adj[a] = (1<<b) | (1<<c)
        adj[b] = (1<<a) | (1<<c)
        adj[c] = (1<<a) | (1<<b)
    
    best_n = n
    best_edges = edges[:]
    
    # Greedy: try to add vertices one at a time
    # A new vertex v can be added if:
    # 1. No C4: for any two existing vertices u,w that share v as neighbor, 
    #    they must not share another common neighbor
    # 2. Alpha stays < 11
    
    def check_c4_free(adj, n, candidate_neighbors):
        """Check if adding vertex n with given neighbors keeps graph C4-free."""
        # For each pair of neighbors (u,w) of the new vertex,
        # u and w must have at most 1 common neighbor in the existing graph
        # (the new vertex itself will be a common neighbor, so they can have at most 0 other common neighbors)
        nbrs = candidate_neighbors
        for i in range(n):
            if not (nbrs >> i) & 1:
                continue
            for j in range(i+1, n):
                if not (nbrs >> j) & 1:
                    continue
                # u=i, w=j are both neighbors of new vertex
                # They must not share any common neighbor in existing graph
                common = adj[i] & adj[j]
                if common != 0:
                    return False
        return True
    
    def check_alpha_lt_11(adj, n):
        """Check if independence number < 11 using greedy + exact for small cases."""
        # Greedy upper bound: repeatedly pick vertex with min degree in remaining
        # If we can't find independent set of size 11, return True
        # Exact: try to find independent set of size 11
        # Use branch and bound
        
        def find_independent_set(adj, n, target, remaining_mask, current_size):
            """Try to find independent set of size target."""
            if current_size + popcount(remaining_mask) < target:
                return False
            if current_size == target:
                return True
            if popcount(remaining_mask) == 0:
                return False
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with fewest neighbors in remaining
            best_v = -1
            best_count = n + 1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & (-temp_mask)
                v = lsb.bit_length() - 1
                temp_mask ^= lsb
                # Count neighbors of v in remaining
                nbrs_in_rem = adj[v] & remaining_mask
                cnt = popcount(nbrs_in_rem)
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                if best_count == 0:
                    break
            
            if best_v == -1:
                return False
            
            # Option 1: include best_v
            new_mask = remaining_mask & ~(1 << best_v)
            # Remove all neighbors of best_v
            new_mask &= ~adj[best_v]
            if find_independent_set(adj, n, target, new_mask, current_size + 1):
                return True
            
            # Option 2: exclude best_v
            new_mask2 = remaining_mask & ~(1 << best_v)
            if find_independent_set(adj, n, target, new_mask2, current_size):
                return True
            
            return False
        
        return not find_independent_set(adj, n, 11, (1 << n) - 1, 0)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Try adding vertices
    while n < 43:
        if time.time() > deadline - 5:
            break
        
        # Try different connection patterns for new vertex
        # Start with high-degree connections for better C4 properties
        found = False
        
        # Try connecting to subsets of existing vertices
        # Heuristic: connect to vertices in different triangles
        # A new vertex connected to one vertex from each of k triangles
        # For C4-freeness: any two neighbors must not share a common neighbor
        
        # Try: connect to one vertex from each of 10 triangles (different ones)
        # Or connect to 2 vertices from some triangles
        
        # Strategy: try random subsets of size 3-8
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random subset of existing vertices
            k = random.randint(3, min(8, n))
            candidates = random.sample(range(n), k)
            nbr_mask = 0
            for v in candidates:
                nbr_mask |= (1 << v)
            
            if check_c4_free(adj, n, nbr_mask):
                # Check alpha
                # Tentatively add vertex
                adj.append(nbr_mask)
                for v in candidates:
                    adj[v] |= (1 << n)
                new_n = n + 1
                
                if check_alpha_lt_11(adj, new_n):
                    # Accept
                    for v in candidates:
                        edges.append([n, v])
                    n = new_n
                    best_n = n
                    best_edges = edges[:]
                    found = True
                    break
                else:
                    # Reject
                    adj.pop()
                    for v in candidates:
                        adj[v] &= ~(1 << n)
        
        if not found:
            # Try smaller subsets
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                k = random.randint(1, 3)
                candidates = random.sample(range(n), k)
                nbr_mask = 0
                for v in candidates:
                    nbr_mask |= (1 << v)
                
                if check_c4_free(adj, n, nbr_mask):
                    adj.append(nbr_mask)
                    for v in candidates:
                        adj[v] |= (1 << n)
                    new_n = n + 1
                    
                    if check_alpha_lt_11(adj, new_n):
                        for v in candidates:
                            edges.append([n, v])
                        n = new_n
                        best_n = n
                        best_edges = edges[:]
                        found = True
                        break
                    else:
                        adj.pop()
                        for v in candidates:
                            adj[v] &= ~(1 << n)
            
            if not found:
                break
    
    return {'n': best_n, 'edges': best_edges}
