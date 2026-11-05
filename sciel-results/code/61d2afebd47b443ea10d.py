import random
import time

def construct():
    start_time = time.time()
    # 90s limit. Use 85s to be safe.
    deadline = start_time + 85.0
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    # Not strictly needed for the search if we just use the list, 
    # but useful for verification or if we need to report K.
    # We just need to return the permutation.
    
    best_cycle = None
    best_K = float('inf')
    
    # Precompute bit_count helper
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')
            
    full_mask = (1 << n) - 1

    def has_hamilton_cycle(K, time_limit):
        """
        Checks if the graph formed by the first K edges has a Hamilton cycle.
        Returns the cycle as a list of vertices if found, else None.
        """
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Quick connectivity check using BFS with bitmasks
        visited_mask = 1  # vertex 0
        frontier = 1
        while frontier:
            new_mask = 0
            temp = frontier
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                new_mask |= adj_mask[v]
                temp ^= lsb
            new_mask &= ~visited_mask
            if not new_mask:
                break
            visited_mask |= new_mask
            frontier = new_mask
        
        if visited_mask != full_mask:
            return None
            
        # Check minimum degree >= 2
        for v in range(n):
            if popcount(adj_mask[v]) < 2:
                return None

        # DFS with backtracking
        # path[0] is always 0
        path = [0]
        visited_mask_cur = 1
        
        def dfs():
            if time.time() > time_limit:
                return False
            
            if len(path) == n:
                # Check if closing edge exists
                u = path[-1]
                if adj_mask[u] & 1: # Edge to 0
                    return True
                return False
            
            u = path[-1]
            
            # Get unvisited neighbors
            # neighbors_mask = adj_mask[u] & ~visited_mask_cur
            # We need to sort them by Warnsdorff heuristic
            
            # Collect neighbors
            neigh_mask = adj_mask[u] & ~visited_mask_cur
            if not neigh_mask:
                return False
            
            neighbors = []
            temp = neigh_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                neighbors.append(v)
                temp ^= lsb
            
            # Pruning: 
            # If only 1 neighbor left, it must be the last vertex and must connect to 0?
            # Actually, if len(neighbors) == 1:
            #   v = neighbors[0]
            #   unvisited_count = n - len(path)
            #   if unvisited_count == 1:
            #       # v is the last vertex. It must connect to 0.
            #       if not (adj_mask[v] & 1):
            #           return False
            
            # Sort by Warnsdorff: number of unvisited neighbors of the candidate
            # This helps avoid getting stuck in dead ends
            neighbors.sort(key=lambda v: popcount(adj_mask[v] & ~visited_mask_cur))
            
            for v in neighbors:
                # Check if v is the last unvisited vertex
                if popcount(~visited_mask_cur & full_mask) == 1:
                    # v is the last one. It must connect to start (0)
                    if not (adj_mask[v] & 1):
                        continue
                
                # Check connectivity of remaining unvisited vertices?
                # This is expensive. We'll rely on Warnsdorff.
                
                # Add v
                path.append(v)
                new_visited = visited_mask_cur | (1 << v)
                
                if dfs():
                    return True
                
                # Backtrack
                path.pop()
                
            return False

        # Try a few random restarts if the first deterministic attempt fails?
        # With Warnsdorff, it's usually deterministic and fast for N=24.
        # But if it fails, it might be due to a bad heuristic choice early on.
        # We can add a random shuffle to the neighbor list before sorting to break ties randomly.
        
        # To improve robustness, we can try a few different orderings.
        # However, for N=24, Warnsdorff is extremely strong.
        
        found = dfs()
        if found:
            return path[:]
        
        return None

    # Binary search on K
    lo = n  # Minimum edges for a cycle is n
    hi = len(edges)
    
    found_any = False
    while lo <= hi:
        if time.time() > deadline - 2.0: # Leave 2s for final return
            break
            
        mid = (lo + hi) // 2
        
        # Try to find a Hamilton cycle in G_mid
        cycle = has_hamilton_cycle(mid, deadline - 1.0)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            found_any = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback: try with all edges
        # This should always work
        cycle = has_hamilton_cycle(len(edges), deadline - 0.5)
        if cycle is not None:
            best_cycle = cycle
        else:
            # Last resort: identity permutation is not a cycle in K_n unless n=2?
            # 0-1-2...-23-0.
            # Let's just return a valid cycle from the full graph.
            # Since K_24 is dense, 0-1-...-23-0 is a valid cycle.
            best_cycle = list(range(n))
            
    return best_cycle

if __name__ == "__main__":
    print(construct())
