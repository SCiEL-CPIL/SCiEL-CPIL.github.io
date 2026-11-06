import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill
    
    random.seed(42)
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k4(adj, n):
        """Returns True if K4 exists, else False. 
        Also returns the K4 vertices if found for repair, but let's keep it simple first: just check."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        # Check if w is connected to any other vertex in common
                        # common ^ lsb removes w from the set
                        if adj[w] & (common ^ lsb):
                            return True
                        c ^= lsb
        return False

    def find_k4(adj, n):
        """Returns a tuple (a,b,c,d) if K4 found, else None."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        others = common ^ lsb
                        if adj[w] & others:
                            # Found w connected to someone in others
                            # Find that someone
                            t = others
                            while t:
                                lsb2 = t & -t
                                x = lsb2.bit_length() - 1
                                if adj[w] & lsb2:
                                    return (i, j, w, x)
                                t ^= lsb2
                        c ^= lsb
        return None

    def has_ind_set(adj, n, size):
        """Check if independent set of given size exists using backtracking with bitmasks."""
        # candidates: bitmask of available vertices
        # chosen: not needed if we just track size
        def backtrack(candidates, remaining):
            if remaining == 0:
                return True
            if popcount(candidates) < remaining:
                return False
            
            # Heuristic: pick a vertex with minimum degree in candidates?
            # For speed, just pick the first one? Or the one with highest degree in candidates (to prune more)?
            # If we pick a vertex v, including it removes all its neighbors.
            # Excluding it removes just v.
            # To prune effectively, we should pick a vertex that has many neighbors in candidates (so including it removes many options).
            # Let's pick the vertex with max degree in the induced subgraph of candidates.
            
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c ^= lsb
            
            if best_v == -1:
                return False
                
            v = best_v
            v_mask = 1 << v
            
            # Option 1: Include v
            # New candidates: vertices not adjacent to v
            new_cands = candidates & ~adj[v] & ~v_mask
            if backtrack(new_cands, remaining - 1):
                return True
            
            # Option 2: Exclude v
            new_cands = candidates & ~v_mask
            if backtrack(new_cands, remaining):
                return True
            
            return False
        
        initial_candidates = (1 << n) - 1
        return backtrack(initial_candidates, size)

    def find_ind_set(adj, n, size):
        """Returns a list of vertices forming an independent set of given size, or None."""
        # This is more expensive. We might just rely on has_ind_set for checking,
        # and for repair, if has_ind_set is true, we can find one via a similar search that records the path.
        
        result = []
        
        def backtrack(candidates, remaining, path):
            if remaining == 0:
                result.extend(path)
                return True
            if popcount(candidates) < remaining:
                return False
            
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c ^= lsb
            
            if best_v == -1:
                return False
                
            v = best_v
            v_mask = 1 << v
            
            # Option 1: Include v
            path.append(v)
            new_cands = candidates & ~adj[v] & ~v_mask
            if backtrack(new_cands, remaining - 1, path):
                return True
            path.pop()
            
            # Option 2: Exclude v
            new_cands = candidates & ~v_mask
            if backtrack(new_cands, remaining, path):
                return True
            
            return False
        
        initial_candidates = (1 << n) - 1
        if backtrack(initial_candidates, size, []):
            return result
        return None

    def is_valid(adj, n):
        if has_k4(adj, n):
            return False
        if has_ind_set(adj, n, 6):
            return False
        return True

    def local_search(n):
        """
        Attempts to find a valid graph on n vertices.
        Returns adj (list of int bitmasks) if found, else None.
        """
        # Initialize with random graph
        # Density ~0.5 is a good start for Ramsey-type problems
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Check if already valid
        if is_valid(adj, n):
            return adj
            
        # Local Search Loop
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                return None
            
            # Check for K4
            k4 = find_k4(adj, n)
            if k4:
                a, b, c, d = k4
                # Try to remove one of the edges in the K4
                # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                edges_to_try = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                random.shuffle(edges_to_try)
                for u, v in edges_to_try:
                    # Flip edge (remove it)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Check if this removed the K4 and didn't create an Ind6
                    # Note: Removing an edge cannot create a K4.
                    # It CAN create an independent set.
                    if not has_ind_set(adj, n, 6):
                        # We also need to ensure no other K4s exist? 
                        # Removing an edge only reduces K4s, so if we removed one specific K4, 
                        # we should check if ANY K4 remains.
                        if not has_k4(adj, n):
                            return adj
                    
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # If no single edge removal worked, try random flips? 
                # Or break and restart?
                # For now, if we can't fix a K4 by removing one of its edges, we are stuck.
                # Let's try a random flip elsewhere to escape local minimum?
                # Actually, if has_k4 is true, we MUST remove some edge. 
                # If removing any edge from a specific K4 creates an Ind6, we are in trouble.
                # Let's try removing an edge NOT in the K4? No, that won't remove the K4.
                # So we must remove an edge from the K4.
                # If all 6 removals create Ind6, we fail this instance.
                return None

            # Check for Ind6
            ind6 = find_ind_set(adj, n, 6)
            if ind6:
                # We have an independent set {v1..v6}.
                # We must add an edge between two of them to break it.
                # Adding an edge cannot create an Ind6 (it reduces independent sets).
                # It CAN create a K4.
                # Try adding random edges between the set.
                pairs = list(combinations(ind6, 2))
                random.shuffle(pairs)
                for u, v in pairs:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    # Check if this created a K4
                    if not has_k4(adj, n):
                        # We also need to ensure no other Ind6s exist?
                        # Adding an edge only reduces Ind6s, so if we broke this one,
                        # we should check if ANY Ind6 remains.
                        if not has_ind_set(adj, n, 6):
                            return adj
                    
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                
                # If no single edge addition worked, fail
                return None
            
            # If neither K4 nor Ind6 found, it's valid?
            # has_k4 and has_ind_set should be exhaustive.
            return adj
        
        return None

    # Try to construct graphs for decreasing n
    for n in range(35, 19, -1):
        # Try multiple random seeds / restarts for each n
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            
            adj = local_search(n)
            if adj is not None:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                
                # Final verification
                # Re-verify to be safe
                if is_valid(adj, n):
                    return {'n': n, 'edges': edges}
                else:
                    # Should not happen if local_search is correct, but just in case
                    pass
        
        # If we failed to find a valid graph for this n, drop to n-1
        # (Note: local_search returns None if it gets stuck or times out)
        
    # Fallback: Return a small valid graph
    # C5 is valid (n=5)
    adj_c5 = [0] * 5
    for i in range(5):
        adj_c5[i] |= (1 << ((i + 1) % 5))
        adj_c5[(i + 1) % 5] |= (1 << i)
    
    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            if adj_c5[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': 5, 'edges': edges}
