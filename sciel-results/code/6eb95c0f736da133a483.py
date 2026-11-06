import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def has_c4_fast(adj):
        """Fast C4 check using numpy matrix multiplication."""
        common = adj @ adj.T
        np.fill_diagonal(common, 0)
        return np.any(common >= 2)
    
    def min_degree_fast(adj):
        return int(np.min(adj.sum(axis=1)))
    
    def try_repair(adj, n, target_min_deg):
        """
        Repair loop: remove edges to eliminate C4s while trying to maintain min degree.
        Returns True if successful.
        """
        # Precompute degrees for fast access
        degs = adj.sum(axis=1).copy()
        
        max_iterations = 20000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return False
            
            # Find a violating pair (u, v) with >1 common neighbor
            # common[u,v] = number of common neighbors
            common = adj @ adj.T
            np.fill_diagonal(common, 0)
            
            # Find max common count
            max_common = np.max(common)
            if max_common <= 1:
                # No C4s, check min degree
                if min(degs) >= target_min_deg:
                    return True
                else:
                    return False
            
            # Get indices of the pair with max common
            idx = np.unravel_index(np.argmax(common), common.shape)
            u, v = idx[0], idx[1]
            
            if u == v:
                continue
                
            # Get common neighbors
            common_neighbors = np.where((adj[u] == 1) & (adj[v] == 1))[0]
            
            if len(common_neighbors) <= 1:
                # Should not happen if max_common > 1, but safety
                continue
            
            # We need to reduce common neighbors to 1.
            # Strategy: pick a common neighbor 'a' and remove edge (u,a) or (v,a).
            # Prefer removing from the vertex with higher degree to preserve min degree?
            # Actually, removing (u,a) reduces deg(u) and deg(a).
            # We want to keep deg(u) and deg(v) high if they are near target.
            # But we also don't want to drop deg(a) below target.
            
            # Let's try a smarter heuristic:
            # Remove edge (u, a) if deg(u) > deg(v), else (v, a).
            # This balances degrees.
            
            # Pick a random common neighbor to remove
            a = common_neighbors[random.randint(0, len(common_neighbors)-1)]
            
            if degs[u] >= degs[v]:
                # Remove (u, a)
                adj[u, a] = 0
                adj[a, u] = 0
                degs[u] -= 1
                degs[a] -= 1
            else:
                # Remove (v, a)
                adj[v, a] = 0
                adj[a, v] = 0
                degs[v] -= 1
                degs[a] -= 1
                
        return False

    def try_construct(n, p, seed):
        """
        Generate random graph and repair.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate random graph
        # Use numpy for speed
        # Create upper triangle random matrix
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        rand_mat = np.random.random((n, n))
        adj = (rand_mat < p).astype(np.uint8)
        adj = adj * mask
        adj = adj + adj.T
        
        # Check min degree first (cheap)
        degs = adj.sum(axis=1)
        if np.min(degs) < n - 39:
            # Try to add edges? Or just skip.
            # Adding edges increases C4 risk. 
            # If p is high enough, min degree should be okay.
            # Let's try to repair min degree by adding random non-edges?
            # Or just skip this attempt if min degree is too low.
            # Actually, if p is too low, we'll fail. If p is too high, we'll have many C4s.
            # Let's just proceed to repair C4s, but if min degree is already low,
            # removing edges will make it worse.
            # So if min_degree < target, it's likely hopeless unless we add edges.
            # But adding edges creates C4s.
            # Let's stick to: if min_degree < target, return None.
            # We tune p to be high enough.
            return None
        
        # Repair C4s
        if try_repair(adj, n, n - 39):
            # Final verification
            if not has_c4_fast(adj) and min_degree_fast(adj) >= n - 39:
                return adj
        return None

    best_n = 39
    best_edges = []
    
    # Try n=46 and n=47
    # For n=46, target min deg = 7
    # For n=47, target min deg = 8
    
    # Probabilities to try
    # For n=46, p=0.3 gives expected deg ~ 13.8. p=0.2 gives ~9.2.
    # We need min deg >= 7.
    # Let's try a range of p.
    
    for n in [47, 46]:
        target_min_deg = n - 39
        
        # Try different probabilities
        # Higher p -> higher min deg, but more C4s.
        # Lower p -> lower min deg, but fewer C4s.
        # We need a sweet spot.
        
        p_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        
        found = False
        for p in p_list:
            if found:
                break
            # Try multiple seeds
            for seed in range(200):
                if time.time() > deadline:
                    break
                
                adj = try_construct(n, p, seed)
                if adj is not None:
                    # Convert to edge list
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] == 1:
                                edges.append([u, v])
                    
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            if time.time() > deadline:
                break
        
        if found:
            # If we found n=47, we are done.
            # If we found n=46, we could try n=47 if time permits, but the loop structure
            # tries 47 first. So if 47 failed, we try 46.
            # If 46 succeeds, we break out of the n-loop?
            # The task says maximize n. So 47 is better than 46.
            # The loop tries 47 first. If it fails, it tries 46.
            # If 46 succeeds, we return 46.
            # We don't need to try 47 again if it already failed.
            break

    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
