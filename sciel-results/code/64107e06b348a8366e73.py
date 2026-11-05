import time
import random
import sys

def _remaining_time(deadline):
    return deadline - time.time()

def _has_triangle_bitset(adj, n):
    """Check if graph has a triangle using bitsets."""
    for u in range(n):
        au = adj[u]
        # Only consider neighbors v > u to avoid duplicates
        v = u + 1
        while v < n:
            if au & (1 << v):
                common = au & adj[v]
                # Only consider w > v
                if common & ~((1 << (v + 1)) - 1):
                    return True
            v += 1
    return False

def _max_independent_set_size_bitset(adj, n, k):
    """
    Check if there exists an independent set of size k using bitmask backtracking.
    Returns True if found, False otherwise.
    Uses branch and bound with coloring heuristic for speed.
    """
    full_mask = (1 << n) - 1
    # non_neighbors[i] = vertices not adjacent to i (excluding i itself)
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        
        # Prune: not enough candidates
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - current_size:
            return
        
        # Get list of candidate vertices
        candidates = []
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            candidates.append(v)
            mask ^= lsb
        
        # Heuristic: pick vertex with most non-neighbors in candidates
        # This helps prune the "include" branch faster if the vertex is highly connected
        # in the complement (i.e., has few non-neighbors in the original graph? No, wait.)
        # In the independent set problem, we are looking for a set with no edges.
        # If we include v, we must remove all neighbors of v from candidates.
        # So we want to pick v such that it removes the most candidates? 
        # Actually, standard heuristic is to pick v with max degree in the candidate subgraph.
        # Here, "degree" in the candidate subgraph for the independence problem is 
        # the number of neighbors in the original graph that are in candidates.
        
        best_v = None
        best_deg = -1
        for v in candidates:
            # Count neighbors of v in candidates
            deg_in_cands = bin(candidates_mask & adj[v]).count('1')
            if deg_in_cands > best_deg:
                best_deg = deg_in_cands
                best_v = v
        
        if best_v is None:
            return
            
        # Branch 1: Include best_v
        # New candidates are those not adjacent to best_v
        new_candidates = candidates_mask & ~adj[best_v] & ~(1 << best_v)
        dfs(new_candidates, current_size + 1)
        
        if found[0]:
            return
        
        # Branch 2: Exclude best_v
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _find_independent_set_bitset(adj, n, k):
    """Find an independent set of size k. Returns list of vertices or None."""
    full_mask = (1 << n) - 1
    
    found = [False]
    result_set = []
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            result_set.extend(current_set)
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        # Pick vertex with max degree in candidate subgraph
        candidates = []
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            candidates.append(v)
            mask ^= lsb
            
        best_v = None
        best_deg = -1
        for v in candidates:
            deg_in_cands = bin(candidates_mask & adj[v]).count('1')
            if deg_in_cands > best_deg:
                best_deg = deg_in_cands
                best_v = v
                
        if best_v is None:
            return
        
        # Include
        new_cands = candidates_mask & ~adj[best_v] & ~(1 << best_v)
        dfs(new_cands, current_set + [best_v])
        
        if found[0]:
            return
        
        # Exclude
        dfs(candidates_mask & ~(1 << best_v), current_set)
    
    dfs(full_mask, [])
    if found[0]:
        return result_set
    return None

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _schlaefi_graph():
    """
    Returns the Schläfli graph (27 vertices, 3-regular? No, 16-regular).
    It is triangle-free and has alpha=5.
    Vertices are pairs (i, j) where i in Z_3, j in Z_9? 
    Actually, Schläfli graph is the complement of the point graph of the generalized quadrangle GQ(2,4).
    Simpler construction:
    Vertices are (x, y) where x in {0,1,2} and y in {0,...,8}.
    Wait, standard Schläfli graph has 27 vertices.
    Let's use a known explicit construction.
    
    The Schläfli graph is the unique 16-regular triangle-free graph on 27 vertices with alpha=5.
    It can be constructed as follows:
    Let V = Z_3 x Z_9.
    Two vertices (a,b) and (a',b') are adjacent if a != a' and b' - b is in S_a where S_a depends on a?
    
    Actually, a simpler way to get a strong seed is to use the Clebsch graph (16 vertices, alpha=5)
    and expand it.
    
    However, for n=39, we need a specific known graph.
    The largest known triangle-free graph with alpha < 10 is n=39.
    
    Let's try to construct a graph on n=39 by taking the Schläfli graph (27 vertices)
    and adding 12 vertices? That's hard.
    
    Alternative: Use the "Hoffman-Singleton" graph? No, that's 50 vertices and has triangles? No, HS is triangle-free, alpha=12? 
    R(3,10) <= 41 means max n is 40.
    
    Let's use a randomized local search that is very aggressive.
    """
    pass

def _local_search_optimized(adj, n, target_alpha, deadline):
    """
    Optimized local search to reduce alpha below target_alpha.
    Uses a greedy edge addition strategy.
    """
    # Precompute non-neighbors for faster access? 
    # We will recompute in the inner loop for simplicity but keep it fast.
    
    while time.time() < deadline:
        # Check if valid
        if not _max_independent_set_size_bitset(adj, n, target_alpha):
            return True
        
        # Find an independent set of size target_alpha
        iso = _find_independent_set_bitset(adj, n, target_alpha)
        if iso is None:
            return True
            
        # Try to add an edge within this independent set that doesn't create a triangle
        # Randomize the order of pairs to escape local minima
        random.shuffle(iso)
        
        improved = False
        for i in range(len(iso)):
            for j in range(i + 1, len(iso)):
                u, v = iso[i], iso[j]
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates a triangle
                    # Triangle exists if there is w adjacent to both u and v
                    common = adj[u] & adj[v]
                    if common == 0:
                        # Safe to add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # If no edge can be added without creating a triangle, 
            # we might be stuck. Try removing an edge that is "useless" for triangle-free constraint?
            # Or just break and restart with a different random seed?
            # For now, break to avoid infinite loop in this function.
            break
            
    return not _max_independent_set_size_bitset(adj, n, target_alpha)

def _construct_seed_graph(n, deadline):
    """
    Construct a good seed graph for local search.
    For n=39, we can try to build it from smaller components or random dense graph.
    """
    # Start with a random graph and remove triangles
    adj = [0] * n
    # Use a density that is high enough to keep alpha low
    # For n=39, target alpha < 10.
    # Random graph G(n, p) has alpha ~ 2 log_b n where b = 1/(1-p).
    # If p=0.5, alpha is small. But we need triangle-free.
    
    # Strategy: Start with empty graph, add edges randomly, remove triangles immediately.
    # This is "random greedy triangle-free".
    
    # To bias towards high n, we want to be efficient.
    # Let's try a few different densities.
    
    best_adj = None
    best_alpha = n
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
            
        adj_try = [0] * n
        # Generate random edges
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_try[i] |= (1 << j)
                    adj_try[j] |= (1 << i)
                    
        # Remove triangles
        # Simple iterative removal
        changed = True
        while changed:
            changed = False
            for u in range(n):
                au = adj_try[u]
                v = u + 1
                while v < n:
                    if au & (1 << v):
                        common = au & adj_try[v] & ~((1 << (v + 1)) - 1)
                        if common:
                            lsb = common & -common
                            w = lsb.bit_length() - 1
                            # Remove edge (u, v)
                            adj_try[u] &= ~(1 << v)
                            adj_try[v] &= ~(1 << u)
                            changed = True
                            break
                    v += 1
                if changed:
                    break
                    
        # Check alpha
        # We don't need exact alpha, just if it's < 10
        if not _max_independent_set_size_bitset(adj_try, n, 10):
            return adj_try
            
        # If not valid, keep the one with smallest estimated alpha?
        # For now, just return the last one if we run out of time, 
        # but we want to run local search on it.
        if best_adj is None:
            best_adj = adj_try
            
    return best_adj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Target n=39 first, as it is the known max.
    # If we fail, try 38, 37...
    
    for n in [39, 38, 37, 36]:
        if time.time() > deadline - 20:
            break
            
        # Generate seed
        adj = _construct_seed_graph(n, deadline - 10)
        
        if adj is None:
            continue
            
        # Run local search
        # We have a time budget for local search
        ls_deadline = min(deadline - 5, start_time + 40)
        
        if _local_search_optimized(adj, n, 10, ls_deadline):
            return {'n': n, 'edges': _bitset_to_edges(adj, n)}
            
    # Fallback: return a small valid graph if all else fails
    # n=9 empty graph is valid (alpha=9 < 10, no triangles)
    n = 9
    return {'n': n, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
