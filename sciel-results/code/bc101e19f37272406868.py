import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    adj_mask = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edges.append([i, j])
    
    return edges, adj_mask, n

def _max_independent_set_size(adj_mask, n, time_limit=1.0):
    """
    Find max independent set size using bitmask branch and bound.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is a set of vertices with no edges between them.
    """
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    # non_adj_mask[i] has bit j set if i and j are NOT adjacent (and i != j)
    non_adj_mask = [0] * n
    for i in range(n):
        non_adj_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    def _search(candidates, current_size):
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates)
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Heuristic: pick vertex with max degree in the induced subgraph of candidates?
        # Or just pick the first one.
        # Let's pick a vertex that is "central" in the candidate set to prune better.
        # Actually, for independent set, we want to pick a vertex that removes many candidates.
        # v should have many neighbors in the *complement* graph? No.
        # If we include v, we must remove all neighbors of v from candidates.
        # So we want v to have many neighbors in the original graph *within* candidates?
        # No, if v is in the independent set, we remove its neighbors from candidates.
        # So we want v to have a high degree in the original graph restricted to candidates.
        
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            # Degree of idx in the subgraph induced by candidates
            # Neighbors of idx in candidates: adj_mask[idx] & candidates
            deg = (adj_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v
        # New candidates: candidates without v and without neighbors of v
        new_cands = candidates & ~(adj_mask[v] | (1 << v))
        _search(new_cands, current_size + 1)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size)

    _search(all_mask, 0)
    return best[0]

def _check_c4_free(adj_mask, n):
    """Check if graph is C4-free using bitmasks."""
    for i in range(n):
        for j in range(i+1, n):
            # Common neighbors of i and j
            common = adj_mask[i] & adj_mask[j]
            if common.bit_count() > 1:
                return False
    return True

def _try_extend(adj_mask, n, time_limit=10.0):
    """
    Try to add vertices to the graph.
    Returns new adj_mask, new n, and new edges if successful.
    """
    start_time = time.time()
    current_adj = list(adj_mask)
    current_n = n
    edges = []
    # Reconstruct edges from adj_mask for the base graph
    for i in range(current_n):
        for j in range(i+1, current_n):
            if current_adj[i] & (1 << j):
                edges.append([i, j])
                
    while time.time() - start_time < time_limit:
        added = False
        # Try to add a new vertex current_n
        # We need to find a set S of neighbors for the new vertex.
        # Condition: for all u, w in S, (current_adj[u] & current_adj[w]) == 0
        # And the new graph must have alpha < 11.
        
        # Strategy: Try random subsets S, or systematic small S.
        # Let's try S of size 1, 2, ...
        # Also, we can try to pick S that "hits" all large independent sets.
        
        # For now, let's try random S of varying sizes.
        # We need to be efficient.
        
        # Precompute pairs (u, w) that are "compatible" (disjoint neighborhoods)
        # This is expensive to recompute every time.
        # But n is small (<40).
        
        # Let's just try random S.
        for _ in range(50): # 50 trials per vertex addition attempt
            if time.time() - start_time > time_limit:
                break
            
            # Random size for S
            max_size = min(current_n, 10)
            size = random.randint(1, max_size)
            S = set(random.sample(range(current_n), size))
            
            # Check C4 condition
            valid_c4 = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    if (current_adj[u] & current_adj[w]) != 0:
                        valid_c4 = False
                        break
                if not valid_c4:
                    break
            
            if not valid_c4:
                continue
            
            # Construct new adj_mask
            new_adj = list(current_adj)
            new_adj.append(0) # New vertex
            for u in S:
                new_adj[u] |= (1 << current_n)
                new_adj[current_n] |= (1 << u)
            
            new_n = current_n + 1
            
            # Check alpha
            alpha = _max_independent_set_size(new_adj, new_n, time_limit=0.5)
            
            if alpha < 11:
                # Success
                current_adj = new_adj
                current_n = new_n
                for u in S:
                    edges.append([u, current_n - 1])
                added = True
                break
        
        if not added:
            # Try harder? Or break?
            # Let's try a few more systematic attempts before giving up on this vertex.
            # But if we can't add a vertex, we might be stuck.
            # We can try to remove some edges? No, that's complex.
            # Let's just break.
            break
            
    return current_n, edges, current_adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Build Base Graph
    edges, adj_mask, n = _build_polarity_graph_pg2_5()
    
    # Verify base
    if not _check_c4_free(adj_mask, n):
        return {'n': n, 'edges': edges}
    
    # Check alpha of base
    alpha = _max_independent_set_size(adj_mask, n, time_limit=2.0)
    
    if alpha >= 11:
        return {'n': n, 'edges': edges}

    # 2. Extension Loop
    time_left = deadline - time.time() - 10.0
    if time_left > 0:
        n, edges, adj_mask = _try_extend(adj_mask, n, time_limit=time_left)
        
        # Final validation
        if not _check_c4_free(adj_mask, n):
            # Fallback to base if extension broke something
            edges, adj_mask, n = _build_polarity_graph_pg2_5()
        
        alpha = _max_independent_set_size(adj_mask, n, time_limit=5.0)
        if alpha >= 11:
            # Fallback to base
            edges, adj_mask, n = _build_polarity_graph_pg2_5()
            alpha = _max_independent_set_size(adj_mask, n, time_limit=5.0)
            if alpha >= 11:
                # This shouldn't happen for PG(2,5)
                return {'n': 1, 'edges': []}

    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
