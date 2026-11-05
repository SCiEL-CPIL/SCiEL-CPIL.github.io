import time
import random
import itertools

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    Vertices are points of the plane.
    Two points p and r are adjacent if p lies on the polar line of r.
    This graph is C4-free.
    """
    # Generate all points in PG(2, q)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero coordinate is 1
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    # n_pts should be q^2 + q + 1
    
    adj = [set() for _ in range(n_pts)]
    
    # Polarity: The standard orthogonal polarity
    # Point P=(x1, x2, x3) has polar line x1*x + x2*y + x3*z = 0
    # Point Q=(y1, y2, y3) is on polar of P iff x1*y1 + x2*y2 + x3*y3 = 0
    # We want simple graph, so exclude self-loops (P on its own polar line is 'absolute point', but we don't add self edges)
    # And symmetry holds.
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            # Check if j is on polar of i
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(adj, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses branch and bound with bitmasks for speed.
    Returns True if an independent set of size >= limit exists.
    """
    if n <= 64:
        # Convert adj to bitmasks
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        all_mask = (1 << n) - 1
        
        def _branch(rem_mask, current_size):
            if current_size >= limit:
                return True
            if rem_mask == 0:
                return False
            # Popcount bound
            if current_size + bin(rem_mask).count('1') < limit:
                return False
            
            # Pick vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            m = rem_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                deg = bin(adj_mask[v] & rem_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
            if _branch(rem_include, current_size + 1):
                return True
            
            # Branch 2: Exclude best_v
            rem_exclude = rem_mask & ~(1 << best_v)
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(all_mask, 0)
    else:
        # Fallback for larger n
        remaining = set(range(n))
        
        def _branch(rem, current_size):
            if current_size >= limit:
                return True
            if not rem:
                return False
            if current_size + len(rem) < limit:
                return False
            
            best_v = -1
            best_deg = -1
            for v in rem:
                deg = len(adj[v] & rem)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            rem_include = rem - adj[best_v] - {best_v}
            if _branch(rem_include, current_size + 1):
                return True
            
            rem_exclude = rem - {best_v}
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(remaining, 0)

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Target n=43
    # Try q=7 first (57 vertices)
    # Then q=5 (31 vertices - too small for n=43, but valid for n=31)
    # Then q=11 (133 vertices - allows n=43)
    
    qs_to_try = [7, 11, 5, 13]
    
    for q in qs_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            n_full, edges_full, adj_full = _construct_polarity_graph(q)
            
            # Verify C4-free on full graph
            if not _verify_c4_free(n_full, adj_full):
                continue
                
            # We want a subgraph of size n_target <= 43
            # If n_full >= 43, we can pick a subset of 43 vertices.
            # If n_full < 43, we use n_full.
            
            target_n = min(43, n_full)
            
            if target_n <= best_n:
                continue
                
            # Try to find a subset of size target_n with alpha < 11
            # If the full graph has alpha < 11, any subset works.
            # Check full graph alpha first? It might be slow for n=133.
            # Let's just check the subset.
            
            # If n_full is large, we can try random subsets.
            # If n_full is small (e.g., 31), we take the whole graph.
            
            found_valid_subset = False
            candidate_edges = []
            candidate_n = 0
            
            if n_full == target_n:
                # Take the whole graph
                # Check alpha
                if not _max_independent_set(adj_full, n_full, limit=11):
                    found_valid_subset = True
                    candidate_n = target_n
                    candidate_edges = edges_full
                # If alpha >= 11, we can't use the whole graph. 
                # But we can't remove vertices to get a larger n, we can only get smaller.
                # So we would need to search for a subset of size < target_n.
                # For now, skip if full graph is invalid and we can't go smaller?
                # Actually, we can try to remove vertices to reduce alpha.
                # But for q=5 (n=31), alpha is usually small.
                
            else:
                # n_full > target_n (e.g., 57 > 43)
                # Try to find a subset of size target_n
                # Strategy: Random sampling
                attempts = 0
                max_attempts = 1000 if time.time() < deadline - 20 else 100
                
                while attempts < max_attempts and not found_valid_subset:
                    if time.time() > deadline - 5:
                        break
                    attempts += 1
                    
                    # Randomly select target_n vertices from n_full
                    selected_indices = random.sample(range(n_full), target_n)
                    # Map old indices to new 0..target_n-1
                    old_to_new = {old: new for new, old in enumerate(selected_indices)}
                    
                    new_n = target_n
                    new_adj = [set() for _ in range(new_n)]
                    
                    for old_i in selected_indices:
                        new_i = old_to_new[old_i]
                        for old_j in adj_full[old_i]:
                            if old_j in old_to_new:
                                new_j = old_to_new[old_j]
                                new_adj[new_i].add(new_j)
                                new_adj[new_j].add(new_i)
                                
                    # Verify C4-free (should be, as it's induced subgraph)
                    # But let's be safe
                    if not _verify_c4_free(new_n, new_adj):
                        continue
                        
                    # Check alpha
                    if not _max_independent_set(new_adj, new_n, limit=11):
                        found_valid_subset = True
                        candidate_n = new_n
                        # Build edges
                        candidate_edges = []
                        for i in range(new_n):
                            for j in new_adj[i]:
                                if i < j:
                                    candidate_edges.append([i, j])
                        
            if found_valid_subset:
                if candidate_n > best_n:
                    best_n = candidate_n
                    best_edges = candidate_edges
                    
                # If we found n=43, we are done (max possible)
                if best_n == 43:
                    break
                    
        except Exception:
            continue
            
    # Fallback: If we didn't find anything good, try q=5 (n=31) explicitly if not done
    # Or just return best found.
    
    # If best_n is still 0, something went wrong. Return a small valid graph.
    if best_n == 0:
        # Triangle on 3 vertices? Alpha=1 < 11. C4-free.
        best_n = 3
        best_edges = [[0, 1], [1, 2], [0, 2]]
        
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
