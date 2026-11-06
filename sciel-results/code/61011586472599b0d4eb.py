import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    # Construct Polarity Graph of PG(2, 6)
    q = 6
    # Points in PG(2, q) are non-zero vectors in F_q^3 modulo scalar multiplication.
    # We represent them as tuples (x, y, z) with z=1 if possible, else y=1, else x=1.
    # Or simply enumerate all non-zero vectors and pick a canonical representative.
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: first non-zero component is 1
                # Since q is prime, we can divide by the first non-zero component.
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    p = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    p = (0, 1, (z * inv_y) % q)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n = len(points) # Should be q^2 + q + 1 = 43
    
    # Build adjacency list using bitmask
    # Polarity: (x1, y1, z1) ~ (x2, y2, z2) iff x1*y2 + y1*z2 + z1*x2 == 0
    adj_mask = [0] * n
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n):
            x2, y2, z2 = points[j]
            # Check orthogonality
            if (x1 * y2 + y1 * z2 + z1 * x2) % q == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Helper to check C4-free
    def check_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
        return True

    # Helper to find a maximum independent set (returns bitmask)
    def find_max_independent_set(adj, n):
        best_set = [0]
        best_size = [0]
        
        def dfs(candidates, current_set, size):
            if size > best_size[0]:
                best_size[0] = size
                best_set[0] = current_set
            
            if candidates == 0:
                return
            
            # Upper bound
            if size + candidates.bit_count() <= best_size[0]:
                return
            
            # Pick vertex with max degree in candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            new_candidates = candidates & ~(adj[chosen] | (1 << chosen))
            dfs(new_candidates, current_set | (1 << chosen), size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, current_set, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0, 0)
        return best_set[0]

    # Helper to get edges from adj_mask
    def get_edges(adj, n):
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # Verify initial graph
    if not check_c4_free(adj_mask, n):
        # Fallback if construction failed (should not happen)
        return {'n': 30, 'edges': []}
        
    current_n = n
    current_adj = adj_mask[:]
    
    # Check alpha
    max_ind_set = find_max_independent_set(current_adj, current_n)
    alpha = max_ind_set.bit_count()
    
    # If alpha >= 11, we need to remove vertices
    # We will greedily remove vertices from the max independent set
    # to reduce the independence number.
    max_iterations = 20
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        if alpha < 11:
            break
            
        # Get the vertices in the max independent set
        # We pick one vertex to remove.
        # Heuristic: Remove a vertex from the independent set that has the highest degree in the current graph.
        # This is likely to break many other independent sets.
        candidates_to_remove = []
        m = max_ind_set
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = current_adj[v].bit_count()
            candidates_to_remove.append((deg, v))
            
        # Sort by degree descending
        candidates_to_remove.sort(reverse=True)
        
        # Try removing the top candidate
        if not candidates_to_remove:
            break
            
        _, v_to_remove = candidates_to_remove[0]
        
        # Remove vertex v_to_remove
        # We need to create a new graph on the remaining vertices.
        # It's easier to keep the original indices and just mark as removed, 
        # but the output requires vertices 0..n-1. 
        # So we must remap.
        
        # Let's just rebuild the graph on the subset of vertices.
        remaining_vertices = [i for i in range(current_n) if i != v_to_remove]
        new_n = len(remaining_vertices)
        
        # Map old index to new index
        old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
        
        new_adj = [0] * new_n
        for old_u in remaining_vertices:
            new_u = old_to_new[old_u]
            m = current_adj[old_u]
            while m:
                lsb = m & (-m)
                old_v = lsb.bit_length() - 1
                m ^= lsb
                if old_v in old_to_new:
                    new_v = old_to_new[old_v]
                    new_adj[new_u] |= (1 << new_v)
        
        current_n = new_n
        current_adj = new_adj
        
        # Recompute alpha
        max_ind_set = find_max_independent_set(current_adj, current_n)
        alpha = max_ind_set.bit_count()
        
        # Safety check: if we drop below 39, we might want to stop or just continue?
        # The task says maximize n. 39 is the known best.
        # If we reach 39, we should stop to avoid going lower unnecessarily if we can't prove 40 is possible.
        # However, if 40 is possible, we should find it.
        # Given the time, if we are at 39, we can stop.
        if current_n <= 39:
            break

    # Final check
    if alpha < 11:
        edges = get_edges(current_adj, current_n)
        return {'n': current_n, 'edges': edges}
    else:
        # Fallback
        fallback_edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            fallback_edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': fallback_edges}

