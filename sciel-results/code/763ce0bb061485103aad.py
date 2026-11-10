# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    best_size = 0
    
    # Precompute for each vertex v, the bitmask of vertices u such that 
    # (v, u) is an edge. This is just adj_bitmasks[v].
    # We need to check if adding v to a set S creates a triangle.
    # A triangle is formed if there exist u, w in S such that (v,u), (v,w), (u,w) are edges.
    # So if we add v, we must remove from 'available' any u such that 
    # (N(u) & N(v) & S) is not empty? 
    # Actually, if we add v, any u in S that is adjacent to v AND adjacent to some w in S 
    # (where w != v, u) would form a triangle (v, u, w).
    # Wait, the condition is: S is triangle-free.
    # If we add v to S, we need to ensure no triangle (v, u, w) is formed with u, w in S.
    # This happens if u is in N(v) & S AND w is in N(v) & N(u) & S.
    # So for each u in N(v) & S, if N(u) & N(v) & S is non-empty, we have a conflict.
    # Actually, it's simpler: 
    # When we add v, the new set of "forbidden" vertices for future additions is not the issue.
    # The issue is whether the *current* set S plus v is triangle-free.
    # S is triangle-free. Adding v is safe iff for all u in S \cap N(v), 
    # the number of neighbors of u in S \cap N(v) is 0? 
    # No. If u in S \cap N(v), and there is w in S \cap N(v) such that (u,w) is an edge, 
    # then (v,u,w) is a triangle.
    # So, v can be added to S iff the subgraph induced by S \cap N(v) is independent (edge-free).
    
    # To make the B&B fast, we need to quickly determine which vertices in 'available' 
    # can be added to the current set 'current_mask'.
    # Actually, standard B&B for Max Independent Set on a graph G' where vertices are 
    # triangles? No.
    
    # Let's use a simpler B&B:
    # State: available_mask, current_mask, current_size.
    # Pruning: if current_size + available_mask.bit_count() <= best_size: return.
    # Heuristic: Pick a vertex v from available_mask.
    # Branch 1: Include v.
    #   Check if adding v to current_mask creates a triangle.
    #   If yes, skip.
    #   If no, new_current_mask = current_mask | (1<<v).
    #   new_available_mask = available_mask & ~(1<<v) & ~conflicts(v, current_mask).
    #   What are conflicts?
    #   If we add v, we must remove any u from available such that adding u later 
    #   would create a triangle with v and some w in new_current_mask?
    #   No, the triangle must be within the final set.
    #   If we add v, any future vertex u must not form a triangle with v and any w in new_current_mask.
    #   Also, u must not form a triangle with any pair in new_current_mask.
    
    # Let's refine the "conflict" removal.
    # If we have set S and add v.
    # New set S' = S U {v}.
    # Any vertex u in available that is adjacent to v AND adjacent to some w in S 
    # (where w is adjacent to v? No, w just needs to be in S and adjacent to u and v)
    # If (u,v) is edge and (u,w) is edge and (v,w) is edge, then triangle.
    # So if we add v, we must remove from 'available' any u such that:
    # 1. (u,v) is an edge.
    # 2. There exists w in S such that (u,w) is an edge AND (v,w) is an edge.
    #    i.e., u is in N(v) & N(w) for some w in S.
    #    This is equivalent to: u is in N(v) & (union of N(w) for w in S).
    #    Let CommonNeighborsInS = N(v) & S.
    #    For any u in N(v), if N(u) & CommonNeighborsInS is not empty, then u conflicts.
    #    Wait, if u is in N(v) and w is in S, and (u,w) is edge, then (v,u,w) is triangle 
    #    IF (v,w) is edge.
    #    So we need w to be in N(v) & S.
    #    So, let V_S_Nv = N(v) & S.
    #    If V_S_Nv is empty, adding v is always safe w.r.t triangles involving v and S.
    #    If V_S_Nv is not empty, then any u in N(v) that is adjacent to any w in V_S_Nv 
    #    forms a triangle (v,u,w).
    #    So, Forbidden = N(v) & (Union of N(w) for w in V_S_Nv).
    #    We must remove Forbidden from 'available'.
    
    # Also, we must ensure that the remaining 'available' vertices don't form triangles 
    # among themselves or with S. But S is triangle-free.
    # The standard B&B for Max Triangle-Free Set is tricky.
    
    # Alternative: Max Independent Set in the "Triangle Graph"?
    # Construct a graph H where vertices are 0..n-1.
    # (u,v) is an edge in H if u,v cannot be in the same triangle-free set.
    # When can u,v not be in the same TF set?
    # If they form a triangle with some w? No, the set must be TF.
    # A set S is TF if it contains no triangles.
    # This is not a pairwise constraint! It's a hypergraph constraint (triangles).
    # So it's not an Independent Set problem.
    
    # However, n=18 is small.
    # Let's stick to the bitmask iteration but optimize it heavily with numpy?
    # Or just a very tight Python loop?
    # 2^18 = 262,144 masks.
    # Check each mask: O(n^2) worst case. 18^2 = 324.
    # 262k * 324 ~ 85M operations.
    # In Python, 85M simple ops might take 5-10 seconds.
    # If we call this 100 times, we are dead.
    
    # We need a faster check.
    # Precompute for each mask if it is triangle-free? No, 2^18 memory is fine (1MB), 
    # but computing it takes time.
    
    # Let's try to speed up the check for a specific mask.
    # A mask is TF iff for all edges (i,j) in the induced subgraph, 
    # there is no k in mask such that (i,k) and (j,k) are edges.
    # i.e., for all i<j in mask, if adj[i][j], then (N(i) & N(j) & mask) == 0.
    
    # We can precompute `tri_edge_mask[i][j]` = N(i) & N(j).
    # Then check: for all i<j in mask, if adj[i][j], then (tri_edge_mask[i][j] & mask) == 0.
    
    # This is still O(n^2) per mask.
    
    # Can we prune the mask iteration?
    # If mask is not TF, any superset is not TF.
    # If mask is TF, any subset is TF.
    # We want the MAX size TF set.
    # We can iterate masks in DECREASING size order?
    # No, generating combinations is slow.
    
    # Let's try a different B&B:
    # We build the set S.
    # If we add v, we check if S U {v} is TF.
    # If yes, recurse.
    # If no, skip.
    # This is basically what I described before.
    # The key is the "Forbidden" calculation.
    
    # Let's implement this B&B carefully.
    
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute N(i) & N(j) for all pairs to check triangle existence quickly?
    # common[i][j] = adj[i] & adj[j]
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
            
    def dfs(available, current_mask, current_size):
        nonlocal best
        # Pruning
        if current_size + available.bit_count() <= best:
            return
            
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Heuristic: pick a vertex. 
        # Choosing a vertex with high degree in 'available' might prune more?
        # Or just lowest bit.
        v_bit = available & -available
        v = v_bit.bit_length() - 1
        
        # Option 1: Include v
        # Check if adding v to current_mask creates a triangle.
        # Triangles involving v: (v, u, w) where u, w in current_mask.
        # This exists if there is u in current_mask such that (v,u) is edge 
        # and there is w in current_mask such that (v,w) and (u,w) are edges.
        # Let N_v_in_cur = adj_bitmasks[v] & current_mask.
        # If N_v_in_cur is empty, no triangle with v and 2 others in cur.
        # If N_v_in_cur is not empty, check if any u in N_v_in_cur has a neighbor w in N_v_in_cur.
        # i.e., is the subgraph induced by N_v_in_cur edge-free?
        
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            # Check if N_v_in_cur has any edge
            temp = N_v_in_cur
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                # Check if u has any neighbor in N_v_in_cur
                if adj_bitmasks[u] & N_v_in_cur:
                    safe_to_add = False
                    break
                temp -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            # Determine which vertices in 'available' are now forbidden.
            # If we add v, we cannot add any u in the future such that 
            # (u, v, w) is a triangle for some w in new_mask.
            # Also, we cannot add u if (u, w1, w2) is a triangle for w1, w2 in new_mask?
            # No, new_mask is TF. So no triangles within new_mask.
            # The only new triangles involving v are (v, u, w) with w in new_mask.
            # Since new_mask is TF, w must be in N(v) & new_mask.
            # So if we add u, we must ensure no w in N(v) & new_mask is adjacent to u.
            # i.e., u must not be in N(v) & (Union of N(w) for w in N(v) & new_mask).
            
            # Let Forbidden = 0
            # For each w in N(v) & new_mask:
            #   Forbidden |= N(w)
            # Forbidden &= N(v)  # u must be adjacent to v to form triangle with v
            # Forbidden &= available  # u must be in available
            
            # Wait, if u is not adjacent to v, can it form a triangle with v? No.
            # So we only care about u in N(v).
            
            N_v_in_new = adj_bitmasks[v] & new_mask
            forbidden = 0
            temp_w = N_v_in_new
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                forbidden |= adj_bitmasks[w]
                temp_w -= w_bit
            
            forbidden &= adj_bitmasks[v] # u must be neighbor of v
            forbidden &= available       # u must be in available
            
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
            
        # Option 2: Exclude v
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _adj_to_bitmasks(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    bm = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        bm[i] = mask
    return bm

def _check_k4_free_bitmask(bm, n):
    """Check if graph represented by bitmasks is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                common = bm[i] & bm[j]
                while common:
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        return False
                    common -= k_bit
    return True

def _random_k4_free_graph(n, seed):
    """Generate a random K4-free graph using bitmask representation."""
    rng = random.Random(seed)
    bm = [0] * n
    
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for i, j in edges_to_try:
        if rng.random() < 0.5:
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def _local_search_bitmask(bm, n, max_iter=500, seed=0, deadline=None):
    """Local search on bitmask representation."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        if not (bm[i] & (1 << j)):
            # Try adding edge
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        else:
            # Try removing edge
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    
    # 1. Baseline: Turan graph T(18, 3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_bm[a] |= (1 << b)
                    baseline_bm[b] |= (1 << a)
    
    baseline_score = _max_tf_size_fast(baseline_bm, n)
    best_score = baseline_score
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if baseline_bm[i] & (1 << j):
                best_edges.append([i, j])
    
    # 2. Random restarts with local search
    # Use more iterations now that evaluation is faster?
    # _max_tf_size_fast is B&B. It should be much faster than 2^18 loop.
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        bm, edges = _random_k4_free_graph(n, seed)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search
        if time.time() > deadline - 10:
            break
            
        ls_edges, ls_score = _local_search_bitmask(bm, n, max_iter=200, seed=seed, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            
    # 3. Final Polish: Try to improve best_edges with aggressive local search
    if time.time() < deadline - 5:
        # Convert best_edges to bm
        best_bm = [0] * n
        for e in best_edges:
            u, v = e
            best_bm[u] |= (1 << v)
            best_bm[v] |= (1 << u)
            
        # Run a dedicated local search on the best candidate
        # Use a different seed or just continue
        polish_edges, polish_score = _local_search_bitmask(best_bm, n, max_iter=1000, seed=999, deadline=deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_edges = polish_edges

    return best_edges
# EVOLVE-BLOCK-END
