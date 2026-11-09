"""Combined program: Bitmask adjacency, Repair-based Local Search for Growth."""
# EVOLVE-BLOCK-START
import time
import random

_START = time.time()
# Safety margin: 150s kill. Leave 5s for overhead/cleanup.
_DEADLINE = _START + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _paley17_bitmask():
    """Return Paley graph on 17 vertices as list of int bitmasks."""
    squares = {x * x % 17 for x in range(1, 17)}
    n = 17
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _find_k4(adj, n):
    """Return a K4 as a tuple of vertices, or None if K4-free."""
    for u in range(n):
        au = adj[u]
        # Neighbors of u greater than u
        vv = au & (~((1 << (u + 1)) - 1))
        while vv:
            v = (vv & -vv).bit_length() - 1
            vv &= vv - 1
            # Common neighbors of u and v, greater than v
            c = au & adj[v] & (~((1 << (v + 1)) - 1))
            if not c:
                continue
            cc = c
            while cc:
                x = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                # Check if x has a neighbor y in c with y > x
                ymask = adj[x] & c & (~((1 << (x + 1)) - 1))
                if ymask:
                    y = (ymask & -ymask).bit_length() - 1
                    return (u, v, x, y)
    return None

def _find_indep6(adj, n):
    """Return an independent set of 6 vertices, or None if alpha < 6."""
    # Independent set in G is a clique of size 6 in complement of G.
    # Complement adjacency: comp[i] = full_mask & ~adj[i] & ~(1<<i)
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]
    
    def _find_clique(k, adj_list, n):
        # Backtracking clique finder
        # Order vertices by degree in subgraph? For small k=6, simple backtracking is often ok.
        # We use a recursive approach with candidate sets.
        
        # Pre-compute candidate masks for each vertex based on order
        # To speed up, we can sort vertices by degree in the complement graph?
        # For now, simple recursive search.
        
        def rec(cand, start, chosen):
            if len(chosen) == k:
                return tuple(chosen)
            # Pruning: not enough candidates left
            # We iterate through candidates >= start
            c = cand & (~((1 << start) - 1))
            if c.bit_count() < k - len(chosen):
                return None
            
            # Iterate through candidates in c
            # Heuristic: try vertices with high degree in the remaining candidate set first?
            # For simplicity, iterate in increasing order.
            cc = c
            while cc:
                v = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                # If v is not in cand, skip (should be handled by mask)
                # New candidates: intersection of current cand and neighbors of v
                new_cand = cand & adj_list[v]
                # Ensure we only pick vertices > v for the next step to avoid duplicates
                # The 'start' parameter in recursion handles ordering, but we pass new_cand & (~((1<<(v+1))-1))
                
                res = rec(new_cand, v + 1, chosen + [v])
                if res is not None:
                    return res
            return None

        return rec(full, 0, [])
        
    return _find_clique(6, comp, n)

def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_indep6(adj, n) is None

def _try_add_vertex(adj, n, neighbors_mask):
    """
    Try to add a new vertex n with given neighbors_mask.
    Returns True if the new graph is valid, False otherwise.
    Does NOT modify adj.
    """
    # Check K4 involving new vertex:
    # New vertex is connected to set S = neighbors_mask.
    # A K4 involving new vertex requires a triangle in the subgraph induced by S.
    # So we check if G[S] has a triangle.
    
    # Check Indep6 involving new vertex:
    # New vertex is NOT connected to set T = V \ S.
    # An independent set of 6 involving new vertex requires an independent set of 5 in G[T].
    # So we check if G[T] has an independent set of 5 (clique of 5 in complement of G[T]).

    # 1. Check Triangle in G[S]
    # We can iterate through edges in S and check common neighbors in S.
    s_mask = neighbors_mask
    # Iterate through vertices in S
    v1 = s_mask
    while v1:
        u = (v1 & -v1).bit_length() - 1
        v1 &= v1 - 1
        # Neighbors of u in S, greater than u
        # adj[u] & s_mask & (~((1 << (u + 1)) - 1))
        nu = adj[u] & s_mask & (~((1 << (u + 1)) - 1))
        if not nu:
            continue
        v2 = nu
        while v2:
            v = (v2 & -v2).bit_length() - 1
            v2 &= v2 - 1
            # Common neighbors of u and v in S, greater than v
            # adj[u] & adj[v] & s_mask & (~((1 << (v + 1)) - 1))
            common = adj[u] & adj[v] & s_mask & (~((1 << (v + 1)) - 1))
            if common:
                # Found triangle u, v, w in S -> K4 with new vertex
                return False
    
    # 2. Check Indep5 in G[T]
    # T = all vertices except S
    # T_mask = ((1 << n) - 1) ^ s_mask
    t_mask = ((1 << n) - 1) ^ s_mask
    
    # We need to check if G[T] has an independent set of size 5.
    # This is equivalent to checking if the complement of G[T] has a clique of size 5.
    # Complement of G[T] induced on T:
    # For i, j in T, they are adjacent in comp if they are NOT adjacent in G.
    
    # Instead of building the full complement graph, we can use the _find_clique logic
    # but restricted to T.
    # Let's write a specialized clique finder for size 5 on the complement restricted to T.
    
    def _has_clique5_in_comp_T(t_mask, n):
        # Candidates are vertices in t_mask
        # We search for a clique of size 5 in the graph where edges are non-edges of G.
        
        # Optimization: If number of vertices in T < 5, return False
        if t_mask.bit_count() < 5:
            return False
            
        # Recursive search
        # State: current candidate set (mask), start vertex, chosen list
        def rec(cand, start, chosen):
            if len(chosen) == 5:
                return True
            need = 5 - len(chosen)
            # Filter candidates to those >= start
            usable = cand & (~((1 << start) - 1))
            if usable.bit_count() < need:
                return False
            
            # Heuristic: Pick vertex with highest degree in the induced subgraph?
            # For small sizes, simple iteration is fine.
            c = usable
            while c:
                v = (c & -c).bit_length() - 1
                c &= c - 1
                
                # New candidates: neighbors of v in the COMP graph, restricted to usable, > v
                # Neighbor in comp means NOT neighbor in G, and in cand, and > v
                # comp_neigh_v = cand & (~adj[v]) & (~((1 << (v + 1)) - 1))
                # Note: v is not in adj[v], so ~adj[v] includes v, but we mask > v.
                new_cand = cand & (~adj[v]) & (~((1 << (v + 1)) - 1))
                
                if rec(new_cand, v + 1, chosen + [v]):
                    return True
            return False
            
        return rec(t_mask, 0, [])
        
    if _has_clique5_in_comp_T(t_mask, n):
        return False
        
    return True

def _grow_and_repair(adj, n):
    """
    Attempt to grow the graph by adding vertices.
    Uses a repair mechanism: if adding fails, perturb the existing graph.
    """
    best_n = n
    best_adj = adj[:]
    
    # We will try to add vertices one by one.
    # If we get stuck, we flip an edge in the current graph to change the structure.
    
    stuck_counter = 0
    max_stuck = 100
    
    while _time_left() > 5.0:
        added = False
        
        # Try to add a vertex
        # Strategy: Try random neighborhoods of various sizes.
        # Bias towards sizes that are likely to be triangle-free and leave T with low alpha.
        # Triangle-free density in S is related to size.
        # Indep5 in T is related to size of T.
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not added:
            attempts += 1
            
            # Random size for neighborhood S
            # Ideally |S| is large enough to make T small (harder to find Indep5 in T)
            # but small enough to be triangle-free.
            # For n=35, average degree is ~15. So |S| ~ 15.
            # Let's sample size from a range.
            min_size = max(1, n // 4)
            max_size = min(n - 1, 3 * n // 4)
            size = random.randint(min_size, max_size)
            
            # Select random subset of size 'size'
            # For speed, we can generate a random mask and mask it to 'size' bits?
            # Simpler: random.sample
            if size > n: size = n
            if size == 0: continue
            
            # Generate random subset
            # Using random.sample is O(size).
            # For faster random mask generation:
            # mask = 0
            # for _ in range(size):
            #    v = random.randint(0, n-1)
            #    mask |= 1 << v
            # This might have duplicates. random.sample is better.
            
            subset = random.sample(range(n), size)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            if _try_add_vertex(adj, n, mask):
                # Success! Add vertex
                new_n = n + 1
                new_adj = adj[:] + [0]
                # Add edges
                new_adj[n] = mask
                for v in subset:
                    new_adj[v] |= (1 << new_n)
                
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = adj[:]
                added = True
                stuck_counter = 0
                break
        
        if not added:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perturb the graph: flip a random edge
                # Pick a random edge and flip it
                # To pick a random edge, pick two random vertices and check if edge exists?
                # Or iterate through edges.
                # For speed, pick random u, v.
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    # Edge exists, remove it
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Edge missing, add it
                    # Be careful: adding edge might create K4.
                    # But we are perturbing to *enable* growth, so we accept temporary invalidity?
                    # No, the task requires the FINAL graph to be valid.
                    # If we perturb, we MUST repair the graph to be valid again before trying to grow.
                    
                    # This is complex. Let's stick to:
                    # 1. If we can't add, we stop growing.
                    # 2. Instead of perturbation, we can try *different* random seeds for the subset selection.
                    # 3. Or, we can try to *remove* a vertex to unstick? No, we want max n.
                    
                    # Let's try a different strategy for stuck:
                    # Try smaller or larger neighborhood sizes explicitly.
                    # The random loop above already varies size.
                    # If 50 random attempts fail, it's likely hard to add a vertex.
                    
                    # Alternative: Use the Repair logic from Inspiration 2 on the *current* graph
                    # to see if we can find a valid graph with the SAME n but different structure
                    # that allows growth.
                    # But we already have a valid graph.
                    
                    # Let's just break out of the growth loop if we are stuck for too long.
                    # The current program breaks. We will do the same but with more attempts.
                    break
        
        if not added and stuck_counter > max_stuck:
            break
            
    return best_n, best_adj

def _to_dict(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    # Start with Paley 17
    adj = _paley17_bitmask()
    n = 17
    
    # Verify
    if not _is_valid(adj, n):
        # Should not happen
        return {'n': 17, 'edges': []}
        
    best_result = _to_dict(n, adj)
    
    # Grow
    n, adj = _grow_and_repair(adj, n)
    
    if n > 17:
        if _is_valid(adj, n):
            best_result = _to_dict(n, adj)
            
    return best_result
# EVOLVE-BLOCK-END
