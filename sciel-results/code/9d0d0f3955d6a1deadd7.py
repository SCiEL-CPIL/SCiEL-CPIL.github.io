import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    rng = random.Random(42)
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        """
        # Adding (u, v) affects triples containing u and v.
        # Specifically, for any w, the set of common neighbors of {u, v, w} changes.
        # New common neighbors = N(u) & N(v) & N(w).
        # We need this to be <= 2 for all w.
        
        # Note: u and v are not neighbors of themselves.
        # N(v) does not contain u initially.
        
        # For any w in N(v) (excluding u, which isn't there yet):
        # The common neighbors of {u, v, w} will be N(u) & N(v) & N(w).
        # We must ensure |N(u) & N(v) & N(w)| <= 2.
        
        nv = adj[v]
        nu = adj[u]
        
        # Iterate over neighbors of v
        mask = nv
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask ^= lsb
            
            # Check common neighbors of u, v, w
            common = nu & nv & adj[w]
            if bit_count(common) > 2:
                return False
                
        # Iterate over neighbors of u
        mask = nu
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask ^= lsb
            
            # Check common neighbors of u, v, w
            common = nu & nv & adj[w]
            if bit_count(common) > 2:
                return False
                
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    edges.append((i, j))
        return edges

    def get_existing_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def find_violation(adj):
        """
        Find a triple (u, v, w) with > 2 common neighbors.
        Returns (u, v, w, common_neighs_list) or None.
        """
        for u in range(n):
            for v in range(u+1, n):
                common_uv = adj[u] & adj[v]
                # If common_uv has >= 3 bits, we need to check triples involving these
                # Actually, we just need to check if any w in common_uv has
                # common neighbors with u and v that are also in common_uv?
                # No, the condition is |N(u) & N(v) & N(w)| >= 3.
                # Let C = N(u) & N(v).
                # We need to check if there exists w such that |C & N(w)| >= 3.
                # Note that w can be any vertex, not necessarily in C.
                # However, if w is not in C, then w is not a common neighbor of u and v.
                # But the triple is {u, v, w}. The common neighbors are N(u) & N(v) & N(w).
                # So we just iterate all w.
                
                # Optimization: If bit_count(common_uv) < 3, we can skip?
                # No. If |C| < 3, then |C & N(w)| <= |C| < 3. So it's safe.
                if bit_count(common_uv) < 3:
                    continue
                    
                # Check all w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    common_uvw = common_uv & adj[w]
                    if bit_count(common_uvw) >= 3:
                        # Found a violation
                        # Return the triple and the common neighbors
                        cn_list = []
                        mask = common_uvw
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            cn_list.append(idx)
                            mask ^= lsb
                        return (u, v, w, cn_list)
        return None

    def repair_graph(adj):
        """
        Remove edges until the graph is valid.
        """
        while True:
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w, cn_list = viol
            # Remove a random edge incident to one of the common neighbors
            # to break the K3,3
            # We can remove edge (u, cn) for a random cn in cn_list
            cn = rng.choice(cn_list)
            remove_edge(adj, u, cn)
            # Also, removing (u, cn) might not be enough if (v, cn) and (w, cn) are present?
            # Yes, K3,3 requires all 9 edges. Removing one edge from the 9 breaks it.
            # So removing (u, cn) is sufficient to break this specific violation.
            
    def local_search(adj, max_iters=200):
        """
        Perform local search: try adding edges, then try swaps.
        """
        cur_edges = count_edges(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Phase 1: Add-only
            missing = get_missing_edges(adj)
            rng.shuffle(missing)
            for (u, v) in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            
            # Phase 2: Swap
            if not improved:
                existing = get_existing_edges(adj)
                missing = get_missing_edges(adj)
                rng.shuffle(existing)
                rng.shuffle(missing)
                
                # Try a limited number of swaps
                swap_limit = 100
                swaps_tried = 0
                for e1 in existing:
                    if swaps_tried >= swap_limit:
                        break
                    e1_u, e1_v = e1
                    for e2 in missing:
                        if swaps_tried >= swap_limit:
                            break
                        e2_u, e2_v = e2
                        
                        # Skip if same edge
                        if (e1_u, e1_v) == (e2_u, e2_v) or (e1_u, e1_v) == (e2_v, e2_u):
                            continue
                        
                        # Try swap
                        remove_edge(adj, e1_u, e1_v)
                        if is_valid_add(adj, e2_u, e2_v):
                            add_edge(adj, e2_u, e2_v)
                            improved = True
                            # After one successful swap, restart
                            break
                        else:
                            add_edge(adj, e1_u, e1_v)
                        swaps_tried += 1
                    if improved:
                        break
                        
        return adj, cur_edges

    best_edges = 0
    best_adj = None
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_starts = 50
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # 1. Random Initialization
        adj = [0] * n
        # Start with a moderately dense random graph
        for (u, v) in all_edges:
            if rng.random() < 0.6:
                add_edge(adj, u, v)
        
        # 2. Repair
        repair_graph(adj)
        
        # 3. Local Search
        adj, cur_edges = local_search(adj)
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
            
        # 4. Perturbation and Re-search (if time)
        if time.time() > deadline - 20:
            continue
            
        # Perturb: Remove 5 random edges
        adj_pert = [a for a in adj]
        edges_pert = get_existing_edges(adj_pert)
        if len(edges_pert) >= 5:
            for _ in range(5):
                e = rng.choice(edges_pert)
                remove_edge(adj_pert, e[0], e[1])
                edges_pert.remove(e)
            
        # Re-repair (should be valid already, but just in case)
        repair_graph(adj_pert)
        
        # Re-search
        adj_pert, cur_edges_pert = local_search(adj_pert)
        
        if cur_edges_pert > best_edges:
            best_edges = cur_edges_pert
            best_adj = [a for a in adj_pert]

    if best_adj is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
