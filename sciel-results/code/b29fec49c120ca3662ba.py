"""Improved K4-free graph construction for n=18 with fast bitmask evaluation."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 18
    random.seed(42)
    
    # Bitmask helpers
    def build_adj_mask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_k4_free_mask(adj):
        """Check K4-freeness using bitmasks."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # For each common neighbor k > j, check if i,j,k have common neighbor l > k
                    m = common
                    while m:
                        k_bit = m & (-m)
                        k = k_bit.bit_length() - 1
                        m ^= k_bit
                        if k <= j:
                            continue
                        if adj[i] & adj[j] & adj[k]:
                            # i,j,k form triangle; check if any l in (i,j,k) common neighborhood
                            # Actually need: exists l such that l adjacent to i,j,k
                            # l must be different from i,j,k
                            common_ijk = adj[i] & adj[j] & adj[k]
                            # Remove i,j,k from common_ijk
                            common_ijk &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                            if common_ijk:
                                return False
        return True
    
    def max_tf_size_exact(adj, time_limit=1.0):
        """Exact branch and bound for max triangle-free induced subgraph size."""
        best = [0]
        # Precompute: for each pair (u,v) that are adjacent, common neighbors
        # A set S is triangle-free iff no three vertices in S are mutually adjacent
        # Equivalent: for every edge (u,v) in S, u and v have no common neighbor in S
        
        # We'll do branch and bound: try to include/exclude vertices
        # Order vertices by degree (descending) for better pruning
        
        def rec(mask, size, excluded):
            if time.time() > time_limit:
                return
            if size + popcount(mask) <= best[0]:
                return
            if mask == 0:
                if size > best[0]:
                    best[0] = size
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with most constraints
            m = mask
            # Pick first vertex
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            
            # Option 1: include v
            # Check if adding v creates a triangle with existing included vertices
            # Actually, we're building the set incrementally, so we need to check
            # if v is compatible with all pairs in the current set
            # But we don't track the current set explicitly...
            
            # Better approach: track the set as a bitmask
            # Let me restructure
            
            pass
        
        # Simpler: use the approach from inspiration 3
        # Check all subsets of size k from high to low
        # For k=12: C(18,12) = 18564, each check is O(n^2) with bitmasks = fast
        
        def is_tf_subset(subset_mask):
            """Check if vertex set (bitmask) is triangle-free."""
            # For each edge (u,v) in the subset, check if they share a common neighbor in the subset
            m = subset_mask
            while m:
                u_bit = m & (-m)
                u = u_bit.bit_length() - 1
                m ^= u_bit
                # Neighbors of u within the subset
                neigh = adj[u] & subset_mask
                while neigh:
                    v_bit = neigh & (-neigh)
                    v = v_bit.bit_length() - 1
                    neigh ^= v_bit
                    if v < u:
                        continue
                    # Check if u and v share a common neighbor w in the subset
                    # w != u, w != v
                    common = adj[u] & adj[v] & subset_mask
                    common &= ~(1 << u) & ~(1 << v)
                    if common:
                        return False
            return True
        
        # Check sizes from 12 down to 1
        # If we find a TF set of size 12, return 12 (or higher, but we check descending)
        # Actually we want the MAX, so check from high to low, return first found
        
        import itertools
        
        # Check size 12 first (most common case for K6,6,6)
        for k in range(12, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_tf_subset(mask):
                    return k
            if time.time() > time_limit:
                break
        
        return best[0] if best[0] > 0 else 1
    
    def max_tf_size_fast(adj):
        """Fast evaluation: check if any 12-set is TF. If yes, score >= 12.
        If no, check 11-sets. This is much faster than exact for typical cases."""
        import itertools
        
        def is_tf_subset(subset_mask):
            m = subset_mask
            while m:
                u_bit = m & (-m)
                u = u_bit.bit_length() - 1
                m ^= u_bit
                neigh = adj[u] & subset_mask
                while neigh:
                    v_bit = neigh & (-neigh)
                    v = v_bit.bit_length() - 1
                    neigh ^= v_bit
                    if v < u:
                        continue
                    common = adj[u] & adj[v] & subset_mask
                    common &= ~(1 << u) & ~(1 << v)
                    if common:
                        return False
            return True
        
        # Check if any 12-set is TF
        for subset in itertools.combinations(range(n), 12):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_tf_subset(mask):
                return 12
        
        # No 12-set is TF, check 11-sets
        for subset in itertools.combinations(range(n), 11):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_tf_subset(mask):
                return 11
        
        # Check 10-sets
        for subset in itertools.combinations(range(n), 10):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_tf_subset(mask):
                return 10
        
        return 9  # Conservative lower bound
    
    # Initial graph: K6,6,6 (complete tripartite)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edge_set = set()
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edge_set.add((min(a,b), max(a,b)))
    
    adj = build_adj_mask(edge_set)
    
    # Verify K4-free
    if not is_k4_free_mask(adj):
        # Should not happen for K6,6,6, but fallback
        return [[a, b] for i in range(3) for j in range(i+1, 3)
                for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    current_score = max_tf_size_fast(adj)  # Should be 12
    best_score = current_score
    best_edges = set(edge_set)
    
    # Local search: try to add edges within parts (reduces TF sets) and remove some inter-part edges
    # Key: adding intra-part edges can reduce max TF size, but must maintain K4-freeness
    
    def try_add_edge(u, v, edge_set, adj):
        """Try adding edge (u,v), return True if K4-free maintained."""
        u, v = min(u,v), max(u,v)
        if (u, v) in edge_set:
            return False
        # Check if adding (u,v) creates a K4
        # Need w, z such that all 6 edges among {u,v,w,z} exist
        # w must be adjacent to both u and v
        common_uv = adj[u] & adj[v]
        common_uv &= ~(1 << u) & ~(1 << v)
        # For each w in common_uv, check if there's z adjacent to u, v, w
        m = common_uv
        while m:
            w_bit = m & (-m)
            w = w_bit.bit_length() - 1
            m ^= w_bit
            # z must be adjacent to u, v, w
            common_uvw = adj[u] & adj[v] & adj[w]
            common_uvw &= ~(1 << u) & ~(1 << v) & ~(1 << w)
            if common_uvw:
                return False
        return True
    
    # Local search loop
    all_possible_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible_edges.add((i, j))
    
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Strategy: try adding a random non-edge
        non_edges = all_possible_edges - edge_set
        if not non_edges:
            break
        
        # Prioritize intra-part edges (more likely to reduce TF sets)
        intra_edges = []
        for (u, v) in non_edges:
            # Check if u and v are in the same part
            part_u = u // 6
            part_v = v // 6
            if part_u == part_v:
                intra_edges.append((u, v))
        
        # 70% chance pick intra-part, 30% any non-edge
        if intra_edges and random.random() < 0.7:
            u, v = random.choice(intra_edges)
        else:
            u, v = random.choice(tuple(non_edges))
        
        if not try_add_edge(u, v, edge_set, adj):
            continue
        
        # Add the edge
        edge_set.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Evaluate
        new_score = max_tf_size_fast(adj)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(edge_set)
        
        # Accept if not worse (to explore), revert if significantly worse
        if new_score > current_score + 1:
            # Revert
            edge_set.discard((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            current_score = new_score
    
    # Final verification
    final_adj = build_adj_mask(best_edges)
    if not is_k4_free_mask(final_adj):
        # Fallback
        return [[a, b] for i in range(3) for j in range(i+1, 3)
                for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
