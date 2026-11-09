import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin

    # Base: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    base_n = q

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def count_violations(n, adj):
        """Count K5 and independent 5 violations using bitset operations."""
        viol = 0
        # Precompute neighbor lists for faster access
        neigh = []
        for i in range(n):
            mask = adj[i]
            lst = []
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                lst.append(j)
                mask ^= lsb
            neigh.append(lst)
        
        # Count K5: for each vertex i, count K4 in N(i)
        # K4 in a set S: count triangles in the induced subgraph on S that contain a specific vertex
        # More efficient: for each i, for each pair (j,k) in N(i), count common neighbors of j,k in N(i)
        # that form edges with both j and k -> that gives K4 containing i,j,k,l
        # Actually, let's count K5 directly:
        # A K5 exists iff there's a 5-clique.
        # For n up to ~45, C(n,5) is manageable with bitset tricks.
        
        # Alternative: count K5 by iterating over 5-tuples with early exit via bitsets
        # For efficiency, use the fact that a K5 requires all 10 edges.
        
        # Let's use a smarter approach:
        # For each i, consider its neighbors. A K5 containing i requires a K4 in N(i).
        # K4 in N(i): for each pair (j,k) in N(i), let common = N(j) & N(k) & N(i)
        # A K4 in N(i) exists iff there's an edge (l,m) in common where l,m are both in common
        # and l-m edge exists.
        
        # Actually, let me just do the direct check but optimized with bitsets:
        for i in range(n):
            # Get neighbors of i as a bitmask
            ni = adj[i]
            # For each pair (j, k) in neighbors of i
            # j and k must be in ni
            mask_j = ni
            while mask_j:
                lsb_j = mask_j & (-mask_j)
                j = lsb_j.bit_length() - 1
                mask_j ^= lsb_j
                if j <= i:
                    continue
                # For each k in neighbors of i, k > j
                mask_k = ni & (ni >> (j + 1)) << (j + 1)  # neighbors > j
                # Actually simpler:
                mask_k = ni
                while mask_k:
                    lsb_k = mask_k & (-mask_k)
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    if k <= j:
                        continue
                    # j and k are both neighbors of i
                    # Check if j-k edge exists
                    if not ((adj[j] >> k) & 1):
                        continue
                    # Now find l, m in N(i) ∩ N(j) ∩ N(k) with l < m, l > k, and l-m edge
                    common = ni & adj[j] & adj[k]
                    common &= ~((1 << (k + 1)) - 1)  # only vertices > k
                    # Find edges in common
                    mask_l = common
                    found_k4 = False
                    while mask_l and not found_k4:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        # Find m in common > l with l-m edge
                        mask_m = common & adj[l]
                        mask_m &= ~((1 << (l + 1)) - 1)  # only vertices > l
                        if mask_m:
                            found_k4 = True
                            # Found K4 in N(i), so K5 exists
                            viol += 1
                            # We can break early for this i since we just need to know if any K5 exists
                            # But we're counting, so let's continue to count all
                            # Actually for validity check, we just need to know if viol > 0
                            # Let's just return 1 if any found for speed
                            return 1
                    # If we get here, no K4 in N(i) for this (j,k) pair
                    # Continue
            # End for j
        # End for i - no K5 found
        
        # Now check independent 5: similar but with non-edges
        # A independent 5 means 5 vertices with no edges between them.
        # For each i, consider non-neighbors of i (excluding i itself).
        # Non-neighbors of i: all vertices except i and N(i)
        nonni = ((1 << n) - 1) & ~(adj[i] | (1 << i))
        
        for i in range(n):
            mask_j = nonni
            while mask_j:
                lsb_j = mask_j & (-mask_j)
                j = lsb_j.bit_length() - 1
                mask_j ^= lsb_j
                if j <= i:
                    continue
                # j is a non-neighbor of i
                mask_k = nonni
                while mask_k:
                    lsb_k = mask_k & (-mask_k)
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    if k <= j:
                        continue
                    # k is a non-neighbor of i, k > j
                    # Check if j-k non-edge
                    if (adj[j] >> k) & 1:
                        continue
                    # Find l, m in nonni ∩ nonN(j) ∩ nonN(k) with l > k, l-m non-edge
                    # nonN(j) = all except j and N(j)
                    nonnj = ((1 << n) - 1) & ~(adj[j] | (1 << j))
                    nonnk = ((1 << n) - 1) & ~(adj[k] | (1 << k))
                    common = nonni & nonnj & nonnk
                    common &= ~((1 << (k + 1)) - 1)  # only vertices > k
                    # Find l, m in common with l < m and l-m non-edge
                    mask_l = common
                    found_ind4 = False
                    while mask_l and not found_ind4:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        # Find m in common > l with l-m non-edge
                        mask_m = common & ~adj[l]
                        mask_m &= ~((1 << (l + 1)) - 1)  # only vertices > l
                        if mask_m:
                            found_ind4 = True
                            viol += 1
                            return 1
        return 0

    def is_valid(n, adj):
        """Check if graph has no K5 and no independent 5."""
        return count_violations(n, adj) == 0

    # Faster local search with incremental updates
    def local_search(n, adj, time_limit, max_iters=100000):
        """Local search to find a valid (5,5)-free graph."""
        current_adj = adj[:]
        
        # Precompute for faster violation delta
        # When flipping edge (u,v), the change in violations is complex.
        # For simplicity, use full recount but with early exit.
        
        def count_viol_fast(n, adj):
            """Fast violation count with early exit."""
            # Check K5
            for i in range(n):
                ni = adj[i]
                # Get neighbor list
                neighbors = []
                mask = ni
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    neighbors.append(j)
                    mask ^= lsb
                
                # Count K4 in neighbors
                for idx_j in range(len(neighbors)):
                    j = neighbors[idx_j]
                    for idx_k in range(idx_j + 1, len(neighbors)):
                        k = neighbors[idx_k]
                        if not ((adj[j] >> k) & 1):
                            continue
                        # j and k are connected, both neighbors of i
                        # Find l, m in N(i) ∩ N(j) ∩ N(k) with l-m edge
                        common = ni & adj[j] & adj[k]
                        # Remove i, j, k
                        common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                        # Find an edge in common
                        mask_l = common
                        while mask_l:
                            lsb_l = mask_l & (-mask_l)
                            l = lsb_l.bit_length() - 1
                            mask_l ^= lsb_l
                            # Check if l has a neighbor in common > l
                            mask_m = common & adj[l]
                            mask_m &= ~((1 << (l + 1)) - 1)
                            if mask_m:
                                return 1  # Found K5
            # Check independent 5
            for i in range(n):
                nonni = ((1 << n) - 1) & ~(adj[i] | (1 << i))
                non_neighbors = []
                mask = nonni
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    non_neighbors.append(j)
                    mask ^= lsb
                
                for idx_j in range(len(non_neighbors)):
                    j = non_neighbors[idx_j]
                    for idx_k in range(idx_j + 1, len(non_neighbors)):
                        k = non_neighbors[idx_k]
                        if (adj[j] >> k) & 1:
                            continue  # j-k edge exists, not independent
                        # Find l, m in nonN(i) ∩ nonN(j) ∩ nonN(k) with l-m non-edge
                        nonnj = ((1 << n) - 1) & ~(adj[j] | (1 << j))
                        nonnk = ((1 << n) - 1) & ~(adj[k] | (1 << k))
                        common = nonni & nonnj & nonnk
                        common &= ~((1 << (k + 1)) - 1)  # vertices > k
                        mask_l = common
                        while mask_l:
                            lsb_l = mask_l & (-mask_l)
                            l = lsb_l.bit_length() - 1
                            mask_l ^= lsb_l
                            # Check if l has a non-neighbor in common > l
                            mask_m = common & ~adj[l]
                            mask_m &= ~((1 << (l + 1)) - 1)
                            if mask_m:
                                return 1  # Found independent 5
            return 0
        
        best_adj = current_adj[:]
        best_viol = count_viol_fast(n, best_adj)
        
        if best_viol == 0:
            return best_adj
        
        no_improve_count = 0
        iter_count = 0
        
        while time.time() < time_limit and iter_count < max_iters:
            iter_count += 1
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            new_viol = count_viol_fast(n, current_adj)
            
            if new_viol < best_viol:
                best_viol = new_viol
                best_adj = current_adj[:]
                no_improve_count = 0
                if best_viol == 0:
                    return best_adj
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                no_improve_count += 1
            
            # Restart if stuck
            if no_improve_count > 5000:
                # Perturb: flip a few random edges
                for _ in range(5):
                    pu = random.randint(0, n - 1)
                    pv = random.randint(0, n - 1)
                    if pu == pv:
                        continue
                    if pu > pv:
                        pu, pv = pv, pu
                    current_adj[pu] ^= (1 << pv)
                    current_adj[pv] ^= (1 << pu)
                no_improve_count = 0
        
        return best_adj if best_viol == 0 else None

    # Try to grow from Paley 37
    # Strategy: add one vertex at a time, connecting to a subset of existing vertices
    
    best_result = (base_n, base_edges)
    
    # First, verify the base is valid
    base_adj = build_adj(base_n, base_edges)
    if is_valid(base_n, base_adj):
        best_result = (base_n, base_edges)
    else:
        # Should not happen, but fallback
        return base_n, base_edges
    
    # Try growing to 38, 39, 40
    for target_n in [38, 39, 40, 41]:
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Multiple attempts with different random seeds for new vertex connections
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize: start with Paley 37, add new vertices
            init_adj = base_adj[:]
            init_adj.extend([0] * (target_n - base_n))
            
            # Connect new vertices to existing ones
            for v in range(base_n, target_n):
                # Connect to ~half of existing vertices
                for u in range(v):
                    if random.random() < 0.5:
                        init_adj[u] |= (1 << v)
                        init_adj[v] |= (1 << u)
            
            # Run local search
            time_slice = min(40.0, deadline - time.time() - 10)
            if time_slice < 5:
                break
            end_time = time.time() + time_slice
            
            result = local_search(target_n, init_adj, end_time)
            
            if result is not None:
                # Valid (5,5)-free graph found
                edges = adj_to_edges(target_n, result)
                return target_n, edges
        
        # If we couldn't find valid for target_n, try next
    
    # Fallback to seed
    return best_result
