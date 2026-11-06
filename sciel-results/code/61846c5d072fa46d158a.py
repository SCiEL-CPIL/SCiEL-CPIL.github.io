"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    import random
    start_time = time.time()
    DEADLINE = 118.0
    n = 31
    target_edges = 91
    
    # Polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                edges.add((i, j))
    
    edges = list(edges)
    
    # Build adjacency matrix for fast common-neighbor counting
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def count_c4(adj):
        """Count 4-cycles using matrix multiplication."""
        A2 = adj @ adj
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                total += A2[i, j] * A2[i, j]
        return total // 8
    
    def c4_delta_add_edge(adj, u, v):
        """Compute change in C4 count when adding edge (u,v)."""
        # New C4s created = number of common neighbors of u and v
        # Each common neighbor w creates a path u-w-v, and any other common neighbor w'
        # creates a 4-cycle u-w-v-w'-u
        common = int(np.sum(adj[u] & adj[v]))
        # Each pair of common neighbors forms a C4 with edge (u,v)
        return common * (common - 1) // 2
    
    def c4_delta_remove_edge(adj, u, v):
        """Compute change in C4 count when removing edge (u,v)."""
        # C4s destroyed = number of C4s that use edge (u,v)
        # A C4 uses edge (u,v) if there are two common neighbors w,w' such that
        # u-w-v-w'-u is a cycle, i.e., w and w' are both adjacent to u and v
        common = int(np.sum(adj[u] & adj[v]))
        return common * (common - 1) // 2
    
    if len(edges) < target_edges:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = set(edges)
        candidates = [p for p in all_pairs if p not in existing]
        edge_set = set(edges)
        
        for _ in range(target_edges - len(edges)):
            if time.time() - start_time > DEADLINE - 10:
                break
            best, best_cost = None, float('inf')
            for u, v in candidates:
                common = int(np.sum(adj[u] & adj[v]))
                if common < best_cost:
                    best_cost, best = common, (u, v)
            if best:
                edges.append(best)
                edge_set.add(best)
                candidates.remove(best)
                adj[best[0], best[1]] = 1
                adj[best[1], best[0]] = 1
    
    elif len(edges) > target_edges:
        # Remove edges that contribute most to C4
        for _ in range(len(edges) - target_edges):
            if time.time() - start_time > DEADLINE - 10:
                break
            best, best_cost = None, float('inf')
            for u, v in edges:
                common = int(np.sum(adj[u] & adj[v]))
                if common > best_cost:
                    best_cost, best = common, (u, v)
            if best:
                edges.remove(best)
                adj[best[0], best[1]] = 0
                adj[best[1], best[0]] = 0
    
    # Local search: try swapping edges to reduce C4 count
    # Use incremental delta computation for speed
    current_c4 = count_c4(adj)
    edge_set = set(edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [p for p in all_pairs if p not in edge_set]
    
    def swap_delta(adj, remove_e, add_e):
        """
        Compute the net change in C4 count for a swap: remove remove_e, add add_e.
        This is approximate but fast: we compute delta_add(add_e) + delta_remove(remove_e)
        ignoring interactions between the two changes.
        """
        u1, v1 = remove_e
        u2, v2 = add_e
        
        # Delta from removing edge (u1, v1): loses C4s that use this edge
        common_rem = int(np.sum(adj[u1] & adj[v1]))
        delta_rem = -common_rem * (common_rem - 1) // 2
        
        # Delta from adding edge (u2, v2): gains C4s that use this edge
        # But we need to account for the fact that edge (u1,v1) is being removed
        # If u2 or v2 is u1 or v1, the common neighbor count changes
        common_add = int(np.sum(adj[u2] & adj[v2]))
        # Adjust: if removing (u1,v1) affects common neighbors of (u2,v2)
        # This happens if u2==u1 or u2==v1 or v2==u1 or v2==v1
        adj_u2 = adj[u2].copy()
        adj_v2 = adj[v2].copy()
        # Simulate removal of (u1,v1)
        if u2 == u1 or u2 == v1:
            adj_u2[v1 if u2 == u1 else u1] = 0
        if v2 == u1 or v2 == v1:
            adj_v2[v1 if v2 == u1 else u1] = 0
        # Also if the removed edge connects a common neighbor
        if adj[u1, u2] == 1 and adj[v1, u2] == 1:
            pass  # u2 is common neighbor of u1,v1, already counted
        if adj[u1, v2] == 1 and adj[v1, v2] == 1:
            pass  # v2 is common neighbor of u1,v1, already counted
        
        # Recompute common neighbors after removal
        # For simplicity, use the original count as approximation
        delta_add = common_add * (common_add - 1) // 2
        
        return delta_add + delta_rem
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < DEADLINE - 10:
        improved = False
        best_swap = None
        best_delta = 0
        best_new_c4 = current_c4
        iteration += 1
        
        # Use a different random seed each iteration for diversity
        rng = random.Random(iteration * 12345 + 678)
        
        # Increase sample size over iterations to explore more
        sample_size_e = min(len(edges), 20 + iteration * 2)
        sample_size_ne = min(len(non_edges), 20 + iteration * 2)
        sample_edges = rng.sample(edges, sample_size_e)
        sample_non_edges = rng.sample(non_edges, sample_size_ne)
        
        for e1 in sample_edges:
            if time.time() - start_time > DEADLINE - 10:
                break
            u1, v1 = e1
            for e2 in sample_non_edges:
                u2, v2 = e2
                # Check if swap is valid (no duplicate edges)
                if (u1, v1) in edge_set and (u2, v2) not in edge_set:
                    delta = swap_delta(adj, (u1, v1), (u2, v2))
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = ((u1, v1), (u2, v2))
        
        if best_swap:
            e1, e2 = best_swap
            # Apply swap
            adj[e1[0], e1[1]] = 0
            adj[e1[1], e1[0]] = 0
            adj[e2[0], e2[1]] = 1
            adj[e2[1], e2[0]] = 1
            
            edges.remove(e1)
            edges.append(e2)
            edge_set.remove(e1)
            edge_set.add(e2)
            
            non_edges.remove(e2)
            non_edges.append(e1)
            
            # Recompute exact C4 count periodically to correct drift
            if iteration % 5 == 0:
                current_c4 = count_c4(adj)
            else:
                current_c4 += best_delta
            
            improved = True
    
    # Final verification
    current_c4 = count_c4(adj)
    
    # Try a few more full-scan swaps if time permits
    if time.time() - start_time < DEADLINE - 5:
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            best_swap = None
            best_delta = 0
            for e1 in edges:
                u1, v1 = e1
                for e2 in non_edges:
                    u2, v2 = e2
                    delta = swap_delta(adj, (u1, v1), (u2, v2))
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = ((u1, v1), (u2, v2))
            if best_swap:
                e1, e2 = best_swap
                adj[e1[0], e1[1]] = 0
                adj[e1[1], e1[0]] = 0
                adj[e2[0], e2[1]] = 1
                adj[e2[1], e2[0]] = 1
                edges.remove(e1)
                edges.append(e2)
                edge_set.remove(e1)
                edge_set.add(e2)
                non_edges.remove(e2)
                non_edges.append(e1)
                current_c4 = count_c4(adj)
            else:
                break
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
