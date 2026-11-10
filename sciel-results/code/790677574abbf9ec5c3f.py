import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    n = 20
    total_possible = n * (n - 1) // 2  # 190

    # Precompute edge index mapping
    edge_indices = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_indices)
            edge_indices.append((u, v))
    m = len(edge_indices)

    # Precompute for each edge, which other edges form a K2,3 violation when both are present
    # Actually, K2,3 violation: two vertices u,v share 3+ common neighbors
    # This is a higher-order constraint. Let's think differently.
    # 
    # For simulated annealing, we need to compute the change in "energy" when flipping an edge.
    # Energy = -num_edges + penalty * (number of K2,3 violations)
    # A K2,3 violation exists for pair (u,v) if |N(u) ∩ N(v)| >= 3
    # 
    # When we add/remove edge (a,b), which pairs' common neighbor counts change?
    # - Pairs (a, w) for w != a,b: common neighbors change if b is neighbor of w
    # - Pairs (b, w) for w != a,b: common neighbors change if a is neighbor of w
    # - Pair (a, b): their common neighbors are N(a) ∩ N(b) excluding a,b themselves
    #
    # Let's use a different approach: track common neighbor counts for all pairs
    # and update them incrementally.

    def count_edges_from_adj(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True

    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
        return True

    def greedy_construction(adj=None):
        if adj is None:
            adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    return adj
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        return adj

    def fix_violations(adj):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = common.bit_count()
                    if cnt >= 3:
                        common_neighbors = [w for w in range(n) if (common >> w) & 1]
                        w = common_neighbors[0]
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        changed = True
                        break
                if changed:
                    break
        return adj

    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    # Simulated Annealing with NumPy vectorization
    def simulated_annealing(adj, max_iters=50000, T_start=10.0, T_end=0.01, cooling=0.999):
        """SA that tries to flip edges, accepting worse solutions probabilistically."""
        if adj is None:
            adj = [0] * n
        
        # Convert to numpy edge array for vectorized operations
        # edge_states[i] = 1 if edge i is present, 0 otherwise
        edge_states = np.zeros(m, dtype=np.int8)
        for i, (u, v) in enumerate(edge_indices):
            if (adj[u] >> v) & 1:
                edge_states[i] = 1
        
        # Build adjacency matrix from edge_states
        def build_adj_from_edges(es):
            adj = [0] * n
            for i, (u, v) in enumerate(edge_indices):
                if es[i]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            return adj
        
        # Compute current energy: -num_edges + penalty * violations
        # We'll track violations incrementally
        # For each pair (u,v), track common neighbor count
        # common_count[u,v] = |N(u) ∩ N(v)|
        
        def compute_common_counts(adj):
            cc = np.zeros((n, n), dtype=np.int32)
            for u in range(n):
                for v in range(u + 1, n):
                    cnt = (adj[u] & adj[v]).bit_count()
                    cc[u, v] = cnt
                    cc[v, u] = cnt
            return cc
        
        # Penalty for each pair: max(0, common_count - 2)
        # Total penalty = sum of penalties over all pairs
        
        def compute_energy(adj, cc):
            ne = count_edges(adj)
            penalty = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if cc[u, v] > 2:
                        penalty += (cc[u, v] - 2)
            return -ne + 10 * penalty  # Large penalty to discourage violations
        
        current_adj = adj[:]
        current_cc = compute_common_counts(current_adj)
        current_energy = compute_energy(current_adj, current_cc)
        best_energy = current_energy
        best_adj = current_adj[:]
        
        T = T_start
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random edge to flip
            idx = random.randint(0, m - 1)
            u, v = edge_indices[idx]
            
            # Compute delta energy if we flip this edge
            # If edge is present, removing it:
            #   num_edges decreases by 1
            #   For each w != u,v:
            #     if v in N(w): common(u,w) decreases by 1
            #     if u in N(w): common(v,w) decreases by 1
            #   common(u,v) decreases by 1 for each common neighbor... wait
            #   Actually, common(u,v) = |N(u) ∩ N(v)|. If we remove edge (u,v),
            #   u is no longer in N(v) and v is no longer in N(u), but that doesn't
            #   affect N(u) ∩ N(v) directly since u,v are not in their own neighborhoods.
            #   Wait, N(u) and N(v) don't include u or v themselves in a simple graph.
            #   So removing edge (u,v) doesn't change N(u) ∩ N(v).
            #   But it changes N(u) and N(v), which affects common(u,w) and common(v,w).
            
            is_present = (current_adj[u] >> v) & 1
            
            delta_energy = 0
            if is_present:
                # Removing edge (u,v)
                delta_energy += 1  # num_edges decreases, so -ne increases by 1... wait
                # energy = -ne + penalty. Removing edge: ne decreases by 1, so -ne increases by 1.
                # So delta_energy from edge count = +1
                # Now penalty changes:
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # common(u,w) changes if v in N(w)
                    if (current_adj[w] >> v) & 1:
                        old = current_cc[u, w]
                        new = old - 1
                        old_pen = max(0, old - 2)
                        new_pen = max(0, new - 2)
                        delta_energy += 10 * (new_pen - old_pen)
                    # common(v,w) changes if u in N(w)
                    if (current_adj[w] >> u) & 1:
                        old = current_cc[v, w]
                        new = old - 1
                        old_pen = max(0, old - 2)
                        new_pen = max(0, new - 2)
                        delta_energy += 10 * (new_pen - old_pen)
            else:
                # Adding edge (u,v)
                delta_energy -= 1  # ne increases, so -ne decreases by 1
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (current_adj[w] >> v) & 1:
                        old = current_cc[u, w]
                        new = old + 1
                        old_pen = max(0, old - 2)
                        new_pen = max(0, new - 2)
                        delta_energy += 10 * (new_pen - old_pen)
                    if (current_adj[w] >> u) & 1:
                        old = current_cc[v, w]
                        new = old + 1
                        old_pen = max(0, old - 2)
                        new_pen = max(0, new - 2)
                        delta_energy += 10 * (new_pen - old_pen)
            
            # Metropolis acceptance
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / max(T, 1e-10)):
                # Accept the flip
                if is_present:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                
                # Update common counts
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (current_adj[w] >> v) & 1 or (current_adj[w] >> u) & 1:
                        pass
                    # Re-compute affected pairs
                # For efficiency, recompute all common counts (n=20, so 190 pairs)
                current_cc = compute_common_counts(current_adj)
                
                current_energy = compute_energy(current_adj, current_cc)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = current_adj[:]
            
            T *= cooling
        
        return best_adj

    # Run multiple random restarts
    best_adj = None
    best_count = 0

    # Strategy 1: Random Greedy
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        adj = greedy_construction()
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Strategy 2: Start from seed (5 K4s) and improve
    def init_seed():
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    seed_adj = init_seed()
    seed_adj = local_search(seed_adj)
    cnt = count_edges(seed_adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = seed_adj[:]

    # Strategy 3: Simulated Annealing from various starting points
    if time.time() < deadline - 20.0:
        # SA from greedy result
        if best_adj is not None:
            sa_adj = simulated_annealing(best_adj[:], max_iters=30000)
            cnt = count_edges(sa_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = sa_adj[:]

    # Strategy 4: More random restarts
    restart = num_restarts
    while time.time() < deadline - 10.0:
        random.seed(restart * 98765 + 43210)
        adj = greedy_construction()
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        restart += 1

    # Final Polish: Try to add edges to best_adj
    if best_adj is not None:
        best_adj = local_search(best_adj)
        best_count = count_edges(best_adj)
        
        # Try random removals and re-additions
        for _ in range(10):
            if time.time() > deadline:
                break
            trial_adj = best_adj[:]
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (trial_adj[u] >> v) & 1:
                        edges_list.append((u, v))
            if not edges_list:
                continue
            u, v = random.choice(edges_list)
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            
            trial_adj = local_search(trial_adj)
            cnt = count_edges(trial_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = trial_adj[:]

    # Convert best_adj to edge list
    if best_adj is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
