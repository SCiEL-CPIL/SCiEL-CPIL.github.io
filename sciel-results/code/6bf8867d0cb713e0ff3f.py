import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_regular_6(vertices, adj):
        """
        vertices: tuple of 6 ints
        adj: list of ints
        Returns True if induced subgraph is regular.
        """
        mask = 0
        for v in vertices:
            mask |= (1 << v)
        deg0 = bin(adj[vertices[0]] & mask).count('1')
        for i in range(1, 6):
            if bin(adj[vertices[i]] & mask).count('1') != deg0:
                return False
        return True

    def check_validity(n, adj):
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                deg0 = bin(adj[comb[0]] & mask).count('1')
                is_reg = True
                for v in comb[1:]:
                    if bin(adj[v] & mask).count('1') != deg0:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def local_search(n, max_iters=5000):
        # Precompute all 6-sets
        all_6sets = list(itertools.combinations(range(n), 6))
        n_6sets = len(all_6sets)
        
        # Map 6-set to index
        # We need to find which 6-sets contain a specific edge (u,v)
        # Precompute list of indices of 6-sets containing each edge
        edge_to_sets = {}
        for i, comb in enumerate(all_6sets):
            for u in comb:
                for v in comb:
                    if u < v:
                        if (u, v) not in edge_to_sets:
                            edge_to_sets[(u, v)] = []
                        edge_to_sets[(u, v)].append(i)

        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Calculate initial bad count
        bad_status = [0] * n_6sets
        current_bad = 0
        for i, comb in enumerate(all_6sets):
            if is_regular_6(comb, adj):
                bad_status[i] = 1
                current_bad += 1
        
        if current_bad == 0:
            if check_validity(n, adj):
                return adj

        best_adj = adj[:]
        best_bad = current_bad

        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Get affected sets
            affected_indices = edge_to_sets.get((u, v), [])
            if not affected_indices:
                continue

            # Calculate change in bad count
            delta = 0
            # Check current status of affected sets
            # We will flip the edge, then check, then revert if worse
            
            # Flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            for idx in affected_indices:
                comb = all_6sets[idx]
                was_bad = bad_status[idx]
                is_now_bad = 1 if is_regular_6(comb, adj) else 0
                if was_bad != is_now_bad:
                    delta += (is_now_bad - was_bad)
                    bad_status[idx] = is_now_bad
            
            new_bad = current_bad + delta
            
            if new_bad <= current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_adj = adj[:]
                    if current_bad == 0:
                        if check_validity(n, adj):
                            return adj
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                # Revert status
                for idx in affected_indices:
                    # We need to know what the status was before
                    # This is tricky. We modified bad_status in place.
                    # We should store the old values or recalculate.
                    # Simpler: just recalculate the affected sets after reverting?
                    # Or store old values in a temp list.
                    pass # Logic error in revert. Let's fix.
        
        return None

    # The revert logic is flawed. Let's simplify.
    # Instead of in-place update and revert, we can just recalculate the delta
    # by checking the "before" and "after" states for the affected sets.
    
    def local_search_v2(n, max_iters=5000):
        all_6sets = list(itertools.combinations(range(n), 6))
        n_6sets = len(all_6sets)
        
        edge_to_sets = {}
        for i, comb in enumerate(all_6sets):
            for u in comb:
                for v in comb:
                    if u < v:
                        if (u, v) not in edge_to_sets:
                            edge_to_sets[(u, v)] = []
                        edge_to_sets[(u, v)].append(i)

        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        bad_status = [0] * n_6sets
        current_bad = 0
        for i, comb in enumerate(all_6sets):
            if is_regular_6(comb, adj):
                bad_status[i] = 1
                current_bad += 1
        
        if current_bad == 0:
            if check_validity(n, adj):
                return adj

        best_adj = adj[:]
        best_bad = current_bad

        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            affected_indices = edge_to_sets.get((u, v), [])
            if not affected_indices:
                continue

            # Calculate delta by simulating the flip
            delta = 0
            for idx in affected_indices:
                comb = all_6sets[idx]
                was_bad = bad_status[idx]
                
                # Temporarily flip to check? No, just calculate is_regular_6 with the flipped edge.
                # We can't easily "temporarily flip" without modifying adj.
                # Let's modify adj, check, and if we decide to revert, we revert adj and status.
                # To avoid complex status reverting, we can just recalculate the status of affected sets
                # after the decision.
                
                # Let's just do:
                # 1. Flip adj
                # 2. Recalculate status for affected sets
                # 3. Calculate new_bad
                # 4. If new_bad > current_bad, revert adj and revert status (by recalculating again or storing old)
                
                pass # This approach is messy.
            
            # Alternative: Just recalculate the whole bad count?
            # 38760 checks per iteration.
            # 5000 iterations -> 193 million checks.
            # In Python, ~10-20 million ops/sec? 10-20 seconds.
            # Too slow for 5000 iterations.
            # But we can reduce iterations.
            # Or use the incremental method correctly.
            
            # Let's try the incremental method with careful bookkeeping.
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            old_statuses = []
            new_statuses = []
            for idx in affected_indices:
                old_statuses.append(bad_status[idx])
                comb = all_6sets[idx]
                new_statuses.append(1 if is_regular_6(comb, adj) else 0)
            
            delta = sum(new_statuses) - sum(old_statuses)
            new_bad = current_bad + delta
            
            if new_bad <= current_bad:
                current_bad = new_bad
                # Update bad_status
                for k, idx in enumerate(affected_indices):
                    bad_status[idx] = new_statuses[k]
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_adj = adj[:]
                    if current_bad == 0:
                        if check_validity(n, adj):
                            return adj
            else:
                # Revert adj
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                # Status remains unchanged because we reverted to old state
                # No need to update bad_status as it matches the reverted adj state
        
        return None

    # Use local_search_v2
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        adj = local_search_v2(20, max_iters=3000)
        if adj:
            return {'n': 20, 'edges': [[u, v] for u in range(20) for v in range(u + 1, 20) if (adj[u] >> v) & 1]}

    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        adj = local_search_v2(19, max_iters=3000)
        if adj:
            return {'n': 19, 'edges': [[u, v] for u in range(19) for v in range(u + 1, 19) if (adj[u] >> v) & 1]}

    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        adj = local_search_v2(18, max_iters=3000)
        if adj:
            return {'n': 18, 'edges': [[u, v] for u in range(18) for v in range(u + 1, 18) if (adj[u] >> v) & 1]}

    edges = [[0,1], [1,2], [2,3], [3,4], [4,5]]
    return {'n': 6, 'edges': edges}
