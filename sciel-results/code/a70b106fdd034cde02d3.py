import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Q3 structure: 3-cube with 8 vertices, 12 edges
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))
    
    # Precompute Q3 adjacency as bitmasks
    q3_adj_mask = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj_mask[u] |= (1 << v)
        q3_adj_mask[v] |= (1 << u)
    
    # All 8-vertex subsets of 0..15
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    def find_one_q3_fast(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        """
        for S in all_8_subsets:
            S_set = set(S)
            
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                mask = adj_mask[v]
                deg = 0
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Backtracking with bitmask pruning
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S
            
            def dfs(idx):
                if time.time() > deadline - 0.5:
                    return False
                
                if idx == 8:
                    return True
                
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                for v in S:
                    if (used >> v) & 1:
                        continue
                    
                    valid = True
                    for nb in assigned_nbs:
                        if not (adj_mask[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                
                return False
            
            if dfs(0):
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return (True, edges)
        
        return (False, None)
    
    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    # Strategy: Greedy Removal + Swap-based Local Search
    
    # 1. Initialize with K16
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_edges = _count_edges(adj)
    
    # 2. Greedy Removal to get a Q3-free graph
    rng = random.Random(42)
    while True:
        if time.time() > deadline - 40.0:
            break
        found, q3_edges = find_one_q3_fast(adj)
        if not found:
            break
        # Remove a random edge from the Q3
        rng.shuffle(q3_edges)
        u, v = q3_edges[0]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_edges -= 1
        
    best_adj = adj[:]
    best_m = current_edges
    
    # 3. Local Search with Swaps
    # Try to add edges. If it creates a Q3, try to swap out another edge from that Q3.
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Try to add each missing edge
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 10.0:
                    break
                if (adj[u] >> v) & 1:
                    continue
                
                # Try adding (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                found, q3_edges = find_one_q3_fast(adj)
                
                if not found:
                    # Success!
                    current_edges += 1
                    if current_edges > best_m:
                        best_m = current_edges
                        best_adj = adj[:]
                    improved = True
                    break # Restart the loop
                else:
                    # Failure. Try to repair by swapping.
                    success_swap = False
                    for a, b in q3_edges:
                        if (a == u and b == v) or (a == v and b == u):
                            continue
                            
                        # Try swapping: Remove (a, b), keep (u, v)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        found2, _ = find_one_q3_fast(adj)
                        
                        if not found2:
                            # Swap successful!
                            success_swap = True
                            break
                        else:
                            # Restore (a, b)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            
                    if success_swap:
                        # We changed the graph. 
                        # Accept this swap.
                        # We don't increase edge count, but we changed structure.
                        # Let's continue to try adding other edges.
                        pass
                    else:
                        # No swap worked. Revert the addition of (u, v)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        
    # 4. Final Polish: Try to add edges again in random order
    if time.time() < deadline - 5.0:
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    candidates.append((u, v))
        
        rng.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 2.0:
                break
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            found, _ = find_one_q3_fast(adj)
            if found:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_edges += 1
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                
    # Return the best graph found
    if best_m > current_edges:
        adj = best_adj[:]
        
    return get_edge_list(adj)
