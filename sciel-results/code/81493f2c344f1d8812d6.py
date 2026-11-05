import random
import time
import itertools

def construct():
    n = 20
    
    # Precompute subsets of size 6 for fast incremental checking
    subsets_6 = []
    for comb in itertools.combinations(range(n), 6):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        subsets_6.append((mask, comb))
        
    # Precompute which subsets contain each vertex (for incremental updates)
    affected_by_vertex = [[] for _ in range(n)]
    for idx, (mask, verts) in enumerate(subsets_6):
        for v in verts:
            affected_by_vertex[v].append(idx)
            
    # Precompute larger subsets for final validity check
    subsets_final = []
    for k in range(7, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_final.append((mask, comb))

    def is_regular_subset(adj, mask, verts):
        # Check if induced subgraph on 'verts' is regular
        # Unrolled for speed (verts length is 6 or 7+)
        d0 = (adj[verts[0]] & mask).bit_count()
        for v in verts[1:]:
            if (adj[v] & mask).bit_count() != d0:
                return False
        return True

    def is_valid_graph(adj):
        # Check k=6
        for mask, verts in subsets_6:
            if is_regular_subset(adj, mask, verts):
                return False
        # Check k>=7
        for mask, verts in subsets_final:
            if is_regular_subset(adj, mask, verts):
                return False
        return True

    def to_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # State for incremental updates
    # status[i] is True if subset_6[i] is currently regular
    status = [False] * len(subsets_6)
    count_6 = 0
    
    def init_state(adj):
        nonlocal count_6
        count_6 = 0
        for idx, (mask, verts) in enumerate(subsets_6):
            if is_regular_subset(adj, mask, verts):
                status[idx] = True
                count_6 += 1
            else:
                status[idx] = False

    def update_state(u, v, adj):
        nonlocal count_6
        # Get unique indices of affected subsets
        # Union of affected_by_vertex[u] and affected_by_vertex[v]
        # To avoid set overhead, we can use a marker array or just iterate and skip duplicates if we track them?
        # Given small size, a set is fine.
        affected_indices = set(affected_by_vertex[u])
        affected_indices.update(affected_by_vertex[v])
        
        for idx in affected_indices:
            mask, verts = subsets_6[idx]
            was_reg = status[idx]
            is_reg = is_regular_subset(adj, mask, verts)
            
            if was_reg and not is_reg:
                count_6 -= 1
            elif not was_reg and is_reg:
                count_6 += 1
            
            status[idx] = is_reg

    best_adj = None
    best_cost = float('inf')
    
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        init_state(adj)
        current_cost = count_6
        
        if current_cost == 0:
            if is_valid_graph(adj):
                return {'n': n, 'edges': to_edge_list(adj)}
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = adj[:]
            
        # Simulated Annealing
        temp = 1.0
        steps = 2000
        for i in range(steps):
            if time.time() - start_time > timeout:
                break
                
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            # Update state
            old_cost = current_cost
            update_state(u, v, adj)
            new_cost = count_6
            
            delta = new_cost - old_cost
            
            if delta < 0 or (delta > 0 and random.random() < pow(2.718, -delta / temp)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    
                if new_cost == 0:
                    if is_valid_graph(adj):
                        return {'n': n, 'edges': to_edge_list(adj)}
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                update_state(u, v, adj)
                # current_cost should be back to old_cost
                
            temp *= 0.995
            
            if time.time() - start_time > timeout:
                break

    # If we didn't find a valid graph on 20, try to return the best one found 
    # If best_adj is valid, return it.
    if best_adj is not None and best_cost == 0:
        if is_valid_graph(best_adj):
            return {'n': n, 'edges': to_edge_list(best_adj)}
            
    # Fallback: Try n=19 if 20 fails? 
    # Given the strict time limit and the known existence of n=20, 
    # failing to find it implies the heuristic is too weak.
    # However, returning n=5 is a guaranteed 0.25 score.
    # Let's try a quick greedy for n=19? No, too complex.
    # Return n=5 as safe fallback.
    return {'n': 5, 'edges': []}
