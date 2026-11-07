"""Packing trees T1..T20 into K20 with improved backtracking and bitset optimization."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin below 150s limit
    
    N = 20
    trees = instance_trees(N, 611)
    
    # Precompute tree data
    tree_data = []
    for idx in range(N):
        k = idx + 1
        edges = trees[idx]
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_data.append({'k': k, 'edges': edges, 'adj': adj, 'deg': deg})
    
    # Bitmask for host vertices (20 bits fit in int)
    all_hosts_mask = (1 << N) - 1
    
    # Precompute edge bitmasks for K20
    # edge_mask[u][v] = bitmask of edge (u,v)
    # Actually, let's use a set of used edges as frozenset of pairs for simplicity,
    # but optimize the inner loop with precomputed adjacency
    
    # For each host vertex, precompute list of neighbors
    host_adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            host_adj[i].append(j)
            host_adj[j].append(i)
    
    def time_ok():
        return time.time() < deadline
    
    def embed_tree(tree_idx, used_edges, host_used_mask, rng, node_budget=200000):
        """
        Embed tree_idx into K20 using backtracking.
        Returns mapping (list of host vertices) or None.
        """
        td = tree_data[tree_idx]
        k = td['k']
        edges = td['edges']
        adj = td['adj']
        deg = td['deg']
        
        if k == 1:
            # Find any unused host
            for v in range(N):
                if not (host_used_mask >> v) & 1:
                    return [v]
            return None
        
        if k == 2:
            # Find any unused edge
            for u in range(N):
                if (host_used_mask >> u) & 1:
                    continue
                for v in range(u + 1, N):
                    if (host_used_mask >> v) & 1:
                        continue
                    if (u, v) not in used_edges:
                        return [u, v]
            return None
        
        # Variable ordering: most constrained first (highest degree)
        var_order = sorted(range(k), key=lambda v: (-deg[v], v))
        
        # Precompute for each tree vertex, which other tree vertices it's adjacent to
        # and their positions in var_order
        var_pos = [0] * k
        for i, v in enumerate(var_order):
            var_pos[v] = i
        
        # Precompute adjacency in terms of var_order indices
        adj_in_order = [[] for _ in range(k)]
        for v in range(k):
            for w in adj[v]:
                adj_in_order[v].append(w)
        
        mapping = [-1] * k
        nodes = [0]
        
        # Remaining degree for host vertices
        host_rem_deg = [0] * N
        for (u, v) in used_edges:
            host_rem_deg[u] -= 1
            host_rem_deg[v] -= 1
        for h in range(N):
            if host_rem_deg[h] < 0:
                host_rem_deg[h] = 0
        
        def backtrack(idx):
            if not time_ok():
                return False
            if nodes[0] > node_budget:
                return False
            if idx == k:
                return True
            
            tv = var_order[idx]
            
            # Get candidate host vertices
            cands = []
            for h in range(N):
                if (host_used_mask >> h) & 1:
                    continue
                
                # Check edges to already-assigned tree neighbors
                ok = True
                for w in adj[tv]:
                    mw = mapping[w]
                    if mw != -1:
                        a, b = (h, mw) if h < mw else (mw, h)
                        if (a, b) in used_edges:
                            ok = False
                            break
                
                if not ok:
                    continue
                
                # Pruning: check if h has enough remaining capacity
                # Count unassigned neighbors
                unassigned_count = 0
                for w in adj[tv]:
                    if mapping[w] == -1:
                        unassigned_count += 1
                
                if host_rem_deg[h] < unassigned_count:
                    continue
                
                cands.append(h)
            
            if not cands:
                return False
            
            # Sort candidates: for high-degree tree vertices, prefer hosts with more remaining degree
            if deg[tv] >= 3:
                cands.sort(key=lambda h: -host_rem_deg[h])
            else:
                rng.shuffle(cands)
            
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_budget:
                    return False
                
                mapping[tv] = h
                new_host_mask = host_used_mask | (1 << h)
                
                # Add edges to used_edges
                added_edges = []
                for w in adj[tv]:
                    mw = mapping[w]
                    if mw != -1:
                        a, b = (h, mw) if h < mw else (mw, h)
                        used_edges.add((a, b))
                        added_edges.append((a, b))
                        host_rem_deg[h] -= 1
                        host_rem_deg[mw] -= 1
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for (a, b) in added_edges:
                    used_edges.discard((a, b))
                    host_rem_deg[a] += 1
                    host_rem_deg[b] += 1
                
                mapping[tv] = -1
            
            return False
        
        if backtrack(0):
            return mapping
        return None
    
    def evaluate_order(order, rng, node_budget=200000):
        """Try to embed all trees in the given order."""
        used_edges = set()
        host_used_mask = 0
        result = [None] * N
        total_edges = 0
        
        for idx in order:
            if not time_ok():
                break
            
            k = idx + 1
            mapping = embed_tree(idx, used_edges, host_used_mask, rng, node_budget)
            
            if mapping is None:
                break
            
            result[idx] = [int(v) for v in mapping]
            host_used_mask |= 0  # We track via mapping, not a single mask for all trees
            
            # Actually, we need to track which hosts are used by ANY tree
            # Let me fix this: we need a global host_used_mask
            
            for u, v in trees[idx]:
                a, b = (mapping[u], mapping[v]) if mapping[u] < mapping[v] else (mapping[v], mapping[u])
                used_edges.add((a, b))
                total_edges += 1
        
        return result, total_edges
    
    # Better approach: track host_used_mask globally
    def evaluate_order_v2(order, rng, node_budget=200000):
        """Try to embed all trees in the given order with proper host tracking."""
        used_edges = set()
        host_used_mask = 0
        result = [None] * N
        total_edges = 0
        
        for idx in order:
            if not time_ok():
                break
            
            mapping = embed_tree(idx, used_edges, host_used_mask, rng, node_budget)
            
            if mapping is None:
                break
            
            result[idx] = [int(v) for v in mapping]
            
            # Update host_used_mask
            for h in mapping:
                host_used_mask |= (1 << h)
            
            # Add edges to used_edges
            for u, v in trees[idx]:
                a, b = (mapping[u], mapping[v]) if mapping[u] < mapping[v] else (mapping[v], mapping[u])
                used_edges.add((a, b))
                total_edges += 1
        
        return result, total_edges
    
    # Strategy: try multiple orderings with restarts
    best_result = None
    best_val = -1
    
    # Order 1: largest first
    orders = []
    orders.append(list(range(N - 1, -1, -1)))
    
    # Order 2: random shuffles with largest first
    master_rng = Random(42)
    for _ in range(50):
        o = list(range(N - 1, -1, -1))
        master_rng.shuffle(o[1:])  # Keep T20 first, shuffle rest
        orders.append(o)
    
    # Order 3: fully random
    for _ in range(200):
        o = list(range(N))
        master_rng.shuffle(o)
        orders.append(o)
    
    attempt = 0
    for o in orders:
        if not time_ok():
            break
        
        attempt += 1
        rng = Random(attempt)
        node_budget = 300000 if attempt <= 20 else 100000
        
        r, v = evaluate_order_v2(o, rng, node_budget)
        
        if v > best_val:
            best_val = v
            best_result = r
        
        if best_val == 190:
            break
    
    # If we found a good solution, return it
    if best_result is not None and best_val > 0:
        return best_result
    
    # Fallback: just embed T20
    result = [None] * N
    used_edges = set()
    host_used_mask = 0
    rng = Random(0)
    mapping = embed_tree(19, used_edges, host_used_mask, rng, 500000)
    if mapping is not None:
        result[19] = [int(v) for v in mapping]
        for u, v in trees[19]:
            a, b = (mapping[u], mapping[v]) if mapping[u] < mapping[v] else (mapping[v], mapping[u])
            used_edges.add((a, b))
    
    return result
# EVOLVE-BLOCK-END
