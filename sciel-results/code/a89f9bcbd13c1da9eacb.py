import time
from random import Random
from collections import defaultdict

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    trees = instance_trees(20, 611)
    N = 20
    
    # Pre-compute tree properties for ordering
    tree_info = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_info.append((0, 0, i)) # max_deg, k, index
            continue
        deg = [0] * k
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        tree_info.append((max(deg), k, i))
    
    # Order: High max degree first, then large k
    # This is often better than just k because high degree trees are harder to place
    order_indices = [x[2] for x in sorted(tree_info, key=lambda x: (-x[0], -x[1]))]

    def embed_csp_bitset(tree_edges, k, used_edges_bitsets, rng, node_limit=100000):
        if k == 1:
            return [0]
        
        # Build tree adjacency
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            deg[a] += 1; deg[b] += 1
            adj[a].append(b); adj[b].append(a)
        
        # Variable order: High degree first
        var_order = sorted(range(k), key=lambda v: -deg[v])
        
        mapping = [-1] * k
        used_hosts = [False] * N
        nodes = [0]
        
        # Pre-compute available hosts for each tree vertex? 
        # No, depends on mapping.
        
        def get_candidates(v):
            cands = []
            for h in range(N):
                if used_hosts[h]:
                    continue
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        # Check if edge (h, mw) is used
                        if (used_edges_bitsets[h] >> mw) & 1:
                            ok = False
                            break
                if ok:
                    cands.append(h)
            return cands

        def backtrack(idx):
            if nodes[0] > node_limit:
                return False
            if idx == k:
                return True
            
            # MRV Heuristic
            best_v = -1
            best_cands = None
            best_len = N + 1
            for i in range(idx, k):
                v = var_order[i]
                if mapping[v] == -1:
                    cands = get_candidates(v)
                    if len(cands) == 0:
                        return False
                    if len(cands) < best_len:
                        best_len = len(cands)
                        best_v = v
                        best_cands = cands
                        if best_len == 1:
                            break
            
            if best_v == -1:
                return True
            
            pos_in_order = var_order.index(best_v)
            if pos_in_order != idx:
                var_order[idx], var_order[pos_in_order] = var_order[pos_in_order], var_order[idx]
            
            v = best_v
            cands = best_cands
            
            # Randomize order of candidates to avoid bias
            rng.shuffle(cands)
            
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_limit:
                    return False
                
                mapping[v] = h
                used_hosts[h] = True
                
                added_edges = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        # Add edge (h, mw)
                        used_edges_bitsets[h] |= (1 << mw)
                        used_edges_bitsets[mw] |= (1 << h)
                        added_edges.append((h, mw))
                        
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for u, w in added_edges:
                    used_edges_bitsets[u] &= ~(1 << w)
                    used_edges_bitsets[w] &= ~(1 << u)
                used_hosts[h] = False
                mapping[v] = -1
                
            return False

        if backtrack(0):
            return mapping
        return None

    def evaluate(order, rng):
        # used_edges_bitsets[i] is a bitmask of neighbors j where (i,j) is used
        used_edges_bitsets = [0] * N
        result = [None] * N
        total = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            k = idx + 1
            m = embed_csp_bitset(trees[idx], k, used_edges_bitsets, rng, node_limit=50000)
            if m is not None:
                result[idx] = [int(x) for x in m]
                total += len(trees[idx])
            else:
                # If embedding fails, we stop this attempt
                break
                
        return result, total

    master_rng = Random(42)
    
    # Generate a diverse set of orders
    orders = [order_indices] # The "smart" order first
    
    # Add random permutations of the smart order? Or fully random?
    # Fully random is good for escaping local minima.
    for _ in range(1000):
        o = list(range(20))
        master_rng.shuffle(o)
        orders.append(o)
        
    best_result = None
    best_val = -1
    attempt = 0
    
    for o in orders:
        if time.time() > deadline - 5:
            break
        attempt += 1
        r, v = evaluate(o, Random(attempt))
        if v > best_val:
            best_val = v
            best_result = r
        if best_val == 190:
            break

    if best_result is not None and best_val > 0:
        return best_result
    return [None] * 19 + [list(range(20))]
