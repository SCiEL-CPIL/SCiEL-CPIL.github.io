import time
import random
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

def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_leaf_first_order(adj, k):
    """
    Returns vertices in an order where leaves come first.
    This is done by repeatedly removing leaves (topological sort of the tree from leaves inward).
    """
    if k == 1:
        return [0]
    if k == 2:
        return [0, 1]
    
    # Compute degrees
    deg = [len(adj[v]) for v in range(k)]
    # Start with all leaves
    from collections import deque
    queue = deque([v for v in range(k) if deg[v] == 1])
    order = []
    remaining = set(range(k))
    
    while queue:
        v = queue.popleft()
        if v not in remaining:
            continue
        order.append(v)
        remaining.discard(v)
        for nb in adj[v]:
            if nb in remaining:
                deg[nb] -= 1
                if deg[nb] == 1:
                    queue.append(nb)
    
    # Add any remaining (shouldn't happen for a tree, but safety)
    for v in range(k):
        if v in remaining:
            order.append(v)
    
    return order

def construct():
    deadline = time.time() + 145  # 5s margin
    trees = instance_trees(20, 611)
    n = 20
    num_hosts = 20
    
    # Precompute tree structures
    tree_adjs = [_build_adjacency(trees[k-1], k) for k in range(1, n + 1)]
    # Use leaf-first ordering for better constraint propagation
    tree_orders = [_get_leaf_first_order(tree_adjs[k-1], k) for k in range(1, n + 1)]
    
    # Process largest trees first
    embed_order = list(range(n, 0, -1))  # 20, 19, ..., 1
    
    def try_pack_deterministic():
        """
        Deterministic packing with forward checking.
        """
        used_edges = set()
        result = [None] * n
        embeddings = {}
        
        def count_available_hosts(placed_mapping, tree_adj, tree_order, current_idx):
            """
            For each unplaced vertex in tree_order[current_idx:], count how many
            host vertices are still feasible. Returns min count, or -1 if infeasible.
            """
            used_in_tree = set(placed_mapping[i] for i in range(len(tree_order)) if placed_mapping[i] is not None)
            
            for idx in range(current_idx, len(tree_order)):
                v = tree_order[idx]
                # Count feasible hosts
                feasible_count = 0
                for h in range(num_hosts):
                    if h in used_in_tree:
                        continue
                    # Check edge conflicts with placed neighbors
                    ok = True
                    for nb in tree_adj[v]:
                        if placed_mapping[nb] is not None:
                            h_nb = placed_mapping[nb]
                            e_key = (min(h, h_nb), max(h, h_nb))
                            if e_key in used_edges:
                                ok = False
                                break
                    if ok:
                        feasible_count += 1
                if feasible_count == 0:
                    return -1
            
            return 0  # All have at least one option
        
        def embed_tree(k):
            """
            Embed tree T_k using backtracking with forward checking.
            Returns mapping list or None.
            """
            if k == 1:
                return [0]
            
            if k == 2:
                for a in range(num_hosts):
                    for b in range(a + 1, num_hosts):
                        if (a, b) not in used_edges:
                            return [a, b]
                return None
            
            adj = tree_adjs[k-1]
            order = tree_orders[k-1]
            mapping = [None] * k
            
            def dfs(idx):
                if idx == k:
                    return True
                
                v = order[idx]
                
                # Find placed neighbors
                placed_neighbors = []
                for nb in adj[v]:
                    if mapping[nb] is not None:
                        placed_neighbors.append(nb)
                
                # Candidate hosts
                used_in_tree = set(mapping[i] for i in range(k) if mapping[i] is not None)
                
                candidates = []
                for h in range(num_hosts):
                    if h in used_in_tree:
                        continue
                    ok = True
                    for nb in placed_neighbors:
                        h_nb = mapping[nb]
                        e_key = (min(h, h_nb), max(h, h_nb))
                        if e_key in used_edges:
                            ok = False
                            break
                    if ok:
                        candidates.append(h)
                        
                if not candidates:
                    return False
                
                # Sort candidates by a heuristic: prefer hosts with fewer remaining options
                # for their neighbors (this is a simple form of MRV)
                def score_candidate(h):
                    # Count how many unplaced neighbors of v will have fewer options if we use h
                    score = 0
                    for nb in adj[v]:
                        if mapping[nb] is None:
                            # If we place v at h, then nb must connect to h
                            # Check if edge (h, nb's future host) would be blocked
                            # This is complex; use a simpler heuristic
                            pass
                    return h  # Just use host index as tiebreaker for determinism
                
                candidates.sort(key=score_candidate)
                
                for h in candidates:
                    mapping[v] = h
                    
                    # Forward check: ensure all unplaced vertices still have options
                    if idx + 1 < k:
                        fc_result = count_available_hosts(mapping, adj, order, idx + 1)
                        if fc_result == -1:
                            mapping[v] = None
                            continue
                    
                    if dfs(idx + 1):
                        return True
                    mapping[v] = None
                    
                return False
            
            if dfs(0):
                return mapping
            return None
        
        # Process trees
        for k in embed_order:
            if time.time() > deadline:
                break
            
            if k == 1:
                result[k-1] = [0]
                continue
                
            mapping = embed_tree(k)
            if mapping is not None:
                result[k-1] = mapping
                # Add edges to used_edges
                for u, v in trees[k-1]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                result[k-1] = None
        
        # Calculate score
        score = 0
        for k in range(1, n + 1):
            if result[k-1] is not None:
                score += len(trees[k-1])
                
        return result, score
    
    # Try deterministic first
    best_result, best_score = try_pack_deterministic()
    
    # If deterministic failed to get full score, try a few randomized restarts
    if best_score < 190:
        for i in range(5):
            if time.time() > deadline:
                break
            
            rng = Random(1000 + i)
            used_edges = set()
            result = [None] * n
            embeddings = {}
            
            def embed_tree_random(k):
                if k == 1:
                    return [0]
                if k == 2:
                    for a in range(num_hosts):
                        for b in range(a + 1, num_hosts):
                            if (a, b) not in used_edges:
                                return [a, b]
                    return None
                
                adj = tree_adjs[k-1]
                order = tree_orders[k-1]
                mapping = [None] * k
                
                def dfs(idx):
                    if idx == k:
                        return True
                    
                    v = order[idx]
                    placed_neighbors = []
                    for nb in adj[v]:
                        if mapping[nb] is not None:
                            placed_neighbors.append(nb)
                    
                    used_in_tree = set(mapping[i] for i in range(k) if mapping[i] is not None)
                    candidates = []
                    for h in range(num_hosts):
                        if h in used_in_tree:
                            continue
                        ok = True
                        for nb in placed_neighbors:
                            h_nb = mapping[nb]
                            e_key = (min(h, h_nb), max(h, h_nb))
                            if e_key in used_edges:
                                ok = False
                                break
                        if ok:
                            candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        mapping[v] = h
                        if dfs(idx + 1):
                            return True
                        mapping[v] = None
                    
                    return False
                
                if dfs(0):
                    return mapping
                return None
            
            for k in embed_order:
                if time.time() > deadline:
                    break
                if k == 1:
                    result[k-1] = [0]
                    continue
                mapping = embed_tree_random(k)
                if mapping is not None:
                    result[k-1] = mapping
                    for u, v in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                else:
                    result[k-1] = None
            
            score = sum(len(trees[k-1]) for k in range(1, n+1) if result[k-1] is not None)
            if score > best_score:
                best_score = score
                best_result = result
            
            if best_score == 190:
                break
    
    return best_result

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k - 1] is not None:
            r = res[k - 1]
            if not isinstance(r, list):
                print(f"T{k}: not a list")
                ok = False
                continue
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v!r}")
                    ok = False
            for (u, v) in trees[k - 1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k - 1]) for k in range(1, 21) if res[k - 1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
