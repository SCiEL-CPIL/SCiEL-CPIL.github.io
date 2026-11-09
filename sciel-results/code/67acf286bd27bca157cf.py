"""Packing trees T1..T20 into K20 with edge-disjointness using optimized backtracking."""
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


def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _embedding_order(adj, k):
    """BFS order starting from a high-degree vertex for better constraint propagation."""
    if k <= 1:
        return list(range(k))
    
    # Pick root: vertex with highest degree
    degrees = [len(adj[v]) for v in range(k)]
    root = max(range(k), key=lambda v: degrees[v])
    
    visited = [False] * k
    order = [root]
    visited[root] = True
    frontier = [root]
    while frontier:
        next_frontier = []
        for node in frontier:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    order.append(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    for v in range(k):
        if not visited[v]:
            order.append(v)
    return order


def construct():
    deadline = time.time() + 142  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    n = 20
    total_vertices = 20
    
    # Precompute
    adj_list = [_build_adjacency(trees[k - 1], k) for k in range(1, n + 1)]
    order_list = [_embedding_order(adj_list[k - 1], k) for k in range(1, n + 1)]
    
    # Embed in order: larger trees first
    embed_order = list(range(n, 0, -1))
    
    used_edges = set()
    result = [None] * n
    
    def _try_embed(k, adj, order, seed=0):
        """Try to embed tree T_k into K20 avoiding used_edges."""
        mapping = [-1] * k
        k_val = k
        rng = Random(seed)
        
        def _dfs(idx):
            if idx == k_val:
                return True
            
            v = order[idx]
            
            # Find placed neighbors
            placed_neighbors = []
            for nb in adj[v]:
                if mapping[nb] >= 0:
                    placed_neighbors.append(nb)
            
            # Used hosts set
            used_hosts_set = set()
            for i in range(k_val):
                if mapping[i] >= 0:
                    used_hosts_set.add(mapping[i])
            
            # Generate candidates
            candidates = []
            for h in range(total_vertices):
                if h in used_hosts_set:
                    continue
                ok = True
                for nb in placed_neighbors:
                    edge_key = (min(h, mapping[nb]), max(h, mapping[nb]))
                    if edge_key in used_edges:
                        ok = False
                        break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort candidates by number of remaining options for unplaced neighbors (MRV-like)
            # Actually just shuffle for diversity
            rng.shuffle(candidates)
            
            for h in candidates:
                mapping[v] = h
                
                # Forward checking: ensure all unplaced vertices still have at least one candidate
                fc_ok = True
                for j in range(idx + 1, k_val):
                    u = order[j]
                    placed_nbs_u = []
                    for nb in adj[u]:
                        if mapping[nb] >= 0:
                            placed_nbs_u.append(nb)
                    
                    if not placed_nbs_u:
                        continue
                    
                    # Check if u has any valid host
                    has_candidate = False
                    for hu in range(total_vertices):
                        if hu in used_hosts_set or hu == h:
                            continue
                        ok_u = True
                        for nb in placed_nbs_u:
                            edge_key = (min(hu, mapping[nb]), max(hu, mapping[nb]))
                            if edge_key in used_edges:
                                ok_u = False
                                break
                        if ok_u:
                            has_candidate = True
                            break
                    if not has_candidate:
                        fc_ok = False
                        break
                
                if fc_ok:
                    if _dfs(idx + 1):
                        return True
                mapping[v] = -1
            
            return False
        
        if _dfs(0):
            return [mapping[i] for i in range(k_val)]
        return None
    
    for k in embed_order:
        if time.time() > deadline:
            break
        
        if k == 1:
            result[k - 1] = [0]
            continue
        
        if k == 2:
            found = False
            for a in range(total_vertices):
                if found:
                    break
                for b in range(a + 1, total_vertices):
                    if (a, b) not in used_edges:
                        result[k - 1] = [a, b]
                        used_edges.add((a, b))
                        found = True
                        break
            if not found:
                result[k - 1] = None
            continue
        
        adj = adj_list[k - 1]
        order = order_list[k - 1]
        
        tree_start = time.time()
        max_time_per_tree = min(8.0, max(1.0, (deadline - tree_start) / max(1, len([x for x in embed_order if x >= k]))))
        
        embedding = None
        attempt = 0
        while time.time() - tree_start < max_time_per_tree and time.time() < deadline:
            embedding = _try_embed(k, adj, order, seed=attempt)
            if embedding is not None:
                break
            attempt += 1
        
        if embedding is not None:
            result[k - 1] = embedding
            for (u, v) in trees[k - 1]:
                edge_key = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
                used_edges.add(edge_key)
        else:
            result[k - 1] = None
    
    return result


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
