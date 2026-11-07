import time
import random
from itertools import combinations

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edges, n):
    """Check if no simple 3-edge path has endpoints with equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edges:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _edge_ok(adj, deg, u, v, n):
    """Check if adding edge (u,v) keeps constraint. Assumes (u,v) not present."""
    for s in adj[u]:
        if s == v:
            continue
        ds = deg[s]
        for t in adj[v]:
            if t == u or t == s:
                continue
            if ds == deg[t]:
                return False
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[u] == deg[y]:
                return False
    for x in adj[u]:
        if x == v:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[x] == deg[u]:
                return False
    return True

def _construct_with_buckets(buckets, n):
    """
    Construct a graph where vertices in the same bucket have the same degree,
    and no two vertices in the same bucket are at distance 2.
    Then greedily add edges to maximize edge count while respecting degrees.
    Returns (edges, adj, deg).
    """
    # Assign target degrees: bucket i gets degree = max_deg - i
    # We want high degrees for small buckets
    num_buckets = len(buckets)
    target_degrees = {}
    for i, bucket in enumerate(buckets):
        # Higher index = lower target degree
        target_degrees[i] = n - 1 - i * 2  # rough estimate, will adjust
    
    # Actually, let's just try to build a valid graph greedily
    # First, ensure the bucket structure is valid (no same-bucket at dist 2)
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    # Greedily add all possible edges that don't violate the constraint
    # and don't make any two same-bucket vertices reach distance 2
    # Actually, the constraint is about the FINAL degrees, not intermediate.
    # So we need a different approach.
    
    # Better approach: 
    # 1. Assign each vertex a "target degree" based on its bucket.
    # 2. Build a graph where each vertex v has exactly target_degree[v] edges.
    # 3. Verify the constraint holds.
    # 4. If it holds, try to add more edges via local search.
    
    # For simplicity, let's use a fixed assignment:
    # Vertex i gets degree = n-1 - (i // bucket_size) * 2
    # But this is tricky. Let's just do a smart greedy.
    
    # Smart greedy: add edges in order of "least constraint violation potential"
    # Start with empty graph, add edges that are "safe"
    present = set()
    for u in range(n):
        for v in range(u + 1, n):
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
                present.add((u, v))
    
    return edges, adj, deg, present

def construct():
    start = time.time()
    deadline = start + 85
    n = 15
    rng = random.Random(12345)

    best_edges = None
    best_count = -1

    def evaluate_and_track(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]

    # Phase 1: Try multiple random orderings for greedy construction
    best_seed_edges = []
    best_seed_count = -1

    for trial in range(80):
        if time.time() > deadline - 30:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        if len(edges) > best_seed_count:
            best_seed_count = len(edges)
            best_seed_edges = [list(e) for e in edges]

    # Start local search from best seed
    edges = [tuple(e) for e in best_seed_edges]
    adj = _build_adj(edges, n)
    deg = [len(adj[i]) for i in range(n)]
    evaluate_and_track(edges)

    # Phase 2: Local search with add, remove, swap
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    present = set()
    for u, v in edges:
        present.add((u, v))

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append((u, v))
        present.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove((u, v))
        present.discard((u, v))

    improved = True
    max_iter = 15
    iters = 0
    while improved and time.time() < deadline - 15 and iters < max_iter:
        improved = False
        iters += 1
        # Try adding missing edges
        for u, v in all_pairs:
            if time.time() > deadline - 15:
                break
            if (u, v) in present:
                continue
            if _edge_ok(adj, deg, u, v, n):
                add_edge(u, v)
                evaluate_and_track(edges)
                improved = True
        # Try swaps: remove e1, add e2
        if time.time() < deadline - 15:
            for (a, b) in list(edges):
                if time.time() > deadline - 15:
                    break
                for (c, d) in all_pairs:
                    if time.time() > deadline - 15:
                        break
                    if (c, d) in present:
                        continue
                    if (a, b) == (c, d):
                        continue
                    adj[a].discard(b)
                    adj[b].discard(a)
                    deg[a] -= 1
                    deg[b] -= 1
                    edges.remove((a, b))
                    present.discard((a, b))
                    if _edge_ok(adj, deg, c, d, n):
                        add_edge(c, d)
                        evaluate_and_track(edges)
                        improved = True
                    else:
                        adj[a].add(b)
                        adj[b].add(a)
                        deg[a] += 1
                        deg[b] += 1
                        edges.append((a, b))
                        present.add((a, b))

    # Phase 3: Random restarts with perturbation
    if time.time() < deadline - 10:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            adj = [set() for _ in range(n)]
            deg = [0] * n
            edges = []
            
            base_edges = [tuple(e) for e in best_edges] if best_edges else []
            rng.shuffle(base_edges)
            num_remove = min(len(base_edges), rng.randint(3, 8))
            for i in range(num_remove):
                e = base_edges[i]
                adj[e[0]].add(e[1])
                adj[e[1]].add(e[0])
                deg[e[0]] += 1
                deg[e[1]] += 1
                edges.append(e)
            
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline - 2:
                    break
                if v in adj[u]:
                    continue
                if _edge_ok(adj, deg, u, v, n):
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append((u, v))
            
            evaluate_and_track(edges)
            
            if len(edges) >= best_count:
                present = set()
                for u, v in edges:
                    present.add((u, v))
                
                improved2 = True
                while improved2 and time.time() < deadline - 2:
                    improved2 = False
                    for u, v in all_pairs:
                        if time.time() > deadline - 2:
                            break
                        if (u, v) in present:
                            continue
                        if _edge_ok(adj, deg, u, v, n):
                            adj[u].add(v)
                            adj[v].add(u)
                            deg[u] += 1
                            deg[v] += 1
                            edges.append((u, v))
                            present.add((u, v))
                            evaluate_and_track(edges)
                            improved2 = True

    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return result
