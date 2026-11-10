def construct():
    import time
    import sys
    sys.setrecursionlimit(1000000)
    
    def build_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def greedy_vertex_cover(adj, n):
        cover = set()
        uncovered = set((u, v) for u in range(n) for v in adj[u] if u < v)
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in cover:
                    continue
                count = sum(1 for u in adj[v] if (min(u,v), max(u,v)) in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            cover.add(best_v)
            uncovered = {e for e in uncovered if e[0] != best_v and e[1] != best_v}
        return cover
    
    def local_search(adj, n, cover, max_iters=10000):
        cover = set(cover)
        for _ in range(max_iters):
            improved = False
            # Try removal
            for v in list(cover):
                if all(u in cover for u in adj[v]):
                    cover.discard(v)
                    improved = True
                    break
            if improved:
                continue
            # Try swap
            for v in list(cover):
                for w in range(n):
                    if w in cover:
                        continue
                    new_cover = (cover - {v}) | {w}
                    if all(u in new_cover for u in adj[v]):
                        cover = new_cover
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return cover
    
    deadline = time.time() + 120
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 5:
            adj = build_graph(n, p, seed)
            cover = greedy_vertex_cover(adj, n)
            results.append(sorted(cover))
            continue
        
        adj = build_graph(n, p, seed)
        cover = greedy_vertex_cover(adj, n)
        cover = local_search(adj, n, cover)
        results.append(sorted(cover))
    
    return results
