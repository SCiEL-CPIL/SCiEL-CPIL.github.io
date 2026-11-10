import itertools

def _valid(edges, n=15):
    """Check if the edge set satisfies the constraint."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Check every path s-u-v-t where (u,v) is an edge, s != t, s != v, t != u
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Start with K_{7,8} as a known valid base (56 edges)
    base_edges = []
    for i in range(7):
        for j in range(7, 15):
            base_edges.append([i, j])
    
    best = list(base_edges)
    best_count = len(best)
    
    # Try to add edges one by one (greedy with backtracking-lite)
    # We'll do a local search: try adding each missing edge, if valid, keep it; otherwise skip.
    # Then try removing edges to allow adding more.
    
    current = list(base_edges)
    current_set = set(map(tuple, current))
    
    # Phase 1: Greedy add
    for e in all_edges:
        if e not in current_set:
            new_current = current + [list(e)]
            if _valid(new_current, n):
                current = new_current
                current_set.add(e)
    
    best = list(current)
    best_count = len(current)
    
    # Phase 2: Local search - try removing one edge and adding two
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(i+1, len(best)):
                # Try removing edges i and j
                remaining = best[:i] + best[i+1:j] + best[j+1:]
                rem_set = set(map(tuple, remaining))
                
                # Try adding two new edges
                for e1, e2 in itertools.combinations(all_edges, 2):
                    if e1 in rem_set or e2 in rem_set or e1 == e2:
                        continue
                    candidate = remaining + [list(e1), list(e2)]
                    if _valid(candidate, n):
                        best = candidate
                        best_count = len(candidate)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try removing one edge and adding two different ones
        for i in range(len(best)):
            remaining = best[:i] + best[i+1:]
            rem_set = set(map(tuple, remaining))
            
            for e1, e2 in itertools.combinations(all_edges, 2):
                if e1 in rem_set or e2 in rem_set or e1 == e2:
                    continue
                # Also make sure e1 and e2 are not the removed edge
                if tuple(e1) == tuple(best[i]) or tuple(e2) == tuple(best[i]):
                    continue
                candidate = remaining + [list(e1), list(e2)]
                if _valid(candidate, n):
                    best = candidate
                    best_count = len(candidate)
                    improved = True
                    break
            if improved:
                break
        if improved:
            break
    
    # Ensure return type is correct: list of [int, int]
    result = []
    for e in best:
        u, v = e
        result.append([int(u), int(v)])
    
    return result
