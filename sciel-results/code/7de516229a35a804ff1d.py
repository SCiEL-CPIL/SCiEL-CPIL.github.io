import random
import time

def _is_valid(G):
    """Check that no simple 3-edge path s-u-v-t has deg(s)==deg(t) in full graph."""
    n = len(G)
    adj = [set() for _ in range(n)]
    for u, v in G:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _violation_count(G):
    """Count number of violating 3-edge paths (for optimization)."""
    n = len(G)
    adj = [set() for _ in range(n)]
    for u, v in G:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
    return count

def _edge_violation_contrib(G, edge):
    """Count violations that specifically use this edge as the middle (u,v)."""
    n = len(G)
    u, v = edge
    adj = [set() for _ in range(n)]
    for a, b in G:
        adj[a].add(b)
        adj[b].add(a)
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for s in adj[u]:
        if s == v:
            continue
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[s] == deg[t]:
                count += 1
    return count

def _greedy_build(n, seed=42):
    """Greedy: start empty, add edges that don't create violations, prefer high-degree vertices."""
    rng = random.Random(seed)
    G = []
    adj = [set() for _ in range(n)]
    deg = [0] * n
    # Order all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    for e in all_edges:
        # Try adding
        u, v = e
        # Check if adding (u,v) creates any violation
        # A violation s-a-b-t uses middle edge (a,b). New edge is (u,v).
        # Violations involving (u,v) as middle: s-u-v-t or s-v-u-t
        # Also (u,v) could be an endpoint edge in s-u-v-t where middle is (u,v) or (v,s) etc.
        # Simpler: just check all paths that use the new edge.
        ok = True
        # Paths where (u,v) is the middle edge: s-u-v-t, s!=t, s!=v, t!=u
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            # Also check paths where new edge is an outer edge
            # e.g., s-u-v-t where (u,v) is middle... already covered
            # s-v-u-t where (v,u) is middle... same as above with s,t swapped roles
            # Actually (u,v) as middle: s connects to u, t connects to v. Covered.
            # (u,v) as first edge: s-u-v-t means s-u, u-v, v-t. Middle is (u,v)? No, middle is the center edge.
            # Path s-u-v-t: edges are (s,u), (u,v), (v,t). Middle edge is (u,v).
            # So (u,v) as middle is the only case where it's the "center".
            # But (u,v) could be an outer edge: e.g., path u-v-w-x where middle is (v,w).
            # In that case, the violation depends on deg(u) and deg(x), not on whether (u,v) exists per se,
            # but the path u-v-w-x only exists if (u,v) is present. So adding (u,v) could enable
            # new violations where (u,v) is an outer edge.
            # Path u-v-w-x: middle (v,w), endpoints u and x. Violation if deg(u)==deg(x).
            # This path exists iff (u,v) and (v,w) and (w,x) all exist.
            # So adding (u,v) could create violations for all w in adj[v] (w!=u) and x in adj[w] (x!=u,v).
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    if deg[u] == deg[x]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            # Similarly path x-w-v-u: middle (w,v), endpoints x and u. Same check as above by symmetry.
            # Path v-u-w-x: middle (u,w), endpoints v and x.
            for w in adj[u]:
                if w == v:
                    continue
                for x in adj[w]:
                    if x == v or x == u:
                        continue
                    if deg[v] == deg[x]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            G.append(e)
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
    return G

def _local_search(G, n, max_iters=5000, seed=0):
    """Local search: try removing edges to fix violations, then adding edges back."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u, v in G:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    for it in range(max_iters):
        # Check validity
        violations = _violation_count(G)
        if violations == 0:
            break
        
        # Find the edge with max violation contribution and remove it
        best_edge = None
        best_contrib = -1
        for e in G:
            c = _edge_violation_contrib(G, e)
            if c > best_contrib:
                best_contrib = c
                best_edge = e
        if best_contrib <= 0:
            break
        u, v = best_edge
        G.remove(best_edge)
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
    
    # Try to add edges back
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    improved = True
    while improved:
        improved = False
        for e in all_edges:
            if e in G or (e[1], e[0]) in G:
                continue
            u, v = e
            # Check if adding creates violation
            ok = True
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                for w in adj[v]:
                    if w == u:
                        continue
                    for x in adj[w]:
                        if x == u or x == v:
                            continue
                        if deg[u] == deg[x]:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                for w in adj[u]:
                    if w == v:
                        continue
                    for x in adj[w]:
                        if x == v or x == u:
                            continue
                        if deg[v] == deg[x]:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                G.append(e)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                improved = True
    return G

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    n = 15
    best_G = None
    best_len = -1
    
    # Try multiple random seeds and approaches
    for seed in range(50):
        if time.time() > deadline:
            break
        
        # Approach 1: Greedy build
        G = _greedy_build(n, seed=seed)
        G = _local_search(G, n, max_iters=2000, seed=seed)
        if len(G) > best_len:
            best_len = len(G)
            best_G = G[:]
        
        if time.time() > deadline:
            break
        
        # Approach 2: Start from a random dense subset, then repair
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng = random.Random(seed + 1000)
        rng.shuffle(all_edges)
        # Start with a random subset of ~40% edges
        k = len(all_edges) * 4 // 10
        G2 = all_edges[:k]
        G2 = _local_search(G2, n, max_iters=2000, seed=seed + 2000)
        if len(G2) > best_len:
            best_len = len(G2)
            best_G = G2[:]
    
    # Final validation
    if best_G is None:
        # Fallback: star graph
        return [[0, v] for v in range(1, n)]
    
    # Ensure types are correct
    result = [[int(u), int(v)] for u, v in best_G]
    return result
