import time
import itertools

def construct():
    n = 15
    deadline = time.time() + 85.0  # safety margin before 90s
    best_edges = None
    best_count = -1

    def check(edges):
        # build adj and degree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def try_improve(start_edges):
        edges = list(start_edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))
        deg = [len(adj[i]) for i in range(n)]
        
        # Greedy add phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    # tentatively add
                    adj[u].add(v); adj[v].add(u)
                    deg[u] += 1; deg[v] += 1
                    if check(edges + [(u, v)]):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u].discard(v); adj[v].discard(u)
                        deg[u] -= 1; deg[v] -= 1
        return edges

    def local_search(start_edges):
        """Try to improve by swapping edges or adding/removing."""
        edges = list(start_edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))
        
        best = len(edges)
        best_edges_copy = [e[:] for e in edges]
        
        # Try adding edges
        for u in range(n):
            if time.time() > deadline:
                return best_edges_copy
            for v in range(u + 1, n):
                if (u, v) in present:
                    continue
                if check(edges + [[u, v]]):
                    edges.append([u, v])
                    present.add((u, v))
                    adj[u].add(v); adj[v].add(u)
                    if len(edges) > best:
                        best = len(edges)
                        best_edges_copy = [e[:] for e in edges]
        
        # Try removing an edge and adding two others (2-opt style)
        # This is expensive, so do it sparingly
        if time.time() < deadline - 10:
            # Simple random restarts with greedy
            import random
            random.seed(42)
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Start from a random subset of best_edges_copy
                start = [e[:] for e in best_edges_copy]
                # Remove a random edge
                if start:
                    idx = random.randint(0, len(start)-1)
                    start.pop(idx)
                improved = try_improve(start)
                if check(improved):
                    if len(improved) > best:
                        best = len(improved)
                        best_edges_copy = [e[:] for e in improved]
        
        return best_edges_copy

    # Base candidates
    candidates = []
    # bipartite K_{a,b}
    for a in range(1, 15):
        b = 15 - a
        edges = []
        A = list(range(a))
        B = list(range(a, 15))
        for x in A:
            for y in B:
                edges.append([x, y])
        candidates.append(edges)
    # complete graph
    complete = [[u, v] for u in range(15) for v in range(u + 1, 15)]
    candidates.append(complete)
    
    # Also try empty graph
    candidates.append([])
    
    # Try a few random sparse graphs
    import random
    random.seed(123)
    for _ in range(10):
        edges = []
        for u in range(15):
            for v in range(u+1, 15):
                if random.random() < 0.3:
                    edges.append([u, v])
        candidates.append(edges)

    for cand in candidates:
        if time.time() > deadline:
            break
        if not check(cand):
            # For complete graph, need to remove edges; do greedy removal
            if len(cand) == 105:
                edges = list(cand)
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v); adj[v].add(u)
                present = set()
                for u, v in edges:
                    a, b = (u, v) if u < v else (v, u)
                    present.add((a, b))
                # remove edges until valid
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    # find a violating middle edge and remove one of its contributing edges
                    # simpler: try removing each edge, see if reduces violations; but expensive.
                    # Instead: recompute and if invalid, remove the first edge that appears in a violation.
                    if check(edges):
                        break
                    # find a violation
                    adj2 = [set() for _ in range(n)]
                    for u, v in edges:
                        adj2[u].add(v); adj2[v].add(u)
                    deg2 = [len(adj2[i]) for i in range(n)]
                    found = False
                    for u in range(n):
                        if found: break
                        for v in adj2[u]:
                            if v <= u: continue
                            for s in adj2[u]:
                                if s == v: continue
                                for t in adj2[v]:
                                    if t == u or t == s: continue
                                    if deg2[s] == deg2[t]:
                                        # violation; remove edge (u,v) or (s,?) remove (u,v)
                                        edges = [e for e in edges if not ((e[0]==u and e[1]==v) or (e[0]==v and e[1]==u))]
                                        found = True
                                        changed = True
                                        break
                                if found: break
                            if found: break
                        if found: break
                    if not found:
                        break
                cand = edges
            else:
                continue
        improved = try_improve(cand)
        if check(improved):
            cnt = len(improved)
            if cnt > best_count:
                best_count = cnt
                best_edges = improved
        
        # Also try local search on the improved version
        if best_edges is not None and time.time() < deadline - 15:
            ls_result = local_search(best_edges)
            if check(ls_result):
                cnt = len(ls_result)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = ls_result

    if best_edges is None:
        # fallback star
        return [[0, v] for v in range(1, 15)]

    # convert to plain python ints and lists
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
