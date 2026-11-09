import time
import itertools
import random

def construct():
    n = 15
    deadline = time.time() + 80.0  # safety margin before 90s
    best_edges = None
    best_count = -1

    def check_fast(adj, deg, present, edges_list):
        """Fast check using pre-built structures. adj[i] is a set of neighbors."""
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

    def check(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        return check_fast(adj, deg, None, edges)

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
                    if check_fast(adj, deg, present, edges):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u].discard(v); adj[v].discard(u)
                        deg[u] -= 1; deg[v] -= 1
        return edges

    def local_search_2opt(start_edges, max_time=10.0):
        """2-opt: try removing one edge and adding one or two others."""
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
        
        best = len(edges)
        best_edges_copy = [e[:] for e in edges]
        start_time = time.time()
        
        # Try adding edges first
        for u in range(n):
            if time.time() - start_time > max_time:
                break
            for v in range(u + 1, n):
                if (u, v) in present:
                    continue
                adj[u].add(v); adj[v].add(u)
                deg[u] += 1; deg[v] += 1
                if check_fast(adj, deg, present, edges):
                    present.add((u, v))
                    edges.append([u, v])
                    if len(edges) > best:
                        best = len(edges)
                        best_edges_copy = [e[:] for e in edges]
                else:
                    adj[u].discard(v); adj[v].discard(u)
                    deg[u] -= 1; deg[v] -= 1
        
        # 2-opt: remove edge (a,b), add edge (c,d)
        if time.time() - start_time < max_time - 2:
            edge_list = list(present)
            random.shuffle(edge_list)
            for ab in edge_list:
                if time.time() - start_time > max_time - 1:
                    break
                a, b = ab
                # Try removing (a,b)
                adj[a].discard(b); adj[b].discard(a)
                deg[a] -= 1; deg[b] -= 1
                present.discard((a, b))
                edges = [e for e in edges if not ((e[0] == a and e[1] == b) or (e[0] == b and e[1] == a))]
                
                # Try adding each missing edge
                for c in range(n):
                    if time.time() - start_time > max_time - 1:
                        break
                    for d in range(c + 1, n):
                        if (c, d) in present:
                            continue
                        adj[c].add(d); adj[d].add(c)
                        deg[c] += 1; deg[d] += 1
                        if check_fast(adj, deg, present, edges):
                            present.add((c, d))
                            edges.append([c, d])
                            if len(edges) > best:
                                best = len(edges)
                                best_edges_copy = [e[:] for e in edges]
                        else:
                            adj[c].discard(d); adj[d].discard(c)
                            deg[c] -= 1; deg[d] -= 1
                
                # Restore (a,b)
                adj[a].add(b); adj[b].add(a)
                deg[a] += 1; deg[b] += 1
                present.add((a, b))
                edges.append([a, b])
        
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
                    if check_fast(adj, [len(adj[i]) for i in range(n)], present, edges):
                        break
                    # find a violation
                    deg2 = [len(adj[i]) for i in range(n)]
                    found = False
                    for u in range(n):
                        if found: break
                        for v in adj[u]:
                            if v <= u: continue
                            for s in adj[u]:
                                if s == v: continue
                                for t in adj[v]:
                                    if t == u or t == s: continue
                                    if deg2[s] == deg2[t]:
                                        edges = [e for e in edges if not ((e[0]==u and e[1]==v) or (e[0]==v and e[1]==u))]
                                        adj[u].discard(v); adj[v].discard(u)
                                        present.discard((u, v) if u < v else (v, u))
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
            ls_result = local_search_2opt(best_edges, max_time=8.0)
            if check(ls_result):
                cnt = len(ls_result)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = ls_result

    # Final polish: try 2-opt on best result
    if best_edges is not None and time.time() < deadline - 5:
        final = local_search_2opt(best_edges, max_time=10.0)
        if check(final):
            cnt = len(final)
            if cnt > best_count:
                best_count = cnt
                best_edges = final

    if best_edges is None:
        # fallback star
        return [[0, v] for v in range(1, 15)]

    # convert to plain python ints and lists
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
