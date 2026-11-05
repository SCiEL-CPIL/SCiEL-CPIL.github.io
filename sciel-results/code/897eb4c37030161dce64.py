import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))

    m = len(edges)

    def count_c4(u, v):
        # number of 4-cycles u-a-b-v where a~u, b~v, a~b, a!=b
        nu = adj[u]
        nv = adj[v]
        c = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    c += 1
        return c

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    def total_c4():
        e_list = list(edges)
        L = len(e_list)
        total = 0
        for i in range(L):
            a, b = e_list[i]
            for j in range(i+1, L):
                c, d = e_list[j]
                if len({a,b,c,d}) < 4:
                    continue
                if (c in adj[a] and d in adj[b]) or (d in adj[a] and c in adj[b]):
                    total += 1
        return total

    def edge_key(e):
        return (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])

    # ---- Adjust to exactly 91 edges using local-search-quality moves ----
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = None
            for (u, v) in list(edges):
                c = count_c4(u, v)
                if best_c is None or c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            remove_edge(best[0], best[1])
    elif m < 91:
        to_add = 91 - m
        edge_set = set(edges)
        for _ in range(to_add):
            best = None
            best_c = None
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    c = count_c4(u, v)
                    if best_c is None or c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(best)
            add_edge(best[0], best[1])

    # ---- Fast vectorized local search ----
    edge_set = set(edge_key(e) for e in edges)
    
    # Build adjacency matrix for fast operations
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1

    def update_A(u, v, val):
        A[u, v] = val
        A[v, u] = val

    def fast_count_c4(u, v):
        """Count C4s containing edge (u,v) using vectorized ops."""
        # C4s: u-x-y-v where x~u, y~v, x~y, x!=v, y!=u, x!=y
        # Equivalent: for each x in N(u)\{v}, count y in N(v)\{u,x} such that x~y
        nu = A[u]
        nv = A[v]
        # Mask out u and v from neighborhoods
        mask_u = nu.copy()
        mask_u[v] = 0
        mask_v = nv.copy()
        mask_v[u] = 0
        # For each x in N(u)\{v}, count neighbors of x in N(v)\{u,x}
        total = 0
        for x in range(n):
            if mask_u[x]:
                # Count y in N(v)\{u} such that x~y and y != x
                common = np.logical_and(A[x], mask_v)
                common[x] = 0  # exclude x itself
                total += np.sum(common)
        return total

    def swap_delta(ru, rv, au, av):
        """Compute exact change in total C4 count for removing (ru,rv) and adding (au,av)."""
        # C4s destroyed by removing (ru,rv): C4s that use edge (ru,rv)
        # A C4 uses edge (ru,rv) when it has form ru-x-y-rv-ru
        # where x~ru, x~y, y~rv, all distinct
        destroyed = 0
        for x in adj[ru]:
            if x == rv:
                continue
            for y in adj[rv]:
                if y == ru or y == x:
                    continue
                if y in adj[x]:
                    destroyed += 1
        
        # C4s created by adding (au,av): C4s that use edge (au,av)
        # A C4 uses edge (au,av) when it has form au-x-y-av-av
        # where x~au, x~y, y~av, all distinct
        # Note: au and av are not yet adjacent, so we count potential C4s
        created = 0
        for x in adj[au]:
            if x == av:
                continue
            for y in adj[av]:
                if y == au or y == x:
                    continue
                if y in adj[x]:
                    created += 1
        
        return created - destroyed

    # Greedy local search with fast delta
    for it in range(100000):
        if time.time() > deadline - 10:
            break
        
        # Compute C4 counts for all edges
        edge_c4 = {}
        for e in edges:
            ek = edge_key(e)
            edge_c4[ek] = fast_count_c4(e[0], e[1])
        
        # Compute C4 counts for all non-edges
        nonedge_c4 = {}
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k not in edge_set:
                    nonedge_c4[k] = fast_count_c4(u, v)
        
        # Sort candidates
        rem_sorted = sorted(edge_c4.items(), key=lambda x: x[1])
        add_sorted = sorted(nonedge_c4.items(), key=lambda x: x[1])
        
        improved = False
        # Try top candidates for improvement
        for rk, rc in rem_sorted[:30]:
            if improved:
                break
            ru, rv = rk
            for ak, ac in add_sorted[:30]:
                au, av = ak
                if rk == ak:
                    continue
                d = swap_delta(ru, rv, au, av)
                if d < 0:
                    # Apply swap
                    remove_edge(ru, rv)
                    edge_set.discard(rk)
                    update_A(ru, rv, 0)
                    
                    add_edge(au, av)
                    edge_set.add(ak)
                    update_A(au, av, 1)
                    
                    edges.remove((ru, rv))
                    edges.append((au, av))
                    improved = True
                    break
        
        if not improved:
            # Random perturbation to escape local minima
            if edges and nonedge_c4:
                re = edges[random.randint(0, len(edges)-1)]
                re_key = edge_key(re)
                non_edges = list(nonedge_c4.keys())
                if non_edges:
                    ae = random.choice(non_edges)
                    remove_edge(re[0], re[1])
                    edge_set.discard(re_key)
                    update_A(re[0], re[1], 0)
                    
                    add_edge(ae[0], ae[1])
                    edge_set.add(ae)
                    update_A(ae[0], ae[1], 1)
                    
                    edges.remove(re)
                    edges.append(ae)

    # Final verification and return
    final_edges = [[e[0], e[1]] for e in edges]
    return final_edges
