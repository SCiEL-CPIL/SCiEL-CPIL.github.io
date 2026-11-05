import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 147.0  # safety margin before 150s kill

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
    edge_set = set(tuple(sorted(e)) for e in edges)

    def count_c4(u, v):
        """Count 4-cycles containing edge (u,v)."""
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

    def compute_total_c4():
        """Exact count of 4-cycles: sum over all edges of c4_through_edge(e) / 4."""
        total = 0
        for e in edges:
            total += count_c4(e[0], e[1])
        return total // 4

    # ---- Adjust to exactly 91 edges ----
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = float('inf')
            for (u, v) in list(edges):
                c = count_c4(u, v)
                if c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            edge_set.discard(tuple(sorted(best)))
            remove_edge(best[0], best[1])
    elif m < 91:
        to_add = 91 - m
        for _ in range(to_add):
            best = None
            best_c = float('inf')
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    c = count_c4(u, v)
                    if c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(best)
            add_edge(best[0], best[1])

    # ---- Fast local search with incremental C4 tracking ----
    # Precompute all non-edges
    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]

    # Compute initial total C4
    total_c4 = compute_total_c4()
    best_edges = list(edges)
    best_total = total_c4

    def delta_swap(ru, rv, au, av):
        """
        Compute delta in total C4 count for removing (ru,rv) and adding (au,av).
        """
        # C4s destroyed: those containing (ru,rv) in current graph
        destroyed = count_c4(ru, rv)
        
        # Temporarily remove (ru,rv)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        
        # C4s created: those containing (au,av) in the modified graph
        created = count_c4(au, av)
        
        # Restore
        adj[ru].add(rv)
        adj[rv].add(ru)
        
        return created - destroyed

    # Local search: steepest descent on edge swaps
    no_improve_count = 0
    max_no_improve = 200

    for it in range(200000):
        if time.time() > deadline - 3.0:
            break
        
        # Try to find an improving swap
        improved = False
        
        # Score all removal candidates
        rem_scores = [(count_c4(e[0], e[1]), e) for e in edges]
        rem_scores.sort(key=lambda x: x[0])
        
        # Score all addition candidates
        add_scores = []
        for (u, v) in all_nonedges:
            add_scores.append((count_c4(u, v), (u, v)))
        add_scores.sort(key=lambda x: x[0])
        
        # Try combinations of top removals and additions
        found = False
        for rc, re in rem_scores[:15]:
            if found:
                break
            ru, rv = re
            for ac, ae in add_scores[:15]:
                au, av = ae
                d = delta_swap(ru, rv, au, av)
                if d < 0:
                    # Commit the swap
                    remove_edge(ru, rv)
                    edge_set.discard(tuple(sorted((ru, rv))))
                    add_edge(au, av)
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    # Update nonedges list
                    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    total_c4 += d
                    found = True
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            if total_c4 < best_total:
                best_total = total_c4
                best_edges = list(edges)
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Perturbation: make 3 random swaps to escape local minimum
                for _ in range(3):
                    re = edges[np.random.randint(len(edges))]
                    nonedge_list = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    if not nonedge_list:
                        break
                    ae = nonedge_list[np.random.randint(len(nonedge_list))]
                    ru, rv = re
                    au, av = ae
                    d = delta_swap(ru, rv, au, av)
                    remove_edge(ru, rv)
                    edge_set.discard(tuple(sorted((ru, rv))))
                    add_edge(au, av)
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    total_c4 += d
                no_improve_count = 0

    # Final polish: try all pairs of edges for best swap
    if time.time() < deadline - 5.0:
        for _ in range(3):  # Multiple polish passes
            improved_polish = True
            while improved_polish and time.time() < deadline - 3.0:
                improved_polish = False
                rem_scores = [(count_c4(e[0], e[1]), e) for e in edges]
                rem_scores.sort(key=lambda x: x[0])
                add_scores = []
                for (u, v) in all_nonedges:
                    add_scores.append((count_c4(u, v), (u, v)))
                add_scores.sort(key=lambda x: x[0])
                
                for rc, re in rem_scores[:25]:
                    ru, rv = re
                    for ac, ae in add_scores[:25]:
                        au, av = ae
                        d = delta_swap(ru, rv, au, av)
                        if d < 0:
                            remove_edge(ru, rv)
                            edge_set.discard(tuple(sorted((ru, rv))))
                            add_edge(au, av)
                            edge_set.add(tuple(sorted((au, av))))
                            edges.remove(re)
                            edges.append((au, av))
                            all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                            total_c4 += d
                            improved_polish = True
                            break
                    if improved_polish:
                        break
                
                if total_c4 < best_total:
                    best_total = total_c4
                    best_edges = list(edges)

    # Verify final count
    final_total = compute_total_c4()
    if final_total < best_total:
        best_total = final_total
        best_edges = list(edges)

    return [[e[0], e[1]] for e in best_edges]
