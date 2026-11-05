import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0

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

    # Build initial polarity graph
    A = np.zeros((n, n), dtype=np.int32)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append((i, j))

    m = len(edges)

    # Adjust to exactly 91 edges
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best_e = None
            best_c = None
            for (u, v) in edges:
                nu = np.where(A[u] == 1)[0]
                nv = np.where(A[v] == 1)[0]
                c = 0
                for a in nu:
                    if a == v:
                        continue
                    common = np.intersect1d(nv, np.where(A[a] == 1)[0])
                    c += len([b for b in common if b != u and b != a])
                if best_c is None or c < best_c:
                    best_c = c
                    best_e = (u, v)
            edges.remove(best_e)
            A[best_e[0], best_e[1]] = 0
            A[best_e[1], best_e[0]] = 0
    elif m < 91:
        to_add = 91 - m
        edge_set = set(edges)
        for _ in range(to_add):
            best_e = None
            best_c = None
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    nu = np.where(A[u] == 1)[0]
                    nv = np.where(A[v] == 1)[0]
                    c = 0
                    for a in nu:
                        if a == v:
                            continue
                        common = np.intersect1d(nv, np.where(A[a] == 1)[0])
                        c += len([b for b in common if b != u and b != a])
                    if best_c is None or c < best_c:
                        best_c = c
                        best_e = (u, v)
            edges.append(best_e)
            edge_set.add(best_e)
            A[best_e[0], best_e[1]] = 1
            A[best_e[1], best_e[0]] = 1

    def count_total_c4():
        """C4_count = (1/2) * sum_{i<j} C(A2[i,j], 2)"""
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                k = A2[i, j]
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    # Precompute for fast swap evaluation
    def c4_contribution(u, v):
        """Number of C4s that include edge (u,v) as one of the 4 edges."""
        nv = np.where(A[v] == 1)[0]
        nu = np.where(A[u] == 1)[0]
        c = 0
        for a in nv:
            if a == u:
                continue
            for b in nu:
                if b == v or b == a:
                    continue
                if A[a, b] == 1:
                    c += 1
        return c

    def swap_delta_fast(u, v, x, y, A2):
        """Fast exact C4 delta for removing (u,v) and adding (x,y).
        
        Uses precomputed A2. Only iterates over affected pairs.
        """
        Au = A[:, u]
        Av = A[:, v]
        Ax = A[:, x]
        Ay = A[:, y]
        
        delta_rm = np.outer(Au, Av) + np.outer(Av, Au)
        delta_ad = np.outer(Ax, Ay) + np.outer(Ay, Ax)
        delta_net = delta_ad - delta_rm
        
        total_change = 0
        nz = np.nonzero(delta_net)
        for idx in range(len(nz[0])):
            i, j = nz[0][idx], nz[1][idx]
            if i == j:
                continue
            old_val = A2[i, j]
            new_val = old_val + delta_net[i, j]
            old_c = old_val * (old_val - 1) // 2 if old_val >= 2 else 0
            new_c = new_val * (new_val - 1) // 2 if new_val >= 2 else 0
            total_change += (new_c - old_c)
        return total_change // 2

    # Local search: edge swaps with spectral-aware scoring
    edge_set = set((min(e[0],e[1]), max(e[0],e[1])) for e in edges)
    
    # Get current C4 count and A2
    current_c4 = count_total_c4()
    A2 = A @ A
    
    best_c4 = current_c4
    best_edges = list(edges)
    
    # Spectral score: sum of squared eigenvalues beyond the trivial ones
    # For a regular-ish graph, minimizing C4s correlates with minimizing sum(λ_i^2) for i>=2
    def spectral_score(A_mat):
        eigvals = np.linalg.eigvalsh(A_mat)
        # Sort ascending; the largest eigenvalue is ~d, we care about the rest
        return float(np.sum(eigvals[1:]**2))
    
    # Precompute degree-based heuristic for fast candidate ranking
    def edge_c4_score_fast(u, v, A2_mat):
        """Fast proxy: number of C4s involving edge (u,v) ~ sum over common-neighbor pairs.
        For edge (u,v), C4s through it = number of paths u-a-b-v with a~b, a!=v, b!=u.
        Approximate via: for each a in N(u)\\{v}, count neighbors of a in N(v)\\{u}.
        """
        nu = np.where(A_mat[u] == 1)[0]
        nv = np.where(A_mat[v] == 1)[0]
        c = 0
        for a in nu:
            if a == v:
                continue
            # Count how many neighbors of a are in N(v)\{u}
            na = np.where(A_mat[a] == 1)[0]
            common = np.intersect1d(na, nv)
            c += len([b for b in common if b != u and b != a])
        return c
    
    # Local search with edge swaps
    max_iters = 200000
    no_improve_count = 0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find best swap using spectral-aware candidate selection
        best_d = 0
        best_move = None
        
        # Score all edges by their C4 contribution (for removal candidates)
        rem_cands = []
        for e in edges:
            c = edge_c4_score_fast(e[0], e[1], A)
            rem_cands.append((c, e))
        rem_cands.sort()
        
        # Score all non-edges by potential C4 contribution (for addition candidates)
        add_cands = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                c = edge_c4_score_fast(u, v, A)
                add_cands.append((c, (u, v)))
        add_cands.sort()
        
        # Try top K removals x top K additions
        K = 40
        for rc, re in rem_cands[:K]:
            ru, rv = re
            for ac, ae in add_cands[:K]:
                au, av = ae
                if au == ru and av == rv:
                    continue
                if au == rv and av == ru:
                    continue
                d = swap_delta_fast(ru, rv, au, av, A2)
                if d < best_d:
                    best_d = d
                    best_move = (re, ae)
        
        if best_move is not None and best_d < 0:
            # Apply the swap
            (ru, rv), (au, av) = best_move
            A[ru, rv] = 0
            A[rv, ru] = 0
            A[au, av] = 1
            A[av, au] = 1
            edges.remove((ru, rv))
            edges.append((au, av))
            edge_set.discard((min(ru,rv), max(ru,rv)))
            edge_set.add((min(au,av), max(au,av)))
            current_c4 += best_d
            A2 = A @ A
            no_improve_count = 0
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
        else:
            no_improve_count += 1
            # Random perturbation when stuck
            if no_improve_count % 50 == 0 and no_improve_count > 0:
                re = edges[np.random.randint(len(edges))]
                while True:
                    u = np.random.randint(n)
                    v = np.random.randint(n)
                    if u == v:
                        continue
                    k = (min(u,v), max(u,v))
                    if k not in edge_set:
                        break
                d = swap_delta_fast(re[0], re[1], u, v, A2)
                # Accept with simulated annealing probability
                if d <= 0 or np.random.rand() < np.exp(-d / max(1.0, current_c4/10.0)):
                    A[re[0], re[1]] = 0
                    A[re[1], re[0]] = 0
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.remove(re)
                    edges.append((u, v))
                    edge_set.discard((min(re[0],re[1]), max(re[0],re[1])))
                    edge_set.add((min(u,v), max(u,v)))
                    current_c4 += d
                    A2 = A @ A
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges = list(edges)
                        no_improve_count = 0
    
    # Final polish: try all single-edge swaps one more time
    if time.time() < deadline - 5.0:
        A2 = A @ A
        for re in list(edges):
            ru, rv = re
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    if u == ru and v == rv:
                        continue
                    if u == rv and v == ru:
                        continue
                    d = swap_delta_fast(ru, rv, u, v, A2)
                    if d < 0:
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[u, v] = 1
                        A[v, u] = 1
                        edges.remove((ru, rv))
                        edges.append((u, v))
                        edge_set.discard((min(ru,rv), max(ru,rv)))
                        edge_set.add((min(u,v), max(u,v)))
                        current_c4 += d
                        A2 = A @ A
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_edges = list(edges)
                        break
                if time.time() > deadline - 2.0:
                    break
    
    # Use best found
    final_edges = best_edges
    return [[e[0], e[1]] for e in final_edges]
