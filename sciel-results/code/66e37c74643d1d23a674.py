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
            # Remove edge with fewest C4s it participates in
            best_e = None
            best_c = None
            for (u, v) in edges:
                # Count C4s containing edge (u,v): for each w in N(u)\{v}, count common neighbors of w and v in N(u)
                nu = np.where(A[u] == 1)[0]
                nv = np.where(A[v] == 1)[0]
                # C4s through edge (u,v): paths u-a-b-v where a in N(u), b in N(v), a~b, a!=v, b!=u
                c = 0
                for a in nu:
                    if a == v:
                        continue
                    # b must be in N(v), b != u, b != a, and b in N(a)
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
        """Count total 4-cycles using A^2: C4 count = (1/8) * sum_{i!=j} (A^2[i,j])^2 - corrections for triangles etc.
        Actually: number of C4s = (1/8) * trace(A^4) - (1/2)*sum_i C(d_i,2) - ... 
        Simpler: for each unordered pair {i,j} with i!=j, the number of length-2 paths i-?-j is (A^2)[i,j].
        Each C4 contributes to 2 pairs of opposite vertices having 2 common neighbors... 
        Let's use: C4_count = sum over all unordered pairs {i,j} of C(common_neighbors(i,j), 2)
        where common_neighbors(i,j) = (A^2)[i,j] for i!=j.
        But this overcounts: each C4 is counted once per pair of opposite vertices. A C4 has 2 pairs of opposite vertices.
        So C4_count = (1/2) * sum_{i<j} C((A^2)[i,j], 2)
        Wait, that's not right either. Let me think again.
        
        A 4-cycle i-a-j-b-i: the pairs of opposite vertices are (i,j) and (a,b).
        For pair (i,j), the common neighbors include a and b (and possibly others).
        C(common(i,j), 2) counts all pairs of common neighbors, each pair defining a C4 through (i,j).
        Each C4 is counted exactly once for each of its 2 diagonal pairs.
        So total C4s = (1/2) * sum_{i<j} C(A2[i,j], 2)
        
        But wait - this also counts "degenerate" 4-cycles where a=b? No, C(k,2) requires distinct pairs.
        And it counts 4-cycles correctly as subgraphs (not necessarily induced).
        """
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
        """Number of C4s that include edge (u,v) as one of the 4 edges.
        A C4 containing edge (u,v) has form u-v-a-b-u or u-v-b-a-u.
        Count = number of paths v-a-b-u where a in N(v), b in N(u), a~b, a!=u, b!=v, a!=b.
        Equivalently: for each a in N(v)\\{u}, for each b in N(u)\\{v}, a!=b, a~b.
        """
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

    def c4_through_pair(u, v):
        """Number of C4s where (u,v) are opposite vertices (diagonal).
        = C(common_neighbors(u,v), 2) where common_neighbors = (A^2)[u,v] for u!=v.
        """
        if u == v:
            return 0
        k = (A @ A)[u, v]
        if k < 2:
            return 0
        return k * (k - 1) // 2

    def swap_delta_exact(u, v, x, y):
        """Compute exact change in total C4 count when removing edge (u,v) and adding edge (x,y).
        
        Total C4 = (1/2) * sum_{i<j} C(A2[i,j], 2)
        
        When we remove (u,v): A2 changes. A2[i,j] = sum_k A[i,k]*A[k,j].
        Removing edge (u,v) subtracts 1 from A2[u,v] and A2[v,u], and also affects A2[i,j] for all i,j 
        through the removed paths going through u-v or v-u.
        
        Specifically: A2_new[i,j] = A2[i,j] - A[i,u]*A[v,j] - A[i,v]*A[u,j]
        (since the only paths of length 2 that use edge (u,v) are i-u-v-j and i-v-u-j)
        
        Similarly for adding (x,y): A2_new[i,j] += A[i,x]*A[y,j] + A[i,y]*A[x,j]
        
        So the change in total C4 is:
        (1/2) * [sum C(A2'[i,j], 2) - sum C(A2[i,j], 2)]
        
        where A2' = A2 - delta_remove + delta_add
        delta_remove[i,j] = A[i,u]*A[v,j] + A[i,v]*A[u,j]  (this is 0,1,or 2)
        delta_add[i,j] = A[i,x]*A[y,j] + A[i,y]*A[x,j]
        
        We only need to compute the difference for pairs where A2[i,j] changes.
        """
        # Compute delta_remove[i,j] for all i,j
        # delta_remove[i,j] = A[i,u]*A[v,j] + A[i,v]*A[u,j]
        # This is non-zero only for i in N(u)∪N(v) and j in N(u)∪N(v)
        
        # Efficient: compute for all i,j using numpy
        Au = A[:, u]  # column u
        Av = A[:, v]  # column v
        delta_rm = np.outer(Au, Av) + np.outer(Av, Au)
        # But we need to be careful: when i=u or j=v etc., the edge (u,v) itself contributes
        # A[u,v]=1, so delta_rm[u,v] includes A[u,u]*A[v,v]+A[u,v]*A[u,v] = 0+1 = 1 (from second term)
        # and delta_rm[v,u] = A[v,u]*A[v,v]+A[v,v]*A[u,v] = 0+1 = 1
        # Also delta_rm[u,u] = A[u,u]*A[v,u]+A[u,v]*A[u,u] = 0+0 = 0
        # delta_rm[v,v] = 0
        # For i=u, j!=u,v: delta_rm[u,j] = A[u,u]*A[v,j]+A[u,v]*A[u,j] = 0+1*A[u,j] = A[u,j]
        #   This means: path u-v-j exists iff j in N(v) and j!=u. Correct: removing (u,v) removes path u-v-j.
        # For i in N(u), i!=u,v: delta_rm[i,j] = A[i,u]*A[v,j]+A[i,v]*A[u,j]
        #   = 1*A[v,j]+A[i,v]*A[u,j]
        #   The first term: path i-u-v-j exists. The second: path i-v-u-j exists (if i~v and u~j).
        
        # Similarly for add:
        Ax = A[:, x]
        Ay = A[:, y]
        delta_ad = np.outer(Ax, Ay) + np.outer(Ay, Ax)
        
        # A2_new = A2 - delta_rm + delta_ad
        # But we don't want to actually modify A yet; just compute the C4 change.
        # C4 = (1/2) * sum_{i<j} C(A2[i,j], 2)
        # Change = (1/2) * sum_{i<j} [C(A2[i,j] - delta_rm[i,j] + delta_ad[i,j], 2) - C(A2[i,j], 2)]
        # Only pairs where delta_rm[i,j] + delta_ad[i,j] != 0 contribute.
        
        A2 = A @ A
        total_change = 0
        # Find indices where delta changes
        delta_net = delta_ad - delta_rm
        # Get nonzero entries
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
        # Each C4 counted twice (once per diagonal pair), so divide by 2
        return total_change // 2

    # Local search: edge swaps
    edge_set = set((min(e[0],e[1]), max(e[0],e[1])) for e in edges)
    
    # Get current C4 count
    current_c4 = count_total_c4()
    
    # Precompute neighbor lists for speed
    def get_neighbors(u):
        return list(np.where(A[u] == 1)[0])
    
    best_c4 = current_c4
    best_edges = list(edges)
    
    # Local search with edge swaps
    max_iters = 500000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find best swap
        best_d = 0
        best_move = None
        
        # Candidate removals: all edges
        # Candidate additions: all non-edges
        # To speed up, precompute c4_contribution for each edge
        rem_cands = []
        for e in edges:
            c = c4_contribution(e[0], e[1])
            rem_cands.append((c, e))
        rem_cands.sort()
        
        add_cands = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                # Estimate: c4_through_pair is not directly the contribution, but a proxy
                # Use c4_contribution as proxy (it's the number of C4s the new edge would be in)
                c = c4_contribution(u, v)  # this works even if edge not present; counts potential C4s
                add_cands.append((c, (u, v)))
        add_cands.sort()
        
        # Try top K removals x top K additions
        K = 30
        improved = False
        for rc, re in rem_cands[:K]:
            if improved:
                break
            ru, rv = re
            for ac, ae in add_cands[:K]:
                au, av = ae
                if au == ru and av == rv:
                    continue
                if au == rv and av == ru:
                    continue
                # Compute exact delta
                d = swap_delta_exact(ru, rv, au, av)
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
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
        else:
            # No improving move found; try random perturbation occasionally
            if it % 100 == 0 and it > 0:
                # Random swap
                re = edges[np.random.randint(len(edges))]
                while True:
                    u = np.random.randint(n)
                    v = np.random.randint(n)
                    if u == v:
                        continue
                    k = (min(u,v), max(u,v))
                    if k not in edge_set:
                        break
                d = swap_delta_exact(re[0], re[1], u, v)
                # Accept with some probability (simulated annealing style)
                if d <= 0 or np.random.rand() < np.exp(-d / max(1, current_c4/10)):
                    A[re[0], re[1]] = 0
                    A[re[1], re[0]] = 0
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.remove(re)
                    edges.append((u, v))
                    edge_set.discard((min(re[0],re[1]), max(re[0],re[1])))
                    edge_set.add((min(u,v), max(u,v)))
                    current_c4 += d
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges = list(edges)
    
    # Use best found
    final_edges = best_edges
    return [[e[0], e[1]] for e in final_edges]
