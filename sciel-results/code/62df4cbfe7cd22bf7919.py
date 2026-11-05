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
    edge_set = set((min(e[0],e[1]), max(e[0],e[1])) for e in edges)

    def count_total_c4(A_mat):
        """C4_count = (1/2) * sum_{i<j} C(A2[i,j], 2)"""
        A2 = A_mat @ A_mat
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                k = A2[i, j]
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    def c4s_created_by_adding(A_mat, u, v):
        """Number of C4s created by adding edge (u,v) to graph A_mat.
        A C4 using new edge (u,v) has form u-a-b-v-u where a~b, a in N(u)\\{v}, b in N(v)\\{u}.
        Count = number of edges (a,b) with a in N(u)\\{v}, b in N(v)\\{u}, a~b, a!=b.
        """
        nu = np.where(A_mat[u] == 1)[0]
        nv = np.where(A_mat[v] == 1)[0]
        count = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if A_mat[a, b] == 1:
                    count += 1
        return count

    def swap_delta_exact(A_mat, A2_mat, rem_u, rem_v, add_u, add_v):
        """Exact C4 change for removing edge (rem_u, rem_v) and adding edge (add_u, add_v)."""
        # A2 changes: for each pair (i,j), A2[i,j] changes by:
        # -1 if i~rem_u and j~rem_v, or i~rem_v and j~rem_u (removal)
        # +1 if i~add_u and j~add_v, or i~add_v and j~add_u (addition)
        # Only pairs where the change is nonzero matter.
        # Affected indices: neighbors of rem_u, rem_v, add_u, add_v
        affected = set()
        for x in [rem_u, rem_v, add_u, add_v]:
            for y in np.where(A_mat[x] == 1)[0]:
                affected.add((min(x,y), max(x,y)))
        # Also include pairs (rem_u,rem_v), (add_u,add_v) themselves and cross pairs
        affected.add((min(rem_u,rem_v), max(rem_u,rem_v)))
        affected.add((min(add_u,add_v), max(add_u,add_v)))
        
        total_change = 0
        for (i, j) in affected:
            if i == j:
                continue
            # Compute old A2[i,j]
            old_val = A2_mat[i, j]
            # Compute delta
            delta = 0
            # Removal of (rem_u, rem_v): A2[i,j] decreases by 1 for each of:
            # i~rem_u and j~rem_v, i~rem_v and j~rem_u
            if A_mat[i, rem_u] == 1 and A_mat[j, rem_v] == 1:
                delta -= 1
            if A_mat[i, rem_v] == 1 and A_mat[j, rem_u] == 1:
                delta -= 1
            # Addition of (add_u, add_v): A2[i,j] increases by 1 for each of:
            if A_mat[i, add_u] == 1 and A_mat[j, add_v] == 1:
                delta += 1
            if A_mat[i, add_v] == 1 and A_mat[j, add_u] == 1:
                delta += 1
            if delta == 0:
                continue
            new_val = old_val + delta
            old_c = old_val * (old_val - 1) // 2 if old_val >= 2 else 0
            new_c = new_val * (new_val - 1) // 2 if new_val >= 2 else 0
            total_change += (new_c - old_c)
        return total_change // 2

    # Phase 1: Find the best single edge to add to the 90-edge polarity graph
    # The polarity graph is C4-free, so C4s after adding edge (u,v) = c4s_created_by_adding(A, u, v)
    best_add = None
    best_c4_add = None
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in edge_set:
                continue
            c = c4s_created_by_adding(A, u, v)
            if best_c4_add is None or c < best_c4_add:
                best_c4_add = c
                best_add = (u, v)

    # Apply best addition
    A[best_add[0], best_add[1]] = 1
    A[best_add[1], best_add[0]] = 1
    edges.append(best_add)
    edge_set.add((min(best_add[0], best_add[1]), max(best_add[0], best_add[1])))

    current_c4 = best_c4_add
    A2 = A @ A
    best_c4 = current_c4
    best_edges = list(edges)

    # Phase 2: Local search with edge swaps
    # Precompute neighbor lists for speed
    def get_neighbors(A_mat, x):
        return np.where(A_mat[x] == 1)[0]

    # Fast C4 count through a specific edge
    def c4_through_edge(A_mat, u, v):
        """Number of C4s that use edge (u,v)."""
        nu = get_neighbors(A_mat, u)
        nv = get_neighbors(A_mat, v)
        count = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if A_mat[a, b] == 1:
                    count += 1
        return count

    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        best_d = 0
        best_move = None

        # Get C4 contribution of each edge for removal candidates
        edge_scores = []
        for e in edges:
            c = c4_through_edge(A, e[0], e[1])
            edge_scores.append((c, e))
        edge_scores.sort()

        # Try removing top-K edges and adding best non-edge
        K = 30
        for _, rem_e in edge_scores[:K]:
            ru, rv = rem_e
            # For each non-edge, compute delta
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    if u == ru and v == rv:
                        continue
                    if u == rv and v == ru:
                        continue
                    d = swap_delta_exact(A, A2, ru, rv, u, v)
                    if d < best_d:
                        best_d = d
                        best_move = (rem_e, (u, v))

        if best_move is not None and best_d < 0:
            rem_e, add_e = best_move
            ru, rv = rem_e
            au, av = add_e
            A[ru, rv] = 0
            A[rv, ru] = 0
            A[au, av] = 1
            A[av, au] = 1
            edges.remove(rem_e)
            edges.append(add_e)
            edge_set.discard((min(ru,rv), max(ru,rv)))
            edge_set.add((min(au,av), max(au,av)))
            current_c4 += best_d
            A2 = A @ A
            no_improve = 0
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
        else:
            no_improve += 1
            # Perturbation: random swap
            if no_improve % 10 == 0:
                rem_e = edges[np.random.randint(len(edges))]
                ru, rv = rem_e
                # Find a random non-edge
                while True:
                    u = np.random.randint(n)
                    v = np.random.randint(n)
                    if u == v:
                        continue
                    k = (min(u,v), max(u,v))
                    if k not in edge_set:
                        break
                d = swap_delta_exact(A, A2, ru, rv, u, v)
                # Simulated annealing accept
                if d <= 0 or np.random.rand() < np.exp(-d / max(1.0, current_c4 / 5.0)):
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.remove(rem_e)
                    edges.append((u, v))
                    edge_set.discard((min(ru,rv), max(ru,rv)))
                    edge_set.add((min(u,v), max(u,v)))
                    current_c4 += d
                    A2 = A @ A
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges = list(edges)
                        no_improve = 0

    # Final exhaustive polish
    if time.time() < deadline - 5.0:
        A2 = A @ A
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
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
                        d = swap_delta_exact(A, A2, ru, rv, u, v)
                        if d < 0:
                            A[ru, rv] = 0
                            A[rv, ru] = 0
                            A[u, v] = 1
                            A[v, u] = 1
                            edges.remove(re)
                            edges.append((u, v))
                            edge_set.discard((min(ru,rv), max(ru,rv)))
                            edge_set.add((min(u,v), max(u,v)))
                            current_c4 += d
                            A2 = A @ A
                            if current_c4 < best_c4:
                                best_c4 = current_c4
                                best_edges = list(edges)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    final_edges = best_edges
    return [[e[0], e[1]] for e in final_edges]
