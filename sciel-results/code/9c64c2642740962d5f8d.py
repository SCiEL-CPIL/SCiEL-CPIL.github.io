"""Combined: structured K_{8,8}+1 starts + targeted local search with incremental triangle tracking."""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0


def _build_triangles(adj):
    """Build list of triangles as sorted tuples (i,j,k) with i<j<k."""
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i][j]:
                continue
            for k in range(j + 1, N):
                if adj[i][k] and adj[j][k]:
                    tris.append((i, j, k))
    return tris


def _max_tri_sum(tris, deg):
    """Max degree sum over triangles."""
    if not tris:
        return 0
    best = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best


def _update_triangles(tris, adj, e_rm, e_add):
    """Incrementally update triangle list after removing e_rm=(u,v) and adding e_add=(x,y)."""
    u, v = e_rm
    x, y = e_add
    if u > v:
        u, v = v, u
    if x > y:
        x, y = y, x

    # Remove triangles containing edge (u,v)
    new_tris = []
    for t in tris:
        tset = set(t)
        if u in tset and v in tset:
            continue
        new_tris.append(t)

    # Add triangles containing edge (x,y)
    for w in range(N):
        if w == x or w == y:
            continue
        if adj[x][w] and adj[y][w]:
            tri = tuple(sorted((x, y, w)))
            new_tris.append(tri)

    return new_tris


def _get_worst_triangle(tris, deg):
    """Find the triangle with maximum degree sum."""
    if not tris:
        return None
    best_tri = None
    best_s = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best_s:
            best_s = s
            best_tri = (i, j, k)
    return best_tri


def _make_k88_plus1(adj_template, internal_edge, rng):
    """Create K_{8,8}+1 graph with specified internal edge."""
    adj = [[False] * N for _ in range(N)]
    # Standard bipartition: 0-7 vs 8-15
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    # Add internal edge
    a, b = internal_edge
    adj[a][b] = adj[b][a] = True

    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1

    tris = _build_triangles(adj)
    score = _max_tri_sum(tris, deg)
    return adj, deg, tris, score


def _local_search(adj, deg, tris, cur_score, t0, rng, iters_budget, target_bias=0.7):
    """Local search with incremental triangle updates and targeted move selection."""
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    start_temp = 2.5
    end_temp = 0.03
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    while iters < iters_budget:
        if (iters % 100 == 0) and (time.time() - t0 > DEADLINE):
            break

        if not edges or not nonedges:
            break

        # Targeted move: focus on worst triangle
        worst_tri = _get_worst_triangle(tris, deg)

        if worst_tri is not None and rng.random() < target_bias:
            i, j, k = worst_tri
            tri_edges = [(i, j), (i, k), (j, k)]
            # Choose which edge to remove from the worst triangle
            er = tri_edges[rng.randrange(3)]
            # Choose a non-edge to add that might help balance degrees
            # Prefer non-edges that connect low-degree vertices
            ea = nonedges[rng.randrange(len(nonedges))]
        else:
            er = edges[rng.randrange(len(edges))]
            ea = nonedges[rng.randrange(len(nonedges))]

        if er == ea:
            iters += 1
            continue

        u, v = er
        x, y = ea

        # Apply swap
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1
        deg[y] += 1

        # Update triangles incrementally
        new_tris = _update_triangles(tris, adj, er, ea)
        new_score = _max_tri_sum(new_tris, deg)

        # Accept/reject
        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.4
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-d / temp) > rng.random()

        if accept:
            cur_score = new_score
            tris = new_tris
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            # Update edge lists
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
        else:
            # Undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1
            deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score


def _perturb(adj, deg, tris, score, rng, num_flips=2):
    """Create a perturbed version of the current solution."""
    adj2 = [row[:] for row in adj]
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj2[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj2[i][j]]

    for _ in range(num_flips):
        if not edges or not nonedges:
            break
        er = edges[rng.randrange(len(edges))]
        ea = nonedges[rng.randrange(len(nonedges))]
        u, v = er
        x, y = ea
        adj2[u][v] = adj2[v][u] = False
        adj2[x][y] = adj2[y][x] = True

    deg2 = [0] * N
    for i in range(N):
        for j in range(N):
            if adj2[i][j]:
                deg2[i] += 1

    tris2 = _build_triangles(adj2)
    score2 = _max_tri_sum(tris2, deg2)
    return adj2, deg2, tris2, score2


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_adj = None
    best_deg = None
    best_score = float('inf')

    # Generate multiple structured starting points
    starts = []

    # Try all internal edges in K_{8,8} structure
    for a in range(8):
        for b in range(a + 1, 8):
            adj, deg, tris, score = _make_k88_plus1(None, (a, b), rng)
            starts.append((adj, deg, tris, score))

    # Sort by initial score and take the best few
    starts.sort(key=lambda x: x[3])
    top_starts = starts[:5]

    # Run local search on the best starts with time allocation
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed - 5.0  # safety margin

    if remaining > 10.0:
        # Allocate time: 40% to best, 25% to second, 15% to third, 20% to perturbed restarts
        allocations = [0.40, 0.25, 0.15, 0.20]

        for idx, (adj, deg, tris, score) in enumerate(top_starts[:3]):
            if time.time() - t0 > DEADLINE - 3.0:
                break

            budget = int(remaining * allocations[idx])
            budget = max(50, budget)

            b_adj, b_deg, b_score = _local_search(adj, deg, tris, score, t0, rng, budget, target_bias=0.7)

            if b_score < best_score:
                best_score = b_score
                best_adj = b_adj
                best_deg = b_deg

        # Perturbed restarts from the best solution
        if best_adj is not None and time.time() - t0 < DEADLINE - 10.0:
            restart_budget = int(remaining * allocations[3])
            num_restarts = min(3, max(1, restart_budget // 500))

            for r in range(num_restarts):
                if time.time() - t0 > DEADLINE - 5.0:
                    break

                # Perturb the best solution
                p_adj, p_deg, p_tris, p_score = _perturb(best_adj, best_deg, best_tris, best_score, rng, num_flips=2)

                # Run local search from perturbed state
                b_adj, b_deg, b_score = _local_search(p_adj, p_deg, p_tris, p_score, t0, rng, restart_budget // num_restarts, target_bias=0.8)

                if b_score < best_score:
                    best_score = b_score
                    best_adj = b_adj
                    best_deg = b_deg

    # Fallback if nothing found
    if best_adj is None:
        best_adj, best_deg, _, best_score = top_starts[0]

    # Extract edges
    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]

    # Safety: ensure exactly 65 edges
    if len(edges) != TARGET_EDGES:
        # Fallback to first start
        adj_f, deg_f, _, _ = top_starts[0]
        edges = [[i, j] for i in range(N) for j in range(i + 1, N) if adj_f[i][j]]

    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
    print("time:", time.time() - t0)
