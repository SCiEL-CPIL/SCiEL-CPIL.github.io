"""Combined approach: K_{8,8}+e structure with targeted local search.

Key improvements over current:
- Use adjacency sets for O(1) edge lookups during scoring
- Targeted moves: prioritize removing edges from worst triangles
- Multiple diverse starting points
- Final polish pass focused on worst triangles
"""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before 150s kill


def _score_from_sets(adj_sets, deg):
    """Compute max degree sum over all triangles using adjacency sets."""
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if j not in adj_sets[i]:
                continue
            common = adj_sets[i] & adj_sets[j]
            for k in common:
                if k > j:  # ensure i < j < k
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best


def _find_worst_triangles(adj_sets, deg, k=5):
    """Find the k worst triangles (highest degree sum)."""
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if j not in adj_sets[i]:
                continue
            common = adj_sets[i] & adj_sets[j]
            for k in common:
                if k > j:
                    s = deg[i] + deg[j] + deg[k]
                    tris.append((s, i, j, k))
    tris.sort(reverse=True)
    return tris[:k]


def _direct_candidate(rng, internal_edge=None):
    """Build a candidate from the K_{8,8}+structure family."""
    adj = [[False] * N for _ in range(N)]
    adj_sets = [set() for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
            adj_sets[i].add(j)
            adj_sets[j].add(i)
    
    if internal_edge is None:
        # Try all internal edges in part A and keep best
        best_adj = None
        best_adj_sets = None
        best_deg = None
        best_score = None
        for a in range(8):
            for b in range(a + 1, 8):
                adj[a][b] = adj[b][a] = True
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                deg = [len(adj_sets[i]) for i in range(N)]
                s = _score_from_sets(adj_sets, deg)
                if best_score is None or s < best_score:
                    best_score = s
                    best_adj = [row[:] for row in adj]
                    best_adj_sets = [set(s) for s in adj_sets]
                    best_deg = deg[:]
                adj[a][b] = adj[b][a] = False
                adj_sets[a].discard(b)
                adj_sets[b].discard(a)
        return best_adj, best_adj_sets, best_deg, best_score
    else:
        a, b = internal_edge
        adj[a][b] = adj[b][a] = True
        adj_sets[a].add(b)
        adj_sets[b].add(a)
        deg = [len(adj_sets[i]) for i in range(N)]
        s = _score_from_sets(adj_sets, deg)
        return adj, adj_sets, deg, s


def _local_search(adj, adj_sets, deg, cur_score, t0, rng, iters_budget, targeted=True):
    """Local search with optional targeted moves."""
    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_adj_sets = [set(s) for s in adj_sets]
    best_deg = deg[:]

    start_temp = 2.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    stagnant = 0
    STAGNANT_LIMIT = 2000

    while iters < iters_budget:
        if (iters % 500 == 0) and (time.time() - t0 > DEADLINE):
            break

        # Get edges and non-edges
        edges = []
        nonedges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j]:
                    edges.append((i, j))
                else:
                    nonedges.append((i, j))

        if not edges or not nonedges:
            break

        # Choose move
        if targeted and stagnant > STAGNANT_LIMIT and rng.random() < 0.7:
            # Targeted: try to break worst triangles
            worst_tris = _find_worst_triangles(adj_sets, deg, k=3)
            if worst_tris:
                s_w, i, j, k = worst_tris[0]
                # Try removing one edge from the worst triangle
                tri_edges = [(i, j), (i, k), (j, k)]
                er = tri_edges[rng.randrange(len(tri_edges))]
                # Try adding a non-edge that doesn't create new high-sum triangles
                ea = nonedges[rng.randrange(len(nonedges))]
            else:
                er = edges[rng.randrange(len(edges))]
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
        adj_sets[u].discard(v)
        adj_sets[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1

        adj[x][y] = adj[y][x] = True
        adj_sets[x].add(y)
        adj_sets[y].add(x)
        deg[x] += 1
        deg[y] += 1

        new_score = _score_from_sets(adj_sets, deg)

        # Accept/reject
        accept = False
        if new_score < cur_score:
            accept = True
            stagnant = 0
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-d / temp) > rng.random()

        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_adj_sets = [set(s) for s in adj_sets]
                best_deg = deg[:]
                stagnant = 0
            else:
                stagnant += 1
        else:
            # Undo
            adj[x][y] = adj[y][x] = False
            adj_sets[x].discard(y)
            adj_sets[y].discard(x)
            deg[x] -= 1
            deg[y] -= 1

            adj[u][v] = adj[v][u] = True
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            deg[u] += 1
            deg[v] += 1
            stagnant += 1

        iters += 1
        temp *= cool

    return best_adj, best_adj_sets, best_deg, best_score


def _polish_pass(adj, adj_sets, deg, cur_score, t0, rng, max_iters=5000):
    """Final polish: focus on breaking worst triangles."""
    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_adj_sets = [set(s) for s in adj_sets]
    best_deg = deg[:]

    iters = 0
    while iters < max_iters:
        if (iters % 100 == 0) and (time.time() - t0 > DEADLINE):
            break

        worst_tris = _find_worst_triangles(adj_sets, deg, k=3)
        if not worst_tris:
            break

        s_w, i, j, k = worst_tris[0]
        if s_w <= 24:
            break

        # Try each edge of the worst triangle for removal
        tri_edges = [(i, j), (i, k), (j, k)]
        found_improvement = False

        for er in tri_edges:
            u, v = er
            # Try several non-edges to add
            nonedges = []
            for a in range(N):
                for b in range(a + 1, N):
                    if not adj[a][b] and (a, b) != er:
                        nonedges.append((a, b))

            if not nonedges:
                continue

            for _ in range(5):  # Try 5 random non-edges
                if time.time() - t0 > DEADLINE:
                    break
                ea = nonedges[rng.randrange(len(nonedges))]
                x, y = ea

                # Apply
                adj[u][v] = adj[v][u] = False
                adj_sets[u].discard(v)
                adj_sets[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1

                adj[x][y] = adj[y][x] = True
                adj_sets[x].add(y)
                adj_sets[y].add(x)
                deg[x] += 1
                deg[y] += 1

                new_score = _score_from_sets(adj_sets, deg)

                if new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                        best_adj_sets = [set(s) for s in adj_sets]
                        best_deg = deg[:]
                    found_improvement = True
                    iters += 1
                    break
                else:
                    # Undo
                    adj[x][y] = adj[y][x] = False
                    adj_sets[x].discard(y)
                    adj_sets[y].discard(x)
                    deg[x] -= 1
                    deg[y] -= 1

                    adj[u][v] = adj[v][u] = True
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    deg[u] += 1
                    deg[v] += 1

        if not found_improvement:
            iters += 1

    return best_adj, best_adj_sets, best_deg, best_score


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Generate diverse starting points
    start_candidates = []
    
    # Best internal edge in part A
    adj, adj_sets, deg, score = _direct_candidate(rng, internal_edge=None)
    start_candidates.append((adj, adj_sets, deg, score))
    
    # Try a few specific internal edges with small random perturbations
    for trial in range(3):
        a = rng.randrange(8)
        b = rng.randrange(a + 1, 8)
        adj2, adj_sets2, deg2, score2 = _direct_candidate(rng, internal_edge=(a, b))
        start_candidates.append((adj2, adj_sets2, deg2, score2))

    # Pick the best starting point
    start_candidates.sort(key=lambda x: x[3])
    adj, adj_sets, deg, cur_score = start_candidates[0]
    
    best_adj = [row[:] for row in adj]
    best_adj_sets = [set(s) for s in adj_sets]
    best_deg = deg[:]
    best_score = cur_score

    # Main local search phase
    remaining = DEADLINE - (time.time() - t0) - 5.0  # leave 5s for polish + output
    if remaining > 10.0:
        iters_budget = int(remaining * 1500)  # ~1500 iters/sec
        best_adj, best_adj_sets, best_deg, best_score = _local_search(
            adj, adj_sets, deg, cur_score, t0, rng, iters_budget, targeted=True
        )

    # Polish phase
    remaining = DEADLINE - (time.time() - t0) - 2.0
    if remaining > 3.0:
        polish_iters = int(remaining * 500)
        best_adj, best_adj_sets, best_deg, best_score = _polish_pass(
            best_adj, best_adj_sets, best_deg, best_score, t0, rng, max_iters=polish_iters
        )

    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]
    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
