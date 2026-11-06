"""Combined approach: K_{8,8}+e structure with targeted local search.

Key improvements:
- Bitmask-based triangle enumeration for faster scoring
- More aggressive targeted moves prioritizing worst triangles
- Final deterministic polish pass for systematic improvement
- Better time management
"""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 140.0  # safety margin before 150s kill


def _score_from_bitmasks(adj_masks, deg):
    """Compute max degree sum over all triangles using bitmasks."""
    best = 0
    for i in range(N):
        mi = adj_masks[i]
        # Only consider j > i
        j = i + 1
        while j < N:
            if mi & (1 << j):
                common = mi & adj_masks[j]
                # Only consider k > j
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = common.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
                    common &= common - 1
            j += 1
    return best


def _find_worst_triangles(adj_masks, deg, k=5):
    """Find the k worst triangles (highest degree sum)."""
    tris = []
    for i in range(N):
        mi = adj_masks[i]
        j = i + 1
        while j < N:
            if mi & (1 << j):
                common = mi & adj_masks[j]
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = common.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    tris.append((s, i, j, k))
                    common &= common - 1
            j += 1
    tris.sort(reverse=True)
    return tris[:k]


def _masks_to_adj(adj_masks):
    """Convert bitmasks to adjacency list."""
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        m = adj_masks[i]
        while m:
            j = m.bit_length() - 1
            adj[i][j] = True
            m &= m - 1
    return adj


def _adj_to_masks(adj):
    """Convert adjacency list to bitmasks."""
    masks = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks


def _direct_candidate(internal_edge=None):
    """Build a candidate from the K_{8,8}+structure family."""
    # Start with K_{8,8}
    adj_masks = [0] * N
    for i in range(8):
        for j in range(8, 16):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
    
    if internal_edge is None:
        # Try all internal edges in part A and keep best
        best_masks = None
        best_deg = None
        best_score = None
        for a in range(8):
            for b in range(a + 1, 8):
                adj_masks[a] |= (1 << b)
                adj_masks[b] |= (1 << a)
                deg = [bin(adj_masks[i]).count('1') for i in range(N)]
                s = _score_from_bitmasks(adj_masks, deg)
                if best_score is None or s < best_score:
                    best_score = s
                    best_masks = adj_masks[:]
                    best_deg = deg[:]
                adj_masks[a] &= ~(1 << b)
                adj_masks[b] &= ~(1 << a)
        return best_masks, best_deg, best_score
    else:
        a, b = internal_edge
        adj_masks[a] |= (1 << b)
        adj_masks[b] |= (1 << a)
        deg = [bin(adj_masks[i]).count('1') for i in range(N)]
        s = _score_from_bitmasks(adj_masks, deg)
        return adj_masks, deg, s


def _local_search(adj_masks, deg, cur_score, t0, rng, iters_budget, targeted=True):
    """Local search with targeted moves."""
    best_score = cur_score
    best_masks = adj_masks[:]
    best_deg = deg[:]

    start_temp = 2.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    stagnant = 0
    STAGNANT_LIMIT = 1000

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    while iters < iters_budget:
        if (iters % 200 == 0) and (time.time() - t0 > DEADLINE):
            break

        # Get current edges and non-edges
        edges = []
        nonedges = []
        for i, j in all_edges:
            if adj_masks[i] & (1 << j):
                edges.append((i, j))
            else:
                nonedges.append((i, j))

        if not edges or not nonedges:
            break

        # Choose move
        if targeted and stagnant > STAGNANT_LIMIT and rng.random() < 0.8:
            # Targeted: try to break worst triangles
            worst_tris = _find_worst_triangles(adj_masks, deg, k=3)
            if worst_tris:
                s_w, i, j, k = worst_tris[0]
                # Try removing one edge from the worst triangle
                tri_edges = [(i, j), (i, k), (j, k)]
                er = tri_edges[rng.randrange(len(tri_edges))]
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
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

        adj_masks[x] |= (1 << y)
        adj_masks[y] |= (1 << x)
        deg[x] += 1
        deg[y] += 1

        new_score = _score_from_bitmasks(adj_masks, deg)

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
                best_masks = adj_masks[:]
                best_deg = deg[:]
                stagnant = 0
            else:
                stagnant += 1
        else:
            # Undo
            adj_masks[x] &= ~(1 << y)
            adj_masks[y] &= ~(1 << x)
            deg[x] -= 1
            deg[y] -= 1

            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            stagnant += 1

        iters += 1
        temp *= cool

    return best_masks, best_deg, best_score


def _deterministic_polish(adj_masks, deg, cur_score, t0, max_iters=10000):
    """Final deterministic polish: systematically try swaps around worst triangles."""
    best_score = cur_score
    best_masks = adj_masks[:]
    best_deg = deg[:]

    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    iters = 0
    no_improve = 0
    NO_IMPROVE_LIMIT = 2000

    while iters < max_iters:
        if (iters % 100 == 0) and (time.time() - t0 > DEADLINE):
            break

        worst_tris = _find_worst_triangles(adj_masks, deg, k=5)
        if not worst_tris:
            break

        s_w, i, j, k = worst_tris[0]
        if s_w <= 24:
            break

        found_improvement = False

        # Try each edge of the worst triangle for removal
        tri_edges = [(i, j), (i, k), (j, k)]
        for er in tri_edges:
            u, v = er
            if not (adj_masks[u] & (1 << v)):
                continue

            # Try all non-edges to add
            for a, b in all_edges:
                if (a, b) == er:
                    continue
                if adj_masks[a] & (1 << b):
                    continue

                # Apply
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                deg[u] -= 1
                deg[v] -= 1

                adj_masks[a] |= (1 << b)
                adj_masks[b] |= (1 << a)
                deg[a] += 1
                deg[b] += 1

                new_score = _score_from_bitmasks(adj_masks, deg)

                if new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_masks = adj_masks[:]
                        best_deg = deg[:]
                    found_improvement = True
                    break
                else:
                    # Undo
                    adj_masks[a] &= ~(1 << b)
                    adj_masks[b] &= ~(1 << a)
                    deg[a] -= 1
                    deg[b] -= 1

                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1

            if found_improvement:
                break

        if not found_improvement:
            no_improve += 1
            if no_improve > NO_IMPROVE_LIMIT:
                break
        else:
            no_improve = 0

        iters += 1

    return best_masks, best_deg, best_score


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Generate diverse starting points
    start_candidates = []
    
    # Best internal edge in part A
    adj_masks, deg, score = _direct_candidate(internal_edge=None)
    start_candidates.append((adj_masks, deg, score))
    
    # Try a few specific internal edges
    for trial in range(3):
        a = rng.randrange(8)
        b = rng.randrange(a + 1, 8)
        masks2, deg2, score2 = _direct_candidate(internal_edge=(a, b))
        start_candidates.append((masks2, deg2, score2))

    # Pick the best starting point
    start_candidates.sort(key=lambda x: x[2])
    adj_masks, deg, cur_score = start_candidates[0]
    
    best_masks = adj_masks[:]
    best_deg = deg[:]
    best_score = cur_score

    # Main local search phase
    remaining = DEADLINE - (time.time() - t0) - 10.0  # leave 10s for polish + output
    if remaining > 5.0:
        iters_budget = int(remaining * 2000)  # ~2000 iters/sec with bitmasks
        best_masks, best_deg, best_score = _local_search(
            adj_masks, deg, cur_score, t0, rng, iters_budget, targeted=True
        )

    # Deterministic polish phase
    remaining = DEADLINE - (time.time() - t0) - 2.0
    if remaining > 3.0:
        polish_iters = int(remaining * 300)
        best_masks, best_deg, best_score = _deterministic_polish(
            best_masks, best_deg, best_score, t0, max_iters=polish_iters
        )

    edges = _masks_to_adj(best_masks)
    edge_list = [[i, j] for i in range(N) for j in range(i + 1, N) if edges[i][j]]
    return {'n': N, 'edges': edge_list}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
