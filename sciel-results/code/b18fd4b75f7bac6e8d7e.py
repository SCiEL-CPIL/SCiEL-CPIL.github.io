"""Direct constructive approach + fast incremental local search.

Structure insight: 65 edges = 64 (K_{8,8}) + 1. Triangles in K_{8,8}+e
involve only the internal edge's endpoints and their common neighbors.
We search over structured perturbations and use fast incremental scoring.
"""
import time
import random
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 148.0  # safety margin before 150s kill


def _build_tri_list(adj):
    """Return list of triangles as (i, j, k) with i<j<k."""
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i][j]:
                continue
            for k in range(j + 1, N):
                if adj[i][k] and adj[j][k]:
                    tris.append((i, j, k))
    return tris


def _max_tri_sum(adj, deg, tris):
    best = 0
    best_tri = None
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
            best_tri = (i, j, k)
    return best, best_tri


def _direct_candidate(rng):
    """Build a candidate from the K_{8,8}+structure family."""
    adj = [[False] * N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    
    adj_np = np.array(adj, dtype=np.int8)
    best_adj = None
    best_deg = None
    best_score = None
    
    for a in range(8):
        for b in range(a + 1, 8):
            adj_np[a, b] = 1
            adj_np[b, a] = 1
            deg_np = np.sum(adj_np, axis=1)
            
            # Calculate score using vectorized method
            tri_mask = adj_np[:, :, np.newaxis] * adj_np[np.newaxis, :, :] * adj_np[:, np.newaxis, :]
            if np.sum(tri_mask) > 0:
                deg_sums = deg_np[:, np.newaxis, np.newaxis] + deg_np[np.newaxis, :, np.newaxis] + deg_np[np.newaxis, np.newaxis, :]
                scored = deg_sums * tri_mask
                s = int(np.max(scored))
            else:
                s = 0
                
            if best_score is None or s < best_score:
                best_score = s
                best_adj = [row[:] for row in adj]
                # Update best_adj to reflect the change
                best_adj[a][b] = True
                best_adj[b][a] = True
                best_deg = deg_np.tolist()
            
            adj_np[a, b] = 0
            adj_np[b, a] = 0
            
    return best_adj, best_deg, best_score


def _compute_score_np(adj_mat, deg_vec):
    """
    Compute the max triangle degree sum using NumPy vectorization.
    adj_mat: 16x16 numpy array (0/1)
    deg_vec: 16-element numpy array
    """
    sums = deg_vec[:, np.newaxis, np.newaxis] + deg_vec[np.newaxis, :, np.newaxis] + deg_vec[np.newaxis, np.newaxis, :]
    tri_mask = adj_mat[:, :, np.newaxis] * adj_mat[np.newaxis, :, :] * adj_mat[:, np.newaxis, :]
    scored_sums = sums * tri_mask
    return int(np.max(scored_sums))


def _get_worst_triangle_np(adj_np, deg_np):
    """Find the triangle with maximum degree sum using NumPy."""
    # adj_np is 16x16, deg_np is 16x1
    # Create 3D mask for triangles (i, j, k)
    # tri[i,j,k] is 1 if edges (i,j), (j,k), (k,i) exist
    tri_mask = adj_np[:, :, np.newaxis] * adj_np[np.newaxis, :, :] * adj_np[:, np.newaxis, :]
    if np.sum(tri_mask) == 0:
        return None
    
    deg_sums = deg_np[:, np.newaxis, np.newaxis] + deg_np[np.newaxis, :, np.newaxis] + deg_np[np.newaxis, np.newaxis, :]
    scored = deg_sums * tri_mask
    
    max_score = np.max(scored)
    if max_score == 0:
        return None
        
    idx = np.argmax(scored)
    i, j, k = np.unravel_index(idx, scored.shape)
    return (i, j, k)


def _fast_local_search(adj, deg, cur_score, t0, rng, iters_budget):
    """Incremental local search with biased moves toward worst triangle."""
    adj_np = np.array(adj, dtype=np.int8)
    deg_np = np.array(deg, dtype=np.int8)
    
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    start_temp = 3.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    while iters < iters_budget:
        if (iters % 100 == 0) and (time.time() - t0 > DEADLINE):
            break

        if not edges or not nonedges:
            break

        # Biased move selection: 60% chance to target worst triangle
        worst_tri = _get_worst_triangle_np(adj_np, deg_np)
        
        if worst_tri is not None and rng.random() < 0.6:
            # Pick an edge from the worst triangle to remove
            i, j, k = worst_tri
            tri_edges = [(i, j), (i, k), (j, k)]
            er = tri_edges[rng.randrange(3)]
            if er not in edges:
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

        # apply
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1; deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1; deg[y] += 1
        
        # Update numpy arrays
        adj_np[u, v] = 0; adj_np[v, u] = 0
        adj_np[x, y] = 1; adj_np[y, x] = 1
        deg_np[u] -= 1; deg_np[v] -= 1
        deg_np[x] += 1; deg_np[y] += 1

        new_score = _compute_score_np(adj_np, deg_np)

        accept = False
        if new_score < cur_score:
            accept = True
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
                best_deg = deg[:]
            # update lists
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
        else:
            # undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1; deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1; deg[v] += 1
            
            # Undo numpy arrays
            adj_np[x, y] = 0; adj_np[y, x] = 0
            adj_np[u, v] = 1; adj_np[v, u] = 1
            deg_np[x] -= 1; deg_np[y] -= 1
            deg_np[u] += 1; deg_np[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Start from direct construction
    adj, deg, score = _direct_candidate(rng)
    cur_score = score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_score = score

    # Run fast local search with a generous but time-capped budget
    remaining = DEADLINE - (time.time() - t0) - 2.0
    if remaining > 5.0:
        iters_budget = max(1, int(remaining * 10000)) 
        best_adj, best_deg, best_score = _fast_local_search(
            adj, deg, cur_score, t0, rng, iters_budget
        )

    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]
    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
