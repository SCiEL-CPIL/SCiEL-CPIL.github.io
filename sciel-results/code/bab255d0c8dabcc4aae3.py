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
DEADLINE = 140.0  # safety margin before 150s kill


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
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best


def _direct_candidate(rng):
    """Build a candidate from the K_{8,8}+structure family.

    We take K_{8,8} (parts A={0..7}, B={8..15}) and add one 'internal' edge
    chosen to minimize triangle degree sum, then optionally apply small
    edge swaps that preserve edge count to improve balance.
    """
    adj = [[False] * N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    # add one internal edge; try all and keep best
    best_adj = None
    best_deg = None
    best_score = None
    for a in range(8):
        for b in range(a + 1, 8):
            adj[a][b] = adj[b][a] = True
            deg = [0] * N
            for i in range(N):
                for j in range(N):
                    if adj[i][j]:
                        deg[i] += 1
            tris = _build_tri_list(adj)
            s = _max_tri_sum(adj, deg, tris)
            if best_score is None or s < best_score:
                best_score = s
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            adj[a][b] = adj[b][a] = False
    return best_adj, best_deg, best_score


def _compute_score_np(adj_mat, deg_vec):
    """
    Compute the max triangle degree sum using NumPy vectorization.
    adj_mat: 16x16 numpy array (0/1)
    deg_vec: 16-element numpy array
    """
    # D_i + D_j + D_k for all i,j,k
    # Shapes: (16, 1, 1) + (1, 16, 1) + (1, 1, 16) -> (16, 16, 16)
    sums = deg_vec[:, np.newaxis, np.newaxis] + deg_vec[np.newaxis, :, np.newaxis] + deg_vec[np.newaxis, np.newaxis, :]
    
    # Mask for triangles: A_ij * A_jk * A_ik
    # A_ij is adj_mat
    # A_jk is adj_mat.T (since we need j,k and matrix is k,j? No, adj[j][k] is adj_mat[j][k])
    # Let's check indices.
    # We want i < j < k.
    # sums[i,j,k] corresponds to triangle (i,j,k).
    # Condition: adj[i][j] AND adj[j][k] AND adj[i][k].
    # adj_mat[i,j] is A_ij.
    # adj_mat[j,k] is A_jk.
    # adj_mat[i,k] is A_ik.
    
    # We can compute the product for all i,j,k and then mask out i>=j or j>=k or i>=k.
    # Actually, since the graph is undirected, A is symmetric.
    # A_ij * A_jk * A_ik is 1 iff i,j,k form a triangle.
    
    tri_mask = adj_mat[:, :, np.newaxis] * adj_mat[np.newaxis, :, :] * adj_mat[:, np.newaxis, :]
    
    # We only care about i < j < k to avoid counting permutations and self-loops (though diag is 0).
    # But max is invariant to permutations if we just take max over all valid triangles.
    # However, we must ensure we don't pick i=j=k etc.
    # The mask will be 0 if any edge is missing.
    # For i=j, A_ii=0, so mask is 0.
    # So we can just take max over the whole 16x16x16 tensor where mask is 1.
    
    # Set non-triangles to 0 (or -inf, but 0 is fine since degrees are positive)
    scored_sums = sums * tri_mask
    
    # The max value is the score. If no triangles, max is 0.
    return int(np.max(scored_sums))


def _fast_local_search(adj, deg, cur_score, t0, rng, iters_budget):
    """Incremental local search with vectorized scoring."""
    # Convert to numpy for fast scoring
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
        if (iters % 500 == 0) and (time.time() - t0 > DEADLINE):
            break

        if not edges or not nonedges:
            break

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
            accept = rng.random() < 0.5
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
    # Vectorized scoring is ~10-20x faster than Python loop scoring
    remaining = DEADLINE - (time.time() - t0) - 2.0
    if remaining > 5.0:
        # Estimate iterations: if 1 iter takes ~50us in numpy, we can do ~20,000/s
        # Let's aim for 100,000 iterations max, or whatever time allows
        iters_budget = max(1, int(remaining * 15000)) 
        best_adj, best_deg, best_score = _fast_local_search(
            adj, deg, cur_score, t0, rng, iters_budget
        )

    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]
    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
