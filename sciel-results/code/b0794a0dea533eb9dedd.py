import random
import time
import math
import sys
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # Safety margin before 150s kill
MASK_ALL = (1 << N) - 1

# Precompute bit masks for each vertex
BM = [1 << i for i in range(N)]

def _score_from_mask(emask):
    """Compute max degree sum over all triangles using bitset operations.
    emask: integer whose bits represent edges (bit i*N+j set iff edge i-j present, i<j).
    Returns the max degree sum, or 0 if no triangles."""
    deg = [0] * N
    adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if (emask >> (i * N + j)) & 1:
                adj[i] |= BM[j]
                adj[j] |= BM[i]
                deg[i] += 1
                deg[j] += 1
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # common has bits for all k adjacent to both i and j
                while common:
                    lsb = common & (-common)
                    k = lsb.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
                    common ^= lsb
    return best

def _mask_to_edges(emask):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (emask >> (i * N + j)) & 1:
                edges.append([i, j])
    return edges

def _edges_to_mask(edges):
    m = 0
    for u, v in edges:
        if u > v:
            u, v = v, u
        m |= (1 << (u * N + v))
    return m

def _random_mask_with_k_edges(rng, k):
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs)
    m = 0
    for u, v in all_pairs[:k]:
        m |= (1 << (u * N + v))
    return m

def _make_k88_plus(rng, internal_edge):
    """K_{8,8} (parts 0-7, 8-15) plus one internal edge."""
    m = 0
    for i in range(8):
        for j in range(8, 16):
            m |= (1 << (i * N + j))
    u, v = internal_edge
    if u > v:
        u, v = v, u
    m |= (1 << (u * N + v))
    return m

def _make_bipartite_plus(rng):
    """Random bipartite-like structure plus a few internal edges."""
    # Split into two groups of 8, connect mostly across, add some within
    m = 0
    cross_pairs = [(i, j) for i in range(8) for j in range(8, 16)]
    rng.shuffle(cross_pairs)
    # take 60 cross edges
    for u, v in cross_pairs[:60]:
        m |= (1 << (u * N + v))
    # add 5 internal edges
    internal = [(i, j) for i in range(N) for j in range(i + 1, N)
                if not (i < 8 and j >= 8)]
    rng.shuffle(internal)
    for u, v in internal[:5]:
        m |= (1 << (u * N + v))
    return m

def _hill_climb(emask, max_time, seed=12345, start_temp=2.5, end_temp=0.05):
    """Simulated annealing on the edge mask with fast bitset scoring."""
    t0 = time.time()
    rng = random.Random(seed)

    cur_score = _score_from_mask(emask)
    best_score = cur_score
    best_mask = emask

    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (emask >> (i * N + j)) & 1:
                edges.append((i, j))
            else:
                nonedges.append((i, j))

    max_iters = 2000000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    iters = 0
    stall_count = 0
    stall_threshold = 15000
    score_count = 0

    while iters < max_iters:
        if iters % 500 == 0:
            if time.time() - t0 > max_time:
                break

        u, v = edges[rng.randrange(len(edges))]
        x, y = nonedges[rng.randrange(len(nonedges))]

        if (u == x and v == y) or (u == y and v == x):
            iters += 1
            temp *= cooling_rate
            continue

        # Apply swap to mask
        emask2 = emask
        emask2 ^= (1 << (u * N + v))
        emask2 ^= (1 << (x * N + y))

        new_score = _score_from_mask(emask2)
        score_count += 1

        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.25
        else:
            delta = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()

        if accept:
            emask = emask2
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_mask = emask
                stall_count = 0
            else:
                stall_count += 1

            edges.remove((u, v))
            nonedges.append((u, v))
            nonedges.remove((x, y))
            edges.append((x, y))
        else:
            pass  # mask unchanged

        iters += 1
        temp *= cooling_rate

        if stall_count > stall_threshold and temp < 0.1:
            break

    return best_mask, best_score

def _polish(mask, max_time, seed=99999):
    """Final high-precision polish: aggressive local search with very low temperature."""
    t0 = time.time()
    rng = random.Random(seed)
    cur_score = _score_from_mask(mask)
    best_score = cur_score
    best_mask = mask

    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (mask >> (i * N + j)) & 1:
                edges.append((i, j))
            else:
                nonedges.append((i, j))

    iters = 0
    max_iters = 1500000
    while iters < max_iters:
        if iters % 500 == 0:
            if time.time() - t0 > max_time:
                break
        u, v = edges[rng.randrange(len(edges))]
        x, y = nonedges[rng.randrange(len(nonedges))]
        if (u == x and v == y) or (u == y and v == x):
            iters += 1
            continue
        emask2 = mask
        emask2 ^= (1 << (u * N + v))
        emask2 ^= (1 << (x * N + y))
        new_score = _score_from_mask(emask2)
        if new_score < cur_score:
            mask = emask2
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_mask = mask
            edges.remove((u, v))
            nonedges.append((u, v))
            nonedges.remove((x, y))
            edges.append((x, y))
        iters += 1
    return best_mask, best_score

def construct():
    t0 = time.time()
    rng = random.Random(2024)

    best_overall_score = float('inf')
    best_overall_mask = None

    # Generate diverse starting masks
    starts = []
    # K_{8,8} + various internal edges
    internal_candidates = [(0, 1), (0, 7), (1, 2), (8, 9), (14, 15), (3, 5), (9, 12)]
    for e in internal_candidates:
        starts.append(_make_k88_plus(rng, e))
    # Random bipartite-plus
    for _ in range(3):
        starts.append(_make_bipartite_plus(rng))
    # Fully random
    for _ in range(3):
        starts.append(_random_mask_with_k_edges(rng, TARGET_EDGES))

    # Run SA on each start, allocating time proportionally
    n_starts = len(starts)
    per_start = (DEADLINE - (time.time() - t0)) / n_starts

    for idx, m in enumerate(starts):
        elapsed = time.time() - t0
        if elapsed > DEADLINE:
            break
        remaining = DEADLINE - elapsed
        budget = min(per_start, remaining)
        # Leave time for polish
        sa_budget = max(1.0, budget - 5.0)
        best_mask, best_score = _hill_climb(m, sa_budget, seed=1000 + idx * 7)
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_mask = best_mask

    # Final polish pass on the best found
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 1.0 and best_overall_mask is not None:
        polish_budget = DEADLINE - elapsed - 1.0
        best_overall_mask, best_overall_score = _polish(best_overall_mask, polish_budget)

    edges = _mask_to_edges(best_overall_mask)
    return {'n': N, 'edges': edges}
