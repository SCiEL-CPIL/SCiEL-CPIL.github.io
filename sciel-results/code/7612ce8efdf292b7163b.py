"""Optimized construction via simulated annealing on 16-vertex graphs with 65 edges."""
import time
import random
import math

_DEADLINE = 145.0  # safety margin below 150s wall-clock


def _score_from_adj(adj):
    """Compute max triangle degree-sum for a given adjacency matrix."""
    n = len(adj)
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                deg[i] += 1
                deg[j] += 1
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best


def construct():
    start = time.time()
    n = 16
    target_edges = 65
    rng = random.Random(42)

    # Start from K_{8,8} + one internal edge
    adj = [[0] * n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = adj[v][u] = 1
    adj[0][1] = adj[1][0] = 1

    best_adj = [row[:] for row in adj]
    best_score = _score_from_adj(adj)

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Simulated annealing: swap edges (remove one, add another) to keep 65 edges
    T_start = 8.0
    T_end = 0.05
    total_steps = 200000
    steps = 0

    while steps < total_steps:
        if time.time() > _DEADLINE:
            break

        # Pick two random edges to swap
        e1 = rng.choice(all_edges)
        e2 = rng.choice(all_edges)
        if e1 == e2:
            steps += 1
            continue

        # We need e1 present and e2 absent (or vice versa)
        if adj[e1[0]][e1[1]] and not adj[e2[0]][e2[1]]:
            pass
        elif adj[e2[0]][e2[1]] and not adj[e1[0]][e1[1]]:
            e1, e2 = e2, e1
        else:
            steps += 1
            continue

        # Apply swap
        adj[e1[0]][e1[1]] = 0
        adj[e1[1]][e1[0]] = 0
        adj[e2[0]][e2[1]] = 1
        adj[e2[1]][e2[0]] = 1

        new_score = _score_from_adj(adj)

        # Temperature schedule
        progress = steps / total_steps
        T = T_start * math.exp(-10.0 * progress)

        if new_score < best_score:
            best_score = new_score
            best_adj = [row[:] for row in adj]
        elif new_score > best_score:
            # Accept worse with probability exp(-delta/T)
            delta = new_score - best_score
            if rng.random() < math.exp(-delta / max(T, 1e-10)):
                pass  # keep the change
            else:
                # Revert
                adj[e1[0]][e1[1]] = 1
                adj[e1[1]][e1[0]] = 1
                adj[e2[0]][e2[1]] = 0
                adj[e2[1]][e2[0]] = 0

        steps += 1

    # Local polish: try all single edge swaps from best_adj
    if time.time() < _DEADLINE - 5:
        polish_adj = [row[:] for row in best_adj]
        polish_score = best_score
        improved = True
        while improved and time.time() < _DEADLINE - 3:
            improved = False
            present = [(i, j) for i in range(n) for j in range(i + 1, n) if polish_adj[i][j]]
            absent = [(i, j) for i in range(n) for j in range(i + 1, n) if not polish_adj[i][j]]
            for pe in present:
                for ae in absent:
                    if time.time() > _DEADLINE - 3:
                        break
                    polish_adj[pe[0]][pe[1]] = 0
                    polish_adj[pe[1]][pe[0]] = 0
                    polish_adj[ae[0]][ae[1]] = 1
                    polish_adj[ae[1]][ae[0]] = 1
                    s = _score_from_adj(polish_adj)
                    if s < polish_score:
                        polish_score = s
                        best_adj = [row[:] for row in polish_adj]
                        best_score = s
                        improved = True
                    else:
                        polish_adj[pe[0]][pe[1]] = 1
                        polish_adj[pe[1]][pe[0]] = 1
                        polish_adj[ae[0]][ae[1]] = 0
                        polish_adj[ae[1]][ae[0]] = 0
                if time.time() > _DEADLINE - 3:
                    break

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])

    return {'n': n, 'edges': edges}
