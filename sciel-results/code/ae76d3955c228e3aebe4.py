import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = 40
    target_edges = 401

    def get_score_from_adj(adj):
        """Compute max book size from adjacency matrix (numpy bool array)."""
        # adj[i][j] = True if edge (i,j) exists
        # For each edge, count common neighbors
        max_b = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    # Common neighbors: w where adj[i][w] and adj[j][w]
                    common = np.logical_and(adj[i], adj[j]).sum()
                    if common > max_b:
                        max_b = common
                        if max_b == 0:
                            return 0
        return max_b

    def edges_to_adj(edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def adj_to_edges(adj):
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.add((i, j))
        return edges

    def initial_k2020_plus1():
        """K_{20,20} + one edge within a part."""
        edges = set()
        for i in range(20):
            for j in range(20, 40):
                edges.add((i, j))
        # Add one edge within first part
        edges.add((0, 1))
        return edges

    def initial_random():
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return set(all_pairs[:target_edges])

    def local_search(edges, time_limit):
        """Run local search with edge swaps to minimize max book."""
        adj = edges_to_adj(edges)
        current_score = get_score_from_adj(adj)
        best_edges = set(edges)
        best_score = current_score
        start = time.time()

        # Precompute all non-edges for efficient sampling
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        non_edges = [e for e in all_pairs if e not in edges]

        T = 1.0
        T_min = 0.01
        alpha = 0.9995

        while time.time() < time_limit:
            # Pick a random existing edge to remove and a random non-edge to add
            if len(non_edges) == 0:
                break

            old_u, old_v = random.choice(list(edges))
            new_u, new_v = random.choice(non_edges)

            # Apply swap
            adj[old_u][old_v] = False
            adj[old_v][old_u] = False
            adj[new_u][new_v] = True
            adj[new_v][new_u] = True

            new_score = get_score_from_adj(adj)
            delta = new_score - current_score

            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / max(T, 1e-10))):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = adj_to_edges(adj)
            else:
                # Revert
                adj[old_u][old_v] = True
                adj[old_v][old_u] = True
                adj[new_u][new_v] = False
                adj[new_v][new_u] = False

            T *= alpha
            if T < T_min:
                T = T_min

        return best_edges, best_score

    # Try multiple strategies
    best_overall = None
    best_overall_score = float('inf')

    # Strategy 1: Start from K_{20,20} + 1 edge
    edges1 = initial_k2020_plus1()
    e1, s1 = local_search(edges1, deadline - 5.0)
    if s1 < best_overall_score:
        best_overall_score = s1
        best_overall = e1

    # Strategy 2: Start from random graph
    edges2 = initial_random()
    e2, s2 = local_search(edges2, deadline - 5.0)
    if s2 < best_overall_score:
        best_overall_score = s2
        best_overall = e2

    # If we still have time, do a final polish on the best
    remaining = deadline - time.time()
    if remaining > 10 and best_overall is not None:
        # Do more local search from the best
        e3, s3 = local_search(best_overall, deadline - 2.0)
        if s3 < best_overall_score:
            best_overall_score = s3
            best_overall = e3

    # Ensure we have exactly 401 edges
    if best_overall is None:
        best_overall = initial_k2020_plus1()

    # Format output
    return [[u, v] for u, v in best_overall]
