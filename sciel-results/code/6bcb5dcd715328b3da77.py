import time
import sys
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Start with K_7,8
    # A: 0-6, B: 7-14
    base_edges = set()
    for u in range(7):
        for v in range(7, 15):
            base_edges.add((u, v))

    # Helper to check validity
    def is_valid(edge_list):
        if len(edge_list) == 0:
            return True
        # Build adjacency and degrees
        adj = {i: set() for i in range(n)}
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        # Check all simple paths of length 3
        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # All possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    missing_edges = list(all_edges - base_edges)

    def edge_priority(e, rng):
        u, v = e
        if u < 7 and v < 7:
            return 0
        elif u >= 7 and v >= 7:
            return 1
        else:
            return 2

    best_edges = list(base_edges)
    best_count = len(base_edges)

    def local_improve(edge_set):
        """Try to add any missing edge that maintains validity, repeat until no more can be added."""
        current = list(edge_set)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for e in missing_edges:
                if e in edge_set:
                    continue
                if time.time() > deadline:
                    break
                next_edges = current + [e]
                if is_valid(next_edges):
                    current = next_edges
                    edge_set.add(e)
                    improved = True
        return current

    # Multiple restarts with different orderings
    rng = random.Random(42)
    num_restarts = 20
    restarts_done = 0

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Different ordering strategies
        strategy = restart % 4
        if strategy == 0:
            # Intra-A first, then intra-B, then cross (original)
            missing_sorted = sorted(missing_edges, key=lambda e: edge_priority(e, rng))
        elif strategy == 1:
            # Intra-B first, then intra-A, then cross
            missing_sorted = sorted(missing_edges, key=lambda e: 2 - edge_priority(e, rng))
        elif strategy == 2:
            # Random shuffle
            missing_sorted = missing_edges[:]
            rng.shuffle(missing_sorted)
        else:
            # Mixed: alternate intra and random
            intra = [e for e in missing_edges if (e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7)]
            cross = [e for e in missing_edges if not ((e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7))]
            rng.shuffle(intra)
            rng.shuffle(cross)
            missing_sorted = intra + cross

        # Greedy pass
        current = list(base_edges)
        current_set = set(base_edges)
        for e in missing_sorted:
            if time.time() > deadline:
                break
            if e in current_set:
                continue
            next_edges = current + [e]
            if is_valid(next_edges):
                current = next_edges
                current_set.add(e)

        # Local improvement
        current = local_improve(current_set)

        if len(current) > best_count:
            best_count = len(current)
            best_edges = current
            restarts_done = restart + 1

        # If we've used more than 60% of time, stop restarting
        elapsed = time.time() - start_time
        if elapsed > 0.6 * (deadline - start_time):
            break

    # Final local improvement on best result
    if time.time() < deadline - 2:
        best_set = set(best_edges)
        best_edges = local_improve(best_set)
        best_count = len(best_edges)

    # Convert to required format
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return final_edges
