import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    n = 16
    target_edges = 65

    # Precompute all possible edges as normalized tuples
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def build_adj(edges):
        adj = [0] * n
        deg = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def compute_score(adj, deg):
        """Compute max degree sum over all triangles using bitsets."""
        max_sum = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    common = ai & adj[j]
                    # Only consider k > j
                    common &= ~((1 << (j + 1)) - 1)
                    while common:
                        lb = common & (-common)
                        k = lb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                        common ^= lb
        return max_sum

    def normalize_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    def find_worst_triangle(adj, deg):
        """Find the triangle with maximum degree sum."""
        worst = None
        worst_sum = -1
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    common = ai & adj[j]
                    common &= ~((1 << (j + 1)) - 1)
                    while common:
                        lb = common & (-common)
                        k = lb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > worst_sum:
                            worst_sum = s
                            worst = (i, j, k)
                        common ^= lb
        return worst, worst_sum

    def local_search(edges, time_limit):
        """Hill-climbing with edge swaps to minimize max triangle degree sum."""
        edge_set = set(normalize_edge(u, v) for (u, v) in edges)
        adj, deg = build_adj(edge_set)
        current_score = compute_score(adj, deg)
        best_score = current_score
        best_edges = set(edge_set)

        end_time = start_time + time_limit

        while time.time() < min(end_time, deadline):
            worst_tri, worst_sum = find_worst_triangle(adj, deg)
            if worst_tri is None:
                break
            i, j, k = worst_tri

            # Try to improve by swapping edges
            # First, try removing one edge from the worst triangle and adding a different edge
            tri_edges = [(i, j), (j, k), (i, k)]
            tri_edges = [normalize_edge(a, b) for a, b in tri_edges]
            random.shuffle(tri_edges)

            improved = False
            for rm_edge in tri_edges:
                rm_u, rm_v = rm_edge
                # Try adding a random new edge
                for attempt in range(10):
                    new_u = random.randint(0, n - 1)
                    new_v = random.randint(0, n - 1)
                    if new_u == new_v:
                        continue
                    new_edge = normalize_edge(new_u, new_v)
                    if new_edge in edge_set:
                        continue
                    if new_edge == rm_edge:
                        continue

                    # Apply swap
                    edge_set.discard(rm_edge)
                    edge_set.add(new_edge)

                    # Update degrees and adjacency
                    adj[rm_u] &= ~(1 << rm_v)
                    adj[rm_v] &= ~(1 << rm_u)
                    deg[rm_u] -= 1
                    deg[rm_v] -= 1

                    adj[new_u] |= (1 << new_v)
                    adj[new_v] |= (1 << new_u)
                    deg[new_u] += 1
                    deg[new_v] += 1

                    new_score = compute_score(adj, deg)

                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = set(edge_set)
                        improved = True
                        break
                    else:
                        # Revert
                        edge_set.discard(new_edge)
                        edge_set.add(rm_edge)

                        adj[rm_u] |= (1 << rm_v)
                        adj[rm_v] |= (1 << rm_u)
                        deg[rm_u] += 1
                        deg[rm_v] += 1

                        adj[new_u] &= ~(1 << new_v)
                        adj[new_v] &= ~(1 << new_u)
                        deg[new_u] -= 1
                        deg[new_v] -= 1

                if improved:
                    break

            if not improved:
                # Try a random perturbation: swap two random edges
                for _ in range(5):
                    rm_edge = random.choice(list(edge_set))
                    rm_u, rm_v = rm_edge
                    new_u = random.randint(0, n - 1)
                    new_v = random.randint(0, n - 1)
                    if new_u == new_v:
                        continue
                    new_edge = normalize_edge(new_u, new_v)
                    if new_edge in edge_set:
                        continue
                    if new_edge == rm_edge:
                        continue

                    edge_set.discard(rm_edge)
                    edge_set.add(new_edge)

                    adj[rm_u] &= ~(1 << rm_v)
                    adj[rm_v] &= ~(1 << rm_u)
                    deg[rm_u] -= 1
                    deg[rm_v] -= 1

                    adj[new_u] |= (1 << new_v)
                    adj[new_v] |= (1 << new_u)
                    deg[new_u] += 1
                    deg[new_v] += 1

                    new_score = compute_score(adj, deg)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = set(edge_set)
                        break
                    else:
                        # Revert
                        edge_set.discard(new_edge)
                        edge_set.add(rm_edge)

                        adj[rm_u] |= (1 << rm_v)
                        adj[rm_v] |= (1 << rm_u)
                        deg[rm_u] += 1
                        deg[rm_v] += 1

                        adj[new_u] &= ~(1 << new_v)
                        adj[new_v] &= ~(1 << new_u)
                        deg[new_u] -= 1
                        deg[new_v] -= 1

        return list(best_edges), best_score

    def random_turan_start():
        """Generate a random near-Turán graph."""
        perm = list(range(n))
        random.shuffle(perm)
        a = 8
        A = perm[:a]
        B = perm[a:]
        # Extra edge within one part
        if random.random() < 0.5:
            u, v = random.sample(A, 2)
        else:
            u, v = random.sample(B, 2)
        edges = []
        for u in A:
            for v in B:
                edges.append((u, v))
        edges.append((u, v))
        return edges

    # Multi-restart local search
    best_overall_score = 999
    best_overall_edges = None

    # Try canonical construction first
    canonical_edges = []
    for u in range(8):
        for v in range(8, 16):
            canonical_edges.append((u, v))
    canonical_edges.append((0, 1))
    adj, deg = build_adj(canonical_edges)
    s = compute_score(adj, deg)
    best_overall_score = s
    best_overall_edges = list(canonical_edges)

    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        edges = random_turan_start()
        found_edges, found_score = local_search(edges, time_limit=8.0)
        if found_score < best_overall_score:
            best_overall_score = found_score
            best_overall_edges = found_edges

        # Also try perturbing the best found
        if best_overall_edges and random.random() < 0.5:
            perturbed = set(best_overall_edges)
            if perturbed:
                rm = random.choice(list(perturbed))
                perturbed.discard(rm)
                new_u = random.randint(0, n - 1)
                new_v = random.randint(0, n - 1)
                if new_u != new_v:
                    new_e = normalize_edge(new_u, new_v)
                    if new_e not in perturbed:
                        perturbed.add(new_e)
                found_edges, found_score = local_search(list(perturbed), time_limit=5.0)
                if found_score < best_overall_score:
                    best_overall_score = found_score
                    best_overall_edges = found_edges

    # Final polish
    if best_overall_edges:
        found_edges, found_score = local_search(best_overall_edges, time_limit=10.0)
        if found_score < best_overall_score:
            best_overall_score = found_score
            best_overall_edges = found_edges

    edges_list = [[u, v] for (u, v) in best_overall_edges]
    return {'n': n, 'edges': edges_list}
