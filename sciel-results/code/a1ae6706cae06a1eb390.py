import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    n = 16
    target_edges = 65

    # Helper to calculate score (max triangle degree sum) and return worst triangle
    def calc_score_and_worst(adj, degs):
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
                        common ^= lsb
        return max_sum, worst_tri

    # QUBO objective: sum of squared triangle degree sums
    def calc_qubo(adj, degs):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        total += s * s
                        common ^= lsb
        return total

    def generate_random_graph():
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]

        adj = [0] * n
        degs = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs, edges

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_score = float('inf')
    best_adj = None
    best_degs = None
    best_edges = None

    # Precompute all possible edges and non-edges for faster sampling
    all_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for restart in range(200):
        if time.time() > deadline:
            break

        adj, degs, edges = generate_random_graph()
        current_score, _ = calc_score_and_worst(adj, degs)
        current_qubo = calc_qubo(adj, degs)

        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj(adj)

        # Targeted Simulated Annealing: prioritize removing edges from worst triangles
        temp = 5.0
        cooling = 0.997
        no_improve_count = 0
        max_no_improve = 1500

        while time.time() < deadline and no_improve_count < max_no_improve:
            # Get current worst triangle to guide move selection
            _, worst_tri = calc_score_and_worst(adj, degs)

            # Strategy: 70% chance target worst triangle, 30% random
            if worst_tri is not None and random.random() < 0.7:
                # Pick an edge from the worst triangle to remove
                tri_u, tri_v, tri_w = worst_tri
                tri_edges = [(tri_u, tri_v), (tri_u, tri_w), (tri_v, tri_w)]
                # Filter to only existing edges
                valid_tri_edges = [(a, b) for a, b in tri_edges if adj[a] & (1 << b)]
                if valid_tri_edges:
                    u1, v1 = random.choice(valid_tri_edges)
                else:
                    # Fallback: random edge
                    while True:
                        u1 = random.randint(0, n - 1)
                        v1 = random.randint(0, n - 1)
                        if u1 < v1 and (adj[u1] & (1 << v1)):
                            break
            else:
                # Random edge removal
                while True:
                    u1 = random.randint(0, n - 1)
                    v1 = random.randint(0, n - 1)
                    if u1 < v1 and (adj[u1] & (1 << v1)):
                        break

            # For adding: prefer edges that don't create new high-degree triangles
            # Collect candidate non-edges and score them
            candidates = []
            num_candidates = min(20, len([1 for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]))
            for _ in range(num_candidates * 3):  # Sample more to get good candidates
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    # Quick heuristic: prefer edges between low-degree vertices
                    candidates.append((degs[u2] + degs[v2], u2, v2))
                    if len(candidates) >= num_candidates:
                        break

            if candidates:
                # Sort by degree sum (lower is better) and pick from top few
                candidates.sort(key=lambda x: x[0])
                top_k = min(5, len(candidates))
                _, u2, v2 = random.choice(candidates[:top_k])
            else:
                # Fallback: random non-edge
                while True:
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 < v2 and not (adj[u2] & (1 << v2)):
                        break

            # Apply swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            degs[u1] -= 1
            degs[v1] -= 1

            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            degs[u2] += 1
            degs[v2] += 1

            new_score, _ = calc_score_and_worst(adj, degs)
            new_qubo = calc_qubo(adj, degs)

            # Acceptance criterion
            delta_qubo = new_qubo - current_qubo
            if delta_qubo <= 0:
                current_score = new_score
                current_qubo = new_qubo
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj(adj)
            else:
                if random.random() < np.exp(-delta_qubo / (temp + 1e-10)):
                    current_score = new_score
                    current_qubo = new_qubo
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                        best_degs = degs[:]
                        best_edges = get_edges_from_adj(adj)
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    degs[u1] += 1
                    degs[v1] += 1

                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    degs[u2] -= 1
                    degs[v2] -= 1

                    no_improve_count += 1

            temp *= cooling

    # Final polish: targeted local search on best solution
    if best_adj is not None and time.time() < deadline:
        adj = best_adj[:]
        degs = best_degs[:]
        current_score, _ = calc_score_and_worst(adj, degs)

        # Iterative targeted improvement
        for _ in range(500):
            if time.time() > deadline:
                break

            current_score, worst_tri = calc_score_and_worst(adj, degs)
            if worst_tri is None:
                break

            tri_u, tri_v, tri_w = worst_tri
            tri_edges = [(tri_u, tri_v), (tri_u, tri_w), (tri_v, tri_w)]
            valid_tri_edges = [(a, b) for a, b in tri_edges if adj[a] & (1 << b)]

            if not valid_tri_edges:
                break

            best_swap_score = current_score
            best_swap = None

            for u1, v1 in valid_tri_edges:
                # Try adding various non-edges
                for u2 in range(n):
                    if time.time() > deadline:
                        break
                    for v2 in range(u2 + 1, n):
                        if time.time() > deadline:
                            break
                        if (adj[u2] & (1 << v2)):
                            continue

                        # Apply swap
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        degs[u1] -= 1
                        degs[v1] -= 1

                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        degs[u2] += 1
                        degs[v2] += 1

                        new_score, _ = calc_score_and_worst(adj, degs)

                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        degs[u1] += 1
                        degs[v1] += 1

                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        degs[u2] -= 1
                        degs[v2] -= 1

                        if new_score < best_swap_score:
                            best_swap_score = new_score
                            best_swap = (u1, v1, u2, v2)

            if best_swap is None:
                break

            u1, v1, u2, v2 = best_swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            degs[u1] -= 1
            degs[v1] -= 1

            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            degs[u2] += 1
            degs[v2] += 1

            current_score = best_swap_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                best_degs = degs[:]
                best_edges = get_edges_from_adj(adj)

    if best_edges is None:
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}

    return {'n': 16, 'edges': best_edges}
