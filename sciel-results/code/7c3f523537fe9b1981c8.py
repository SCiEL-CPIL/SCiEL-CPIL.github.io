import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def get_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_get_score(values, edge_indices):
        """edge_indices: list of (u,v) tuples, length m. Returns (score, sums_set, prods_set)."""
        sums = set()
        prods = set()
        for u, v in edge_indices:
            su = values[u]
            sv = values[v]
            sums.add(su + sv)
            prods.add(su * sv)
        return max(len(sums), len(prods)), sums, prods

    def joint_sa(initial_values, initial_edges, max_iter=20000, T_start=2.0, T_end=0.01):
        current_values = list(initial_values)
        current_edges = set(initial_edges)
        current_edge_list = list(current_edges)

        current_score, current_sums, current_prods = fast_get_score(current_values, current_edge_list)

        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)

        # Precompute: for each vertex, which other vertices form an edge with it
        # This helps for collision-seeking moves
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)

        for i in range(max_iter):
            if time.time() > deadline:
                break

            T = T_start * (T_end / T_start) ** (i / max_iter)
            move_type = random.random()

            if move_type < 0.45:
                # Edge Move: Swap one edge for another
                e_out_idx = random.randint(0, m - 1)
                e_out = current_edge_list[e_out_idx]

                # Pick a non-edge to add
                while True:
                    e_in = random.choice(all_edges)
                    if e_in not in current_edges:
                        break

                # Update edge list
                new_edge_list = list(current_edge_list)
                new_edge_list[e_out_idx] = e_in
                new_edges = set(new_edge_list)

                # Recompute score
                new_score, new_sums, new_prods = fast_get_score(current_values, new_edge_list)

                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                    current_edges = new_edges
                    current_edge_list = new_edge_list
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    # Update adjacency
                    adj[e_out[0]].discard(e_out[1])
                    adj[e_out[1]].discard(e_out[0])
                    adj[e_in[0]].add(e_in[1])
                    adj[e_in[1]].add(e_in[0])

                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)

            elif move_type < 0.85:
                # Value Move: Swap two values
                idx = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx == idx2:
                    continue

                new_values = list(current_values)
                new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]

                new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                    current_values = new_values
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods

                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)

            else:
                # Value Move: Targeted collision-seeking
                # Pick a random vertex, try to set its value so that one of its edges
                # produces a sum or product already in the sets
                idx = random.randint(0, n - 1)
                neighbors = list(adj[idx])
                if not neighbors:
                    continue

                # Try a few neighbor-based adjustments
                found = False
                for _ in range(3):
                    nb = random.choice(neighbors)
                    v_nb = current_values[nb]

                    # Try to match an existing sum: values[idx] + v_nb = s in current_sums
                    candidates = []
                    for s in current_sums:
                        cand = s - v_nb
                        if 1 <= cand <= 10000 and cand != current_values[idx]:
                            # Check distinctness
                            if cand not in current_values:
                                candidates.append(cand)
                    # Also try products
                    for p in current_prods:
                        if p % v_nb == 0:
                            cand = p // v_nb
                            if 1 <= cand <= 10000 and cand != current_values[idx]:
                                if cand not in current_values:
                                    candidates.append(cand)

                    if candidates:
                        new_val = random.choice(candidates[:10])  # Limit to first 10 for speed
                        new_values = list(current_values)
                        new_values[idx] = new_val

                        new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                            current_values = new_values
                            current_score = new_score
                            current_sums = new_sums
                            current_prods = new_prods

                            if current_score < best_score:
                                best_score = current_score
                                best_values = list(current_values)
                                best_edges = set(current_edges)
                            found = True
                            break

                if not found:
                    # Fallback: small random walk
                    delta = random.randint(-10, 10)
                    new_val = current_values[idx] + delta
                    if 1 <= new_val <= 10000 and new_val not in current_values:
                        new_values = list(current_values)
                        new_values[idx] = new_val
                        new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                            current_values = new_values
                            current_score = new_score
                            current_sums = new_sums
                            current_prods = new_prods

                            if current_score < best_score:
                                best_score = current_score
                                best_values = list(current_values)
                                best_edges = set(current_edges)

        return best_score, best_values, best_edges

    def polish_edges(values, edges, max_iters=5000):
        """Local search on edges only, keeping values fixed."""
        current_edges = set(edges)
        current_edge_list = list(current_edges)
        current_score, _, _ = fast_get_score(values, current_edge_list)

        best_score = current_score
        best_edges = set(current_edges)

        for i in range(max_iters):
            if time.time() > deadline:
                break

            e_out_idx = random.randint(0, m - 1)
            e_out = current_edge_list[e_out_idx]

            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges:
                    break

            new_edge_list = list(current_edge_list)
            new_edge_list[e_out_idx] = e_in

            new_score, _, _ = fast_get_score(values, new_edge_list)

            if new_score <= current_score:
                current_edges = set(new_edge_list)
                current_edge_list = new_edge_list
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)

        return best_score, best_edges

    def polish_values(values, edges, max_iters=5000):
        """Local search on values only, keeping edges fixed."""
        current_values = list(values)
        edge_list = list(edges)
        current_score, _, _ = fast_get_score(current_values, edge_list)

        best_score = current_score
        best_values = list(current_values)

        for i in range(max_iters):
            if time.time() > deadline:
                break

            idx = random.randint(0, n - 1)
            # Try swapping
            if random.random() < 0.6:
                idx2 = random.randint(0, n - 1)
                if idx2 == idx:
                    continue
                new_values = list(current_values)
                new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
            else:
                delta = random.randint(-20, 20)
                new_val = current_values[idx] + delta
                if not (1 <= new_val <= 10000) or new_val in current_values:
                    continue
                new_values = list(current_values)
                new_values[idx] = new_val

            new_score, _, _ = fast_get_score(new_values, edge_list)

            if new_score <= current_score:
                current_values = new_values
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)

        return best_score, best_values

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None

    random.seed(42)

    # Generate diverse initial configurations
    initial_configs = []

    # Config 1: Small range
    v1 = random.sample(range(1, 51), n)
    e1 = set(random.sample(all_edges, m))
    initial_configs.append((v1, e1))

    # Config 2: Medium range
    v2 = random.sample(range(1, 101), n)
    e2 = set(random.sample(all_edges, m))
    initial_configs.append((v2, e2))

    # Config 3: Arithmetic progression
    v3 = [i * 10 + 1 for i in range(n)]
    e3 = set(random.sample(all_edges, m))
    initial_configs.append((v3, e3))

    # Config 4: Powers of 2
    v4 = [2**i for i in range(n)]
    if max(v4) <= 10000:
        e4 = set(random.sample(all_edges, m))
        initial_configs.append((v4, e4))

    # Config 5: Random large
    v5 = random.sample(range(1, 10001), n)
    e5 = set(random.sample(all_edges, m))
    initial_configs.append((v5, e5))

    # Config 6: Near-duplicates (many small values to encourage collisions)
    v6 = list(range(1, 8)) + [10, 12, 15, 20, 25, 30, 40]
    random.shuffle(v6)
    e6 = set(random.sample(all_edges, m))
    initial_configs.append((v6, e6))

    # Config 7: Multiplicative structure
    v7 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    random.shuffle(v7)
    e7 = set(random.sample(all_edges, m))
    initial_configs.append((v7, e7))

    # Run SA for each config
    time_per_config = (deadline - time.time()) / len(initial_configs)

    for vals, edges in initial_configs:
        if time.time() > deadline:
            break

        max_iter = int(time_per_config * 800)
        max_iter = min(max_iter, 30000)
        max_iter = max(max_iter, 2000)

        score, best_v, best_e = joint_sa(vals, edges, max_iter=max_iter, T_start=2.0, T_end=0.01)

        # Polish pass
        if time.time() < deadline - 5.0:
            polish_iters = int((deadline - time.time()) * 0.3 * 500)
            polish_iters = max(polish_iters, 1000)
            p_score, p_edges = polish_edges(best_v, best_e, max_iters=polish_iters)
            if p_score < score:
                score = p_score
                best_e = p_edges

            if time.time() < deadline - 2.0:
                pv_iters = int((deadline - time.time()) * 0.2 * 500)
                pv_iters = max(pv_iters, 500)
                pv_score, p_values = polish_values(best_v, best_e, max_iters=pv_iters)
                if pv_score < score:
                    score = pv_score
                    best_v = p_values

        if score < best_global_score:
            best_global_score = score
            best_global_values = best_v
            best_global_edges = best_e

    # Final polish on best result
    if best_global_values is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        pe_iters = int(remaining * 0.5 * 500)
        pe_iters = max(pe_iters, 500)
        ps, pe = polish_edges(best_global_values, best_global_edges, max_iters=pe_iters)
        if ps < best_global_score:
            best_global_score = ps
            best_global_edges = pe

        if time.time() < deadline - 0.5:
            pv_iters = int((deadline - time.time() - 0.5) * 500)
            pv_iters = max(pv_iters, 200)
            pvs, pv = polish_values(best_global_values, best_global_edges, max_iters=pv_iters)
            if pvs < best_global_score:
                best_global_score = pvs
                best_global_values = pv

    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)

    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]

    return {
        'values': final_values,
        'edges': final_edges
    }
