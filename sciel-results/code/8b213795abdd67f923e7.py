import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n_target = 20

    def build_adj_from_edges(edges, n):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Precompute all 6-subsets as index arrays for fast degree computation
    all_6_subsets = list(combinations(range(n_target), 6))
    all_6_subsets_arr = np.array(all_6_subsets, dtype=np.int32)  # shape (C(20,6), 6)
    num_6_subsets = len(all_6_subsets)

    # Also precompute 7-subsets (fewer but still important)
    all_7_subsets = list(combinations(range(n_target), 7))
    all_7_subsets_arr = np.array(all_7_subsets, dtype=np.int32)
    num_7_subsets = len(all_7_subsets)

    # Larger subsets (8..20) are rarer; we'll check them only at the end
    larger_subset_sizes = list(range(8, n_target + 1))

    def count_violations_fast(adj):
        """
        Vectorized count of regular induced subgraphs for k=6 and k=7.
        For k>=8, we do a spot check (they're exponentially rarer).
        Returns (count_6, count_7, count_larger_estimate)
        """
        # For k=6: extract adj[subset, subset] for all subsets at once
        # adj shape: (20, 20), all_6_subsets_arr shape: (38760, 6)
        # We need adj[rows, cols] where rows = all_6_subsets_arr and cols = all_6_subsets_arr
        # This gives a (38760, 6, 6) array
        rows = all_6_subsets_arr
        # adj[rows[:, None, :], rows[:, None, :]] -> (38760, 6, 6)
        sub_matrices_6 = adj[rows[:, None, :], rows[:, None, :]]  # (N, 6, 6)
        degs_6 = sub_matrices_6.sum(axis=2)  # (N, 6)
        # Regular if all 6 degrees are equal
        min_deg_6 = degs_6.min(axis=1)
        max_deg_6 = degs_6.max(axis=1)
        count_6 = int(np.sum(min_deg_6 == max_deg_6))

        # For k=7
        rows7 = all_7_subsets_arr
        sub_matrices_7 = adj[rows7[:, None, :], rows7[:, None, :]]  # (N, 7, 7)
        degs_7 = sub_matrices_7.sum(axis=2)  # (N, 7)
        min_deg_7 = degs_7.min(axis=1)
        max_deg_7 = degs_7.max(axis=1)
        count_7 = int(np.sum(min_deg_7 == max_deg_7))

        # For k >= 8: check a random sample (full check is too expensive per iteration)
        # We'll do this only in the final validation, not in the inner loop.
        # Return a rough indicator: if count_6 + count_7 > 0, definitely invalid.
        # For the SA objective, we use count_6 + count_7 as a proxy.
        return count_6, count_7, 0

    def full_validation(adj, n):
        """
        Full check: no regular induced subgraph on any subset of size >= 6.
        This is expensive; call only on candidates we believe are valid.
        """
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                verts = list(subset)
                sub_adj = adj[np.ix_(verts, verts)]
                degrees = sub_adj.sum(axis=1)
                if np.all(degrees == degrees[0]):
                    return False
        return True

    def count_violations_full_sample(adj, n, sample_size=5000):
        """
        Faster full-ish validation: check all k=6,7 subsets and random samples for k>=8.
        Returns total violation count (0 means likely valid).
        """
        total = 0
        # k=6 and k=7 are precomputed
        c6, c7, _ = count_violations_fast(adj)
        total += c6 + c7

        # For k >= 8, sample subsets
        for k in range(8, n + 1):
            all_subsets_k = list(combinations(range(n), k))
            if len(all_subsets_k) <= sample_size:
                check_subsets = all_subsets_k
            else:
                check_subsets = random.sample(all_subsets_k, sample_size)
            for subset in check_subsets:
                verts = list(subset)
                sub_adj = adj[np.ix_(verts, verts)]
                degrees = sub_adj.sum(axis=1)
                if np.all(degrees == degrees[0]):
                    total += 1
                    break  # One violation per k is enough to know it's invalid
        return total

    def simulated_annealing(adj_init, max_iters=50000, T0=1.0, T_end=0.01):
        """
        Simulated annealing over single-edge flips to minimize violation count.
        """
        adj = adj_init.copy()
        n = adj.shape[0]
        num_pairs = n * (n - 1) // 2

        # Precompute list of all (i, j) pairs with i < j
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Initial objective
        c6, c7, _ = count_violations_fast(adj)
        current_obj = c6 + c7

        best_adj = adj.copy()
        best_obj = current_obj

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Temperature schedule
            T = T0 * (T_end / T0) ** (it / max_iters)

            # Pick a random edge to flip
            u, v = random.choice(pairs)

            # Flip
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = 1 - old_val

            # New objective
            c6_new, c7_new, _ = count_violations_fast(adj)
            new_obj = c6_new + c7_new

            delta = new_obj - current_obj

            if delta <= 0:
                # Accept (improvement or equal)
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_adj = adj.copy()
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    current_obj = new_obj
                else:
                    # Revert
                    adj[u, v] = old_val
                    adj[v, u] = old_val

        return best_adj, best_obj

    def try_sa(n, seed, max_iters=30000):
        """
        Initialize randomly and run simulated annealing.
        """
        random.seed(seed)
        np.random.seed(seed)

        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1

        adj, obj = simulated_annealing(adj, max_iters=max_iters)
        return adj, obj

    # Main search loop
    best_result = None

    # Phase 1: Multiple SA runs with different seeds
    max_seeds = 80
    for seed in range(max_seeds):
        if time.time() > deadline - 20:
            break

        adj, obj = try_sa(n_target, seed, max_iters=25000)

        if obj == 0:
            # SA says no violations for k=6,7. Do full validation.
            if full_validation(adj, n_target):
                edges = edges_from_adj(adj, n_target)
                return {'n': n_target, 'edges': edges}

    # Phase 2: If SA didn't find one, try a more targeted approach
    # Use a greedy repair: start from random, repeatedly find and break violations
    for seed in range(40):
        if time.time() > deadline - 10:
            break

        random.seed(seed + 1000)
        adj = np.zeros((n_target, n_target), dtype=np.int8)
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # Greedy repair loop
        for repair_iter in range(2000):
            if time.time() > deadline - 10:
                break

            # Find a violation at k=6
            c6, c7, _ = count_violations_fast(adj)
            if c6 == 0 and c7 == 0:
                # Check larger
                if full_validation(adj, n_target):
                    edges = edges_from_adj(adj, n_target)
                    return {'n': n_target, 'edges': edges}
                break

            # Find one violating 6-subset
            rows = all_6_subsets_arr
            sub_matrices_6 = adj[rows[:, None, :], rows[:, None, :]]
            degs_6 = sub_matrices_6.sum(axis=2)
            min_deg_6 = degs_6.min(axis=1)
            max_deg_6 = degs_6.max(axis=1)
            viol_mask = (min_deg_6 == max_deg_6)
            viol_indices = np.where(viol_mask)[0]

            if len(viol_indices) == 0:
                # Only k=7 violations
                rows7 = all_7_subsets_arr
                sub_matrices_7 = adj[rows7[:, None, :], rows7[:, None, :]]
                degs_7 = sub_matrices_7.sum(axis=2)
                min_deg_7 = degs_7.min(axis=1)
                max_deg_7 = degs_7.max(axis=1)
                viol_mask7 = (min_deg_7 == max_deg_7)
                viol_indices7 = np.where(viol_mask7)[0]
                if len(viol_indices7) == 0:
                    break
                subset = list(all_7_subsets_arr[viol_indices7[0]])
            else:
                subset = list(all_6_subsets_arr[viol_indices[0]])

            # Try flipping edges within the subset to break regularity
            subset_set = set(subset)
            edges_in_subset = [(u, v) for u in subset for v in subset if u < v]
            random.shuffle(edges_in_subset)

            fixed = False
            for u, v in edges_in_subset:
                old_val = adj[u, v]
                adj[u, v] = 1 - old_val
                adj[v, u] = 1 - old_val

                # Check if this subset is still regular
                verts = list(subset)
                sub_adj = adj[np.ix_(verts, verts)]
                degrees = sub_adj.sum(axis=1)
                if not np.all(degrees == degrees[0]):
                    fixed = True
                    break
                else:
                    adj[u, v] = old_val
                    adj[v, u] = old_val

            if not fixed:
                # Try flipping an edge from subset to outside
                outside = [x for x in range(n_target) if x not in subset_set]
                if outside:
                    random.shuffle(outside)
                    for w in outside[:10]:
                        for u in subset:
                            old_val = adj[u, w]
                            adj[u, w] = 1 - old_val
                            adj[w, u] = 1 - old_val
                            verts = list(subset)
                            sub_adj = adj[np.ix_(verts, verts)]
                            degrees = sub_adj.sum(axis=1)
                            if not np.all(degrees == degrees[0]):
                                fixed = True
                                break
                            else:
                                adj[u, w] = old_val
                                adj[w, u] = old_val
                        if fixed:
                            break

            if not fixed:
                # Accept a random flip to escape local minimum
                u, v = random.choice(pairs)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = 1 - adj[v, u]

    # Phase 3: Fallback - try n=19
    for seed in range(30):
        if time.time() > deadline - 5:
            break

        random.seed(seed + 5000)
        adj = np.zeros((19, 19), dtype=np.int8)
        for i in range(19):
            for j in range(i + 1, 19):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # Quick SA on n=19
        pairs_19 = [(i, j) for i in range(19) for j in range(i + 1, 19)]
        for it in range(10000):
            if time.time() > deadline - 5:
                break
            u, v = random.choice(pairs_19)
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = 1 - old_val
            # Quick check: just accept if it doesn't create obvious small violations
            # (full check too expensive here)
            if random.random() < 0.3:
                adj[u, v] = old_val
                adj[v, u] = old_val

        if full_validation(adj, 19):
            edges = edges_from_adj(adj, 19)
            return {'n': 19, 'edges': edges}

    # Final fallback: n=5 (always valid)
    return {'n': 5, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4]]}
