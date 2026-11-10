import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11

    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool, dtype=np.int64)

    def greedy_select(pool, max_dot, time_limit, num_restarts=20):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)

        best_set_indices = []

        for r in range(num_restarts):
            if time.time() > time_limit:
                break

            np.random.seed(42 + r)
            order = np.random.permutation(n)

            selected_indices = []
            selected_matrix = np.empty((0, d), dtype=np.int64)

            for idx in order:
                if time.time() > time_limit:
                    break

                v = pool[idx]

                if len(selected_indices) == 0:
                    selected_indices.append(idx)
                    selected_matrix = v.reshape(1, -1)
                    continue

                dots = selected_matrix @ v
                if np.max(dots) <= max_dot:
                    selected_indices.append(idx)
                    selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])

            if len(selected_indices) > len(best_set_indices):
                best_set_indices = selected_indices[:]

        if not best_set_indices:
            return np.empty((0, d), dtype=np.int64)

        return pool[best_set_indices]

    def local_search_improve(pool, max_dot, initial_indices, time_limit, max_iters=500):
        """Try to improve a set by swapping out one point for one not in the set."""
        n = len(pool)
        if len(initial_indices) == 0:
            return initial_indices

        selected = list(initial_indices)
        selected_set = set(selected)
        selected_matrix = pool[selected]

        best_size = len(selected)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1

            # For each selected point, try swapping with an unselected point
            for i, sel_idx in enumerate(selected):
                if time.time() > time_limit:
                    return selected

                # Temporarily remove sel_idx
                remaining = [s for j, s in enumerate(selected) if j != i]
                if not remaining:
                    # Try adding any point
                    for cand in range(n):
                        if cand in selected_set:
                            continue
                        v = pool[cand]
                        if len(remaining) == 0 or np.all(pool[remaining] @ v <= max_dot):
                            selected = [cand]
                            selected_set = {cand}
                            selected_matrix = pool[selected]
                            improved = True
                            break
                    if improved:
                        break
                    continue

                # For each candidate not in selected, check if swap works
                for cand in range(n):
                    if cand in selected_set:
                        continue
                    v = pool[cand]
                    if np.all(pool[remaining] @ v <= max_dot):
                        # Swap: remove sel_idx, add cand
                        selected[i] = cand
                        selected_set.discard(sel_idx)
                        selected_set.add(cand)
                        selected_matrix = pool[selected]
                        improved = True
                        break
                if improved:
                    break

        return selected

    def graph_mis_approach(pool, max_dot, time_limit):
        """Use a simple local search / greedy with swaps on the incompatibility graph."""
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)

        # Precompute the dot product matrix
        # M[i,j] = pool[i] @ pool[j]
        M = pool @ pool.T

        best_set = []
        best_size = 0

        np.random.seed(123)
        order = np.random.permutation(n)

        # Greedy initialization
        selected = []
        selected_set = set()

        for idx in order:
            if time.time() > time_limit:
                break
            if len(selected) == 0:
                selected.append(idx)
                selected_set.add(idx)
                continue
            # Check if idx is compatible with all selected
            if all(M[idx, s] <= max_dot for s in selected):
                selected.append(idx)
                selected_set.add(idx)

        best_size = len(selected)
        best_set = selected[:]

        # Local search improvement
        selected = local_search_improve(pool, max_dot, selected, time_limit, max_iters=200)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

        # A few more restarts with local search
        for r in range(1, 10):
            if time.time() > time_limit:
                break
            np.random.seed(123 + r)
            order = np.random.permutation(n)
            selected = []
            selected_set = set()
            for idx in order:
                if time.time() > time_limit:
                    break
                if len(selected) == 0:
                    selected.append(idx)
                    selected_set.add(idx)
                    continue
                if all(M[idx, s] <= max_dot for s in selected):
                    selected.append(idx)
                    selected_set.add(idx)
            selected = local_search_improve(pool, max_dot, selected, time_limit, max_iters=100)
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]

        if not best_set:
            return np.empty((0, d), dtype=np.int64)
        return pool[best_set]

    best_result = None
    best_k = 0

    # Strategy 1: Greedy on w=4 pool
    if time.time() < deadline:
        pool4 = generate_pool(4)
        res4 = greedy_select(pool4, 2, deadline - 120, num_restarts=30)
        if len(res4) > best_k:
            best_k = len(res4)
            best_result = res4

    # Strategy 2: Graph MIS on w=4 pool
    if time.time() < deadline - 120:
        pool4 = generate_pool(4)
        res4_mis = graph_mis_approach(pool4, 2, deadline - 120)
        if len(res4_mis) > best_k:
            best_k = len(res4_mis)
            best_result = res4_mis

    # Strategy 3: Greedy on w=5 pool
    if time.time() < deadline - 100:
        pool5 = generate_pool(5)
        res5 = greedy_select(pool5, 2, deadline - 100, num_restarts=15)
        if len(res5) > best_k:
            best_k = len(res5)
            best_result = res5

    # Strategy 4: Graph MIS on w=5 pool
    if time.time() < deadline - 80:
        pool5 = generate_pool(5)
        res5_mis = graph_mis_approach(pool5, 2, deadline - 80)
        if len(res5_mis) > best_k:
            best_k = len(res5_mis)
            best_result = res5_mis

    # Strategy 5: Greedy on w=6 pool
    if time.time() < deadline - 40:
        pool6 = generate_pool(6)
        res6 = greedy_select(pool6, 3, deadline - 40, num_restarts=10)
        if len(res6) > best_k:
            best_k = len(res6)
            best_result = res6

    # Fallback
    if best_result is None or best_k < 220:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        best_result = np.array(points, dtype=np.int64)
        best_k = len(best_result)

    return np.around(best_result).astype(np.int64)
