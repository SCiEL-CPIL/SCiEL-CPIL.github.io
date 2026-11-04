import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330

    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]

    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)

    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1

    # Build neighbor lists for faster access
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    incompat_bool = incompat.astype(bool)

    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic."""
        selected = set()
        avail = set(available)

        while avail:
            if time.time() > deadline - 15.0:
                break

            if len(avail) > 200:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                min_mask = deg_arr == min_deg
                candidates = avail_arr[min_mask]
                u = int(rng.choice(candidates))
            else:
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)

            selected.add(u)
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove

        return selected

    def local_search_add(selected, rng):
        """Try to add vertices that are compatible with all current selected vertices."""
        selected = set(selected)
        if not selected:
            return selected

        sel_arr = np.array(list(selected), dtype=np.int32)
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        compatible_mask[sel_arr] = False

        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))

        return selected

    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True

        while improved and time.time() < deadline - 10.0:
            improved = False

            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break

                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)

                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break

                if improved:
                    break

        return selected

    def local_search_2_1_swap(selected, rng):
        """Improve MIS using 2-1 swaps: remove two vertices, add one compatible vertex."""
        selected = set(selected)
        improved = True

        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            n_sel = len(sel_list)

            for i in range(n_sel):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n_sel):
                    v1 = sel_list[i]
                    v2 = sel_list[j]
                    remaining = selected - {v1, v2}

                    for x in range(N_sup):
                        if x in selected:
                            continue
                        if not (neighbors[x] & remaining):
                            selected.remove(v1)
                            selected.remove(v2)
                            selected.add(x)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return selected

    best_iset = set()
    best_size = 0

    rng = np.random.default_rng(42)

    n_restarts = 120
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break

        if r == 0:
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 40:
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 80:
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 30)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 60)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)

        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)

    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)

    if time.time() < deadline - 5.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)

    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    if not vectors:
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)

    return np.array(vectors, dtype=np.int8)
