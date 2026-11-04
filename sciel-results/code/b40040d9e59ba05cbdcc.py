import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Build a 3-(11,4,1) packing: no pair of points appears in more than 1 block
    # Greedy with random restarts to find a good packing
    rng = np.random.default_rng(12345)

    def greedy_packing(rng, n_trials=200):
        """Greedy 3-(11,4,1) packing: no pair appears in >1 block."""
        best = []
        best_size = 0
        for trial in range(n_trials):
            if time.time() > deadline - 30.0:
                break
            pair_count = np.zeros((D, D), dtype=np.int8)
            blocks = []
            remaining = set(combinations(range(D), 4))
            # Randomize order
            rem_list = list(remaining)
            rng.shuffle(rem_list)
            for block in rem_list:
                if time.time() > deadline - 30.0:
                    break
                # Check if any pair in this block already has count >= 1
                ok = True
                for a, b in combinations(block, 2):
                    if pair_count[a, b] >= 1:
                        ok = False
                        break
                if ok:
                    blocks.append(block)
                    for a, b in combinations(block, 2):
                        pair_count[a, b] += 1
                        pair_count[b, a] += 1
            if len(blocks) > best_size:
                best_size = len(blocks)
                best = list(blocks)
        return best

    # Get the packing
    packing = greedy_packing(rng)

    # Also try a deterministic construction for reliability
    # Known 3-(11,4,1) packings have at most 32 blocks
    # Use a simple greedy without randomization as fallback
    if len(packing) < 30:
        pair_count = np.zeros((D, D), dtype=np.int8)
        blocks = []
        for block in combinations(range(D), 4):
            ok = True
            for a, b in combinations(block, 2):
                if pair_count[a, b] >= 1:
                    ok = False
                    break
            if ok:
                blocks.append(block)
                for a, b in combinations(block, 2):
                    pair_count[a, b] += 1
                    pair_count[b, a] += 1
        if len(blocks) > len(packing):
            packing = blocks

    # Now try to find additional weight-4 supports that don't conflict with the packing
    # Conflict: |intersection| >= 3 between any two selected supports
    # Also need: no pair appears in >1 block (for the sign vectors to have adequate distance)
    # Actually, the key constraint is |intersection| <= 2 for all pairs of supports

    # Let's build the full conflict graph and find a maximum independent set
    # among all 330 supports, but seed with our packing

    all_supports = list(combinations(range(D), 4))
    N_sup = len(all_supports)

    # Build incompatibility: two supports conflict if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    for i in range(N_sup):
        if time.time() > deadline - 40.0:
            break
        for j in range(i + 1, N_sup):
            inter = len(set(all_supports[i]) & set(all_supports[j]))
            if inter >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1

    incompat_bool = incompat.astype(bool)
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]

    # Find MIS using greedy + local search, seeded with packing
    def greedy_mis(available, rng, seed=None):
        selected = set(seed) if seed else set()
        avail = set(available) - selected
        # Remove neighbors of selected
        for s in selected:
            avail -= neighbors[s]

        while avail:
            if time.time() > deadline - 15.0:
                break
            # Minimum degree heuristic
            if len(avail) > 100:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                candidates = avail_arr[deg_arr == min_deg]
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
                if not candidates:
                    break
                u = rng.choice(candidates)

            selected.add(u)
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove

        return selected

    def local_search_add(selected):
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
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for v in list(selected):
                if time.time() > deadline - 15.0:
                    break
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
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
                            break
                    if improved:
                        break
                if improved:
                    break
        return selected

    # Seed with packing indices
    packing_set = set()
    for block in packing:
        for idx, sup in enumerate(all_supports):
            if sup == block:
                packing_set.add(idx)
                break

    # Try multiple strategies
    best_iset = set()
    best_size = 0

    # Strategy 1: Greedy from packing seed
    iset = greedy_mis(set(range(N_sup)), rng, seed=packing_set)
    iset = local_search_add(iset)
    iset = local_search_1_2_swap(iset, rng)
    if len(iset) > best_size:
        best_size = len(iset)
        best_iset = set(iset)

    # Strategy 2: Multiple random restarts
    for r in range(30):
        if time.time() > deadline - 20.0:
            break
        avail = set(range(N_sup))
        n_remove = rng.integers(10, 50)
        to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
        avail -= to_remove
        iset = greedy_mis(avail, rng)
        iset = local_search_add(iset)
        iset = local_search_1_2_swap(iset, rng)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = set(iset)

    # Final polish
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset, rng)

    # Generate vectors from MIS: all 16 sign patterns per support
    vectors = []
    for i in best_iset:
        sup = all_supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    # Add Type A vectors: ±2 * e_i
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sign
            vectors.append(v)

    # Validate: check that min pairwise distance >= max norm
    if len(vectors) > 1:
        vecs = np.array(vectors, dtype=np.float64)
        norms = np.sqrt((vecs ** 2).sum(axis=1))
        max_norm = norms.max()

        # Check pairwise distances (sample if too large)
        n = len(vecs)
        valid = True
        if n <= 500:
            dists = np.sqrt(((vecs[:, None, :] - vecs[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()
            if min_dist < max_norm - 1e-9:
                valid = False
        else:
            # Sample check
            for _ in range(100):
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                d = np.sqrt(np.sum((vecs[i] - vecs[j]) ** 2))
                if d < max_norm - 1e-9:
                    valid = False
                    break

        if not valid:
            # Fall back to just the packing-based vectors (which should be valid)
            vectors = []
            for i in best_iset:
                sup = all_supports[i]
                for signs in product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int8)
                    for idx, s in zip(sup, signs):
                        v[idx] = s
                    vectors.append(v)
            # Re-add Type A and validate
            for i in range(D):
                for sign in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = 2 * sign
                    vectors.append(v)

    if not vectors:
        # Last resort: Type A only
        vectors = []
        for i in range(D):
            for sign in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                v[i] = 2 * sign
                vectors.append(v)

    return np.array(vectors, dtype=np.int8)
