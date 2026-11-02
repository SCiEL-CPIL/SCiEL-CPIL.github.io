"""Kissing number in dimension 11 - Constructive lattice approach + local search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # === Phase 1: Generate candidate vectors ===
    # Norm-squared 4 vectors: 4 positions, ±1 signs
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Norm-squared 4 vectors: 2 positions, ±2 signs
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = 2 * s
            v2_candidates.append(v)

    # Norm-squared 4 vectors: 1 position, ±2
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v2_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # === Phase 2: Build conflict graph ===
    # Two vectors conflict if their dot product > 2 (distance < 2)
    # distance^2 = ||a||^2 + ||b||^2 - 2*a.b = 8 - 2*a.b for norm-sq 4 vectors
    # distance >= 2 means distance^2 >= 4 means 8 - 2*a.b >= 4 means a.b <= 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    # === Phase 3: Constructive initial solution using lattice structure ===
    # E8 root lattice vectors embedded in first 8 dims (last 3 = 0)
    # A3 root lattice vectors embedded in last 3 dims (first 8 = 0)
    # These are automatically non-conflicting across components (dot = 0)

    def generate_e8_roots():
        """Generate E8 root vectors (240 vectors, norm-sq 2, but we scale to norm-sq 4)."""
        # E8 roots: permutations of (±1, ±1, 0,0,0,0,0,0) and (±1/2,...,±1/2) with even number of minus
        # For our purposes, we need integer vectors with norm-sq 4.
        # The E8 roots scaled by sqrt(2) give norm-sq 4 vectors.
        # But let's just use the (±1,±1,0,...) type which gives norm-sq 2, scaled by sqrt(2) -> not integer.
        # Instead, let's directly construct: vectors in R^8 with norm-sq 4 that form E8-like structure.
        # Actually, let's use a simpler constructive approach.

        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - norm-sq 2, need norm-sq 4
        # So use (±2, 0, 0, 0, 0, 0, 0, 0) or (±1, ±1, ±1, ±1, 0, 0, 0, 0) etc.
        # Let me just generate all norm-sq-4 vectors in 8D and pick a maximal independent set.

        vecs = []
        # 4 non-zero entries of ±1
        for positions in itertools.combinations(range(8), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * 8
                for pos, s in zip(positions, signs):
                    v[pos] = s
                vecs.append(v)
        # 2 non-zero entries of ±2
        for positions in itertools.combinations(range(8), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * 8
                for pos, s in zip(positions, signs):
                    v[pos] = 2 * s
                vecs.append(v)
        # 1 non-zero entry of ±2
        for pos in range(8):
            for sign in [1, -1]:
                v = [0] * 8
                v[pos] = 2 * sign
                vecs.append(v)

        return vecs

    def generate_a3_roots():
        """Generate A3 root vectors in 3D (norm-sq 4)."""
        # A3 roots: (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1) - norm-sq 2
        # Scaled: (±2, ±2, 0), (±2, 0, ±2), (0, ±2, ±2) - norm-sq 8, too big
        # Use (±1, ±1, 0) with norm-sq 2, need norm-sq 4...
        # Actually in our candidate set, norm-sq 4 in 3D means:
        # (±2, 0, 0), (0, ±2, 0), (0, 0, ±2), (±1, ±1, ±1, ...) no that's 4D
        # In 3D: (±2,0,0), (0,±2,0), (0,0,±2) and (±1,±1,±1) has norm-sq 3
        # So just the 6 axis vectors
        vecs = []
        for pos in range(3):
            for sign in [1, -1]:
                v = [0] * 3
                v[pos] = 2 * sign
                vecs.append(v)
        return vecs

    # Constructive approach: greedily build a large independent set
    # using a smart ordering based on lattice structure

    rng = np.random.default_rng(42)

    def greedy_init(seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()

        # Priority: prefer vectors that are "structurally independent"
        # Sort by degree (lower degree = more independent)
        order = np.argsort(deg, kind='stable')

        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search(in_set, max_time=10.0, seed=42):
        """Efficient local search with add, 1-2 swap, 2-2 swap moves."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()

        # Precompute conflict counts
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1

        ls_deadline = time.time() + max_time
        stuck = 0

        while time.time() < ls_deadline:
            improved = False

            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]

            # 1. Add zero-conflict vertices (batch)
            zero_mask = counts == 0
            if np.any(zero_mask):
                zero_idx = np.where(zero_mask)[0]
                n_add = min(10, len(zero_idx))
                chosen = rng_local.choice(zero_idx, size=n_add, replace=False)
                for v_idx in chosen:
                    v = not_in_set[v_idx]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                stuck = 0
                continue

            # 2. 1-2 Swaps
            one_mask = counts == 1
            if np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                max_process = min(len(one_idx), 300)
                process_idx = one_idx[:max_process] if len(one_idx) <= max_process else rng_local.choice(one_idx, size=max_process, replace=False)

                groups = defaultdict(list)
                for idx in process_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        groups[int(nbrs_in_set[0])].append(v)

                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 2:
                        continue
                    if len(vs) <= 30:
                        for i in range(len(vs)):
                            for j in range(i + 1, len(vs)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    in_set[u] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    conflict_count[neighbors[u]] -= 1
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    current_size += 1
                                    improved = True
                                    swap_found = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                    break
                            if swap_found:
                                break
                    else:
                        for _ in range(30):
                            i = rng_local.integers(len(vs))
                            j = rng_local.integers(len(vs))
                            if i == j:
                                continue
                            v1, v2 = vs[i], vs[j]
                            if not conflicts[v1, v2]:
                                in_set[u] = False
                                in_set[v1] = True
                                in_set[v2] = True
                                conflict_count[neighbors[u]] -= 1
                                conflict_count[neighbors[v1]] += 1
                                conflict_count[neighbors[v2]] += 1
                                current_size += 1
                                improved = True
                                swap_found = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                break
                    if swap_found:
                        break
                if swap_found:
                    stuck = 0
                    continue

            # 3. 1-3 Swaps
            if not improved and np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                max_process = min(len(one_idx), 150)
                process_idx = one_idx[:max_process] if len(one_idx) <= max_process else rng_local.choice(one_idx, size=max_process, replace=False)

                groups = defaultdict(list)
                for idx in process_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        groups[int(nbrs_in_set[0])].append(v)

                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 3:
                        continue
                    chosen_vs = []
                    for v in vs:
                        if all(not conflicts[v, cv] for cv in chosen_vs):
                            chosen_vs.append(v)
                            if len(chosen_vs) == 3:
                                break
                    if len(chosen_vs) == 3:
                        in_set[u] = False
                        for v in chosen_vs:
                            in_set[v] = True
                        conflict_count[neighbors[u]] -= 1
                        for v in chosen_vs:
                            conflict_count[neighbors[v]] += 1
                        current_size += 2
                        improved = True
                        swap_found = True
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
                        break
                if swap_found:
                    stuck = 0
                    continue

            # 4. 2-2 Swaps
            if not improved and stuck > 2:
                two_mask = counts == 2
                if np.any(two_mask):
                    two_idx = np.where(two_mask)[0]
                    max_process = min(len(two_idx), 150)
                    process_idx = two_idx[:max_process] if len(two_idx) <= max_process else rng_local.choice(two_idx, size=max_process, replace=False)

                    pairs = defaultdict(list)
                    for idx in process_idx:
                        v = not_in_set[idx]
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 2:
                            key = (int(min(nbrs_in_set)), int(max(nbrs_in_set)))
                            pairs[key].append(v)

                    swap_found = False
                    for (u1, u2), vs in pairs.items():
                        if len(vs) < 2:
                            continue
                        if len(vs) <= 25:
                            for i in range(len(vs)):
                                for j in range(i + 1, len(vs)):
                                    v1, v2 = vs[i], vs[j]
                                    if not conflicts[v1, v2]:
                                        in_set[u1] = False
                                        in_set[u2] = False
                                        in_set[v1] = True
                                        in_set[v2] = True
                                        conflict_count[neighbors[u1]] -= 1
                                        conflict_count[neighbors[u2]] -= 1
                                        conflict_count[neighbors[v1]] += 1
                                        conflict_count[neighbors[v2]] += 1
                                        improved = True
                                        swap_found = True
                                        break
                                if swap_found:
                                    break
                        else:
                            for _ in range(25):
                                i = rng_local.integers(len(vs))
                                j = rng_local.integers(len(vs))
                                if i == j:
                                    continue
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    in_set[u1] = False
                                    in_set[u2] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    conflict_count[neighbors[u1]] -= 1
                                    conflict_count[neighbors[u2]] -= 1
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    improved = True
                                    swap_found = True
                                    break
                        if swap_found:
                            break
                    if swap_found:
                        stuck = 0
                        continue

            if not improved:
                stuck += 1
                if stuck > 10:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = rng_local.integers(1, 6)
                        to_remove = rng_local.choice(in_set_indices, size=min(n_remove, len(in_set_indices)), replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    def simulated_annealing(in_set_init, max_time=8.0, seed=42):
        """SA with add/remove/swap moves for escaping local optima."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9993
        min_T = 0.05

        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            move_type = rng_local.random()

            if move_type < 0.55:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng_local.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng_local.random() < 0.3
                else:
                    accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.75 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng_local.integers(len(in_set_indices))]
                delta = -1
                accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.90 and len(in_set_indices) > 0:
                # SWAP move
                v_remove = in_set_indices[rng_local.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[(count_conflicts[not_in_set] <= 1)]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng_local.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng_local.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
            else:
                # 2-for-1 move
                if len(in_set_indices) >= 2:
                    v_add = not_in_set[rng_local.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    if len(conflicts_with) == 2:
                        to_remove = conflicts_with
                        delta = -1
                        accept = rng_local.random() < np.exp(delta / T)
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            T = max(min_T, T * cooling)

            if stuck_counter > 60:
                T = 5.0
                stuck_counter = 0

        return local_best_in_set, local_best_size

    # === Phase 4: Run search ===
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    # Multiple greedy initializations
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        in_set = greedy_init(seed=seed)
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()

    # Local search on best
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 3:
        ls_time = min(remaining_time * 0.3, 12.0)
        ls_result, ls_size = local_search(best_in_set, max_time=ls_time, seed=42)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # SA on best
    remaining_time = deadline - 8 - time.time()
    if remaining_time > 3:
        sa_time = min(remaining_time * 0.3, 10.0)
        sa_result, sa_size = simulated_annealing(best_in_set, max_time=sa_time, seed=42)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # More restarts with alternating LS and SA
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 40:
        n_restarts += 1
        seed = 100 + n_restarts
        in_set = greedy_init(seed=seed)
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()

        remaining_time = deadline - 5 - time.time()
        if remaining_time < 3:
            break

        if n_restarts % 2 == 0:
            ls_time = min(remaining_time * 0.25, 6.0)
            ls_result, ls_size = local_search(in_set, max_time=ls_time, seed=seed)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()
        else:
            sa_time = min(remaining_time * 0.25, 6.0)
            sa_result, sa_size = simulated_annealing(in_set, max_time=sa_time, seed=seed)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # === Phase 5: Final polish ===
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1

        # Repeatedly add zero-conflict vertices
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            not_in_set = np.where(~best_in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                changed = True

        # One more 1-2 swap pass
        if time.time() < deadline - 1:
            not_in_set = np.where(~best_in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                one_mask = counts == 1
                if np.any(one_mask):
                    one_idx = np.where(one_mask)[0]
                    groups = defaultdict(list)
                    for idx in one_idx[:200]:
                        v = not_in_set[idx]
                        nbrs_in_set = np.where(conflicts[v] & best_in_set)[0]
                        if len(nbrs_in_set) == 1:
                            groups[int(nbrs_in_set[0])].append(v)

                    for u, vs in groups.items():
                        if len(vs) < 2:
                            continue
                        found = False
                        for i in range(min(len(vs), 20)):
                            for j in range(i + 1, min(len(vs), 20)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    best_in_set[u] = False
                                    best_in_set[v1] = True
                                    best_in_set[v2] = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            conflict_count = np.zeros(N, dtype=np.int32)
                            for i in np.where(best_in_set)[0]:
                                conflict_count[neighbors[i]] += 1
                            # Add more zero-conflict
                            not_in_set2 = np.where(~best_in_set)[0]
                            if len(not_in_set2) > 0:
                                counts2 = conflict_count[not_in_set2]
                                zero_conf2 = not_in_set2[counts2 == 0]
                                for v in zero_conf2:
                                    best_in_set[v] = True
                                    conflict_count[neighbors[v]] += 1
                            break

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
