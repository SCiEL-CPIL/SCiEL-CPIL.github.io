"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # --- W4 vectors (weight 4, entries in {-1,0,1}, norm^2 = 4) ---
    # --- W1 vectors (weight 1, entries in {-2,0,2}, norm^2 = 4) ---
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # --- Reduce to support graph ---
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        support_to_indices.setdefault(support, []).append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)

    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    sup_sets = [set(s) for s in supports]
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True

    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)

    def greedy_min_degree_supports():
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def spectral_init_supports():
        """Use eigenvector of conflict graph adjacency to seed an independent set."""
        A = support_conflicts.astype(np.float64)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            # Largest eigenvalue corresponds to last column
            principal = eigenvectors[:, -1]
            # Select vertices with largest absolute components
            order = np.argsort(-np.abs(principal))
            in_set = np.zeros(n_supports, dtype=bool)
            for s in order:
                if not np.any(support_conflicts[s] & in_set):
                    in_set[s] = True
            return in_set
        except Exception:
            return greedy_min_degree_supports()

    def sa_local_search_supports(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 4.0
        cooling = 0.9995
        min_T = 0.01

        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            # Choose move type: 60% add, 25% swap, 15% 2-for-2
            move_type = rng.random()
            
            if move_type < 0.60:
                # Add move: pick node not in set
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                # Weight by low conflict count
                weights = 1.0 / (1.0 + count_conflicts[not_in_set].astype(np.float64))
                weights /= weights.sum()
                s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                k = count_conflicts[s]
                delta = 1 - k
                if delta > 0 or rng.random() < np.exp(delta / T):
                    to_remove = np.where(support_conflicts[s] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors_sup[s]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85:
                # Swap move: remove one in-set node, add one not-in-set node
                in_set_nodes = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_nodes) == 0 or len(not_in_set) == 0:
                    continue
                s = not_in_set[rng.integers(len(not_in_set))]
                k = count_conflicts[s]
                if k == 0:
                    continue
                # Pick a neighbor of s to remove
                nbrs_in_set = np.where(support_conflicts[s] & in_set)[0]
                if len(nbrs_in_set) == 0:
                    continue
                u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                # Net change: remove u, add s
                # But s may conflict with other in-set nodes too
                other_conflicts = np.where(support_conflicts[s] & in_set)[0]
                other_conflicts = other_conflicts[other_conflicts != u]
                delta = 1 - len(other_conflicts) - 1  # add s, remove u, but other conflicts remain
                # Actually: after removing u, s's conflicts with remaining in_set nodes
                # The delta is: +1 (add s) - len(other_conflicts) (they get removed? No, they stay)
                # We only remove u. So delta = 1 - 1 = 0 if no other conflicts, or worse.
                # Let's just do: remove u, add s. Delta = 1 - 1 = 0 always for this move type.
                # This is a neutral move to explore.
                accept = rng.random() < 0.5 if delta == 0 else (rng.random() < np.exp(delta / T))
                if accept:
                    in_set[u] = False
                    count_conflicts[neighbors_sup[u]] -= 1
                    # Now add s, but it may conflict with remaining in_set nodes
                    # We need to remove all conflicts of s
                    to_remove = np.where(support_conflicts[s] & in_set)[0]
                    for v in to_remove:
                        in_set[v] = False
                        count_conflicts[neighbors_sup[v]] -= 1
                    in_set[s] = True
                    count_conflicts[neighbors_sup[s]] += 1
                    # Recalculate size
                    new_size = int(in_set.sum())
                    current_size = new_size
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            else:
                # 2-for-2 move: pick two not-in nodes, remove their union of in-set conflicts
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) < 2:
                    continue
                i1 = rng.integers(len(not_in_set))
                i2 = rng.integers(len(not_in_set))
                if i1 == i2:
                    continue
                s1 = not_in_set[i1]
                s2 = not_in_set[i2]
                if support_conflicts[s1, s2]:
                    continue
                c1 = np.where(support_conflicts[s1] & in_set)[0]
                c2 = np.where(support_conflicts[s2] & in_set)[0]
                union = np.union1d(c1, c2)
                delta = 2 - len(union)
                if delta > 0 or rng.random() < np.exp(delta / T):
                    for u in union:
                        in_set[u] = False
                        count_conflicts[neighbors_sup[u]] -= 1
                    in_set[s1] = True
                    in_set[s2] = True
                    count_conflicts[neighbors_sup[s1]] += 1
                    count_conflicts[neighbors_sup[s2]] += 1
                    current_size = int(in_set.sum())
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 2000:
                T = max(T, 5.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_local_polish(in_set_init, max_time=15.0):
        """Deterministic hill-climb: add free nodes, best single swap,
        best 2-for-2 swap using bitmask for speed; stop at a local optimum."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1

        # Precompute bitmask representation of conflict sets for speed
        # n_supports <= 330, so we need ~6 uint64 words
        n_words = (n_supports + 63) // 64
        conflict_masks = np.zeros((n_supports, n_words), dtype=np.uint64)
        for i in range(n_supports):
            nbrs = neighbors_sup[i]
            for w in range(n_words):
                start = w * 64
                end = min(start + 64, n_supports)
                mask = 0
                for idx in nbrs:
                    if start <= idx < end:
                        mask |= (1 << (idx - start))
                conflict_masks[i, w] = mask

        def mask_to_indices(mask_row):
            """Convert bitmask row to list of indices."""
            result = []
            for w in range(n_words):
                m = mask_row[w]
                if m == 0:
                    continue
                bit = 0
                while m:
                    if m & 1:
                        result.append(w * 64 + bit)
                    m >>= 1
                    bit += 1
            return np.array(result, dtype=np.int64)

        def union_mask_count(m1, m2):
            """Count bits in union of two mask rows."""
            total = 0
            for w in range(n_words):
                u = m1[w] | m2[w]
                total += bin(u).count('1')
            return total

        polish_deadline = time.time() + max_time
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Pass A: add any free node (delta = +1, no removals)
            free = np.where((~in_set) & (count_conflicts == 0))[0]
            for s in free:
                in_set[s] = True
                current_size += 1
                count_conflicts[neighbors_sup[s]] += 1
            if len(free) > 0:
                improved = True

            # Pass B: best single swap (remove neighbors of s, add s)
            best_delta = 0
            best_s = -1
            for s in np.where(~in_set)[0]:
                k = count_conflicts[s]
                if k == 0:
                    continue
                d = 1 - k
                if d > best_delta:
                    best_delta = d
                    best_s = s
            if best_s >= 0:
                to_remove = np.where(support_conflicts[best_s] & in_set)[0]
                for u in to_remove:
                    in_set[u] = False
                    count_conflicts[neighbors_sup[u]] -= 1
                in_set[best_s] = True
                current_size += best_delta
                count_conflicts[neighbors_sup[best_s]] += 1
                improved = True

            # Pass C: best 2-for-2 swap using bitmask for speed
            not_in = np.where(~in_set)[0]
            if len(not_in) >= 2:
                sorted_not_in = not_in[np.argsort(count_conflicts[not_in])]
                max_pairs_to_check = min(len(sorted_not_in), 100)
                best_pair_delta = 0
                best_pair = None
                for idx1 in range(max_pairs_to_check):
                    s1 = sorted_not_in[idx1]
                    m1 = conflict_masks[s1]
                    for idx2 in range(idx1 + 1, max_pairs_to_check):
                        s2 = sorted_not_in[idx2]
                        # Check if s1 and s2 conflict with each other
                        if support_conflicts[s1, s2]:
                            continue
                        m2 = conflict_masks[s2]
                        # Count union of conflicts with in_set nodes
                        # We only care about conflicts with nodes currently in_set
                        # Build in_set mask
                        # Actually, let's just compute union count directly
                        u_count = 0
                        for w in range(n_words):
                            u = (m1[w] | m2[w])
                            u_count += bin(u).count('1')
                        delta = 2 - u_count
                        if delta > best_pair_delta:
                            best_pair_delta = delta
                            best_pair = (s1, s2)
                if best_pair is not None and best_pair_delta > 0:
                    s1, s2 = best_pair
                    # Compute union of conflicts
                    union_set = set()
                    for w in range(n_words):
                        u = conflict_masks[s1][w] | conflict_masks[s2][w]
                        bit = 0
                        while u:
                            if u & 1:
                                idx = w * 64 + bit
                                if idx < n_supports:
                                    union_set.add(idx)
                            u >>= 1
                            bit += 1
                    rem = np.array([x for x in union_set if in_set[x]], dtype=np.int64)
                    for u in rem:
                        in_set[u] = False
                        count_conflicts[neighbors_sup[u]] -= 1
                    in_set[s1] = True
                    in_set[s2] = True
                    current_size += best_pair_delta
                    count_conflicts[neighbors_sup[s1]] += 1
                    count_conflicts[neighbors_sup[s2]] += 1
                    improved = True

        return in_set, current_size

    # --- Multiple restarts on the (much smaller) support graph ---
    n_restarts = 0
    max_restarts = 400

    # First restart uses spectral initialization
    if time.time() < deadline - 8:
        in_set = spectral_init_supports()
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 3:
            sa_time = min(remaining_time * 0.6, 12.0)
            if sa_time < 2:
                sa_time = 2.0
            sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_support_set = sa_result.copy()
            # Polish right after SA
            remaining_time = deadline - 5 - time.time()
            if remaining_time > 3:
                polish_time = min(remaining_time * 0.7, 15.0)
                if polish_time > 2:
                    polish_result, polish_size = exact_local_polish(best_support_set, max_time=polish_time)
                    if polish_size > best_size:
                        best_size = polish_size
                        best_support_set = polish_result.copy()

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree_supports()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()

        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.5, 10.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()

        # Polish the current best with an exact local solve
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4:
            polish_time = min(remaining_time * 0.6, 15.0)
            if polish_time > 3:
                polish_result, polish_size = exact_local_polish(best_support_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_support_set = polish_result.copy()

    # --- Final deterministic polish on the best support set ---
    final_polish_time = max(0.0, deadline - 2 - time.time())
    if final_polish_time > 1:
        fp_result, fp_size = exact_local_polish(best_support_set, max_time=final_polish_time)
        if fp_size > best_size:
            best_size = fp_size
            best_support_set = fp_result.copy()
        # Second polish pass for further improvement
        final_polish_time2 = max(0.0, deadline - 1 - time.time())
        if final_polish_time2 > 1:
            fp_result2, fp_size2 = exact_local_polish(best_support_set, max_time=final_polish_time2)
            if fp_size2 > best_size:
                best_size = fp_size2
                best_support_set = fp_result2.copy()
        # Third polish pass with remaining time
        final_polish_time3 = max(0.0, deadline - 0.5 - time.time())
        if final_polish_time3 > 0.5:
            fp_result3, fp_size3 = exact_local_polish(best_support_set, max_time=final_polish_time3)
            if fp_size3 > best_size:
                best_size = fp_size3
                best_support_set = fp_result3.copy()

    # --- Lift support independent set to full W4 vectors ---
    selected_supports = np.where(best_support_set)[0]
    all_v4 = []
    for si in selected_supports:
        all_v4.extend(support_to_indices[supports[si]])
    if len(all_v4) > 0:
        result_v4 = vectors_v4[np.array(all_v4, dtype=np.int64)]
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
