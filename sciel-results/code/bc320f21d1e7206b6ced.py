import time
import random
import numpy as np
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1, use specialized approach
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline, start_time)

    # General fallback: use the original vectorized approach
    return _construct_general(q, n, radius, deadline, start_time)


def _construct_3_7_1(deadline, start_time):
    """
    Hybrid approach for q=3, n=7, radius=1:
    - Precompute covers as Python-int bitmasks
    - Multiple randomized greedy restarts
    - Local search with removal + swap
    - Simulated annealing with periodic greedy reduction
    """
    N = 2187  # 3^7

    # Precompute covers as bitmasks
    covers = [0] * N
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3 ** pos)
                    mask |= (1 << new_i)
        covers[i] = mask

    full_mask = (1 << N) - 1

    # Precompute cover indices as lists for fast iteration
    cover_indices = []
    for i in range(N):
        mask = covers[i]
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idxs.append(lsb.bit_length() - 1)
            m ^= lsb
        cover_indices.append(idxs)

    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []

        while uncovered:
            if time.time() > deadline - 5.0:
                break

            # Find best candidate using bit_count
            best_count = 0
            best_idx = -1
            for i in range(N):
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i

            if best_count == 0:
                break

            if randomize:
                # Collect all with max count, pick randomly
                candidates = [i for i in range(N) if (uncovered & covers[i]).bit_count() == best_count]
                best_idx = random.choice(candidates)

            selected.append(best_idx)
            uncovered &= ~covers[best_idx]

        return selected, uncovered

    def local_search_remove(code, deadline):
        """Remove redundant codewords."""
        if not code:
            return code

        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)

            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]

                # Check if removing c is safe
                if np.all(counts[cover_indices[c]] > 1):
                    counts[cover_indices[c]] -= 1
                    code.pop(idx_in_code)
                    changed = True

        return code

    def local_search_swap(code, deadline):
        """Try swaps to improve coverage or reduce size."""
        if not code:
            return code

        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        code_set = set(code)
        size = len(code)
        covered = int(np.sum(counts > 0))

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            indices_to_try = list(range(size))
            random.shuffle(indices_to_try)

            for idx_in_code in indices_to_try:
                if idx_in_code >= size:
                    continue
                c = code[idx_in_code]

                # Find essential words for c
                essential = cover_indices[c]
                ess_mask = np.zeros(N, dtype=bool)
                ess_mask[essential] = True

                # Find candidates not in code that cover all essential words
                for j in range(N):
                    if j in code_set:
                        continue
                    # Check if j covers all essential words
                    if not np.all(ess_mask[cover_indices[j]]):
                        continue

                    # Simulate swap
                    new_counts = counts.copy()
                    new_counts[cover_indices[c]] -= 1
                    new_counts[cover_indices[j]] += 1

                    # Check validity
                    if np.any(new_counts < 0):
                        continue

                    # Score: prefer fewer essential words
                    num_essentials = np.sum(new_counts == 1)
                    new_covered = int(np.sum(new_counts > 0))

                    if new_covered >= covered and num_essentials < np.sum(counts == 1):
                        # Perform swap
                        counts = new_counts
                        code[idx_in_code] = j
                        code_set.remove(c)
                        code_set.add(j)
                        covered = new_covered
                        improved = True
                        break

                if improved:
                    break

        return code

    def simulated_annealing(init_code, deadline):
        """Simulated annealing with periodic greedy reduction."""
        if not init_code:
            return init_code

        code = list(init_code)
        code_set = set(code)
        size = len(code)

        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        covered = int(np.sum(counts > 0))

        P = 10.0

        def cost(sz, cov):
            return sz + P * (N - cov)

        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost

        T = 8.0
        T_min = 0.005
        cooling = 0.99985

        all_indices = list(range(N))
        last_reduction = time.time()
        reduction_interval = 1.0

        while time.time() < deadline - 5.0:
            # Periodic greedy reduction
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    c = code[idx]
                    if np.all(counts[cover_indices[c]] > 1):
                        code.pop(idx)
                        code_set.remove(c)
                        counts[cover_indices[c]] -= 1
                        size -= 1
                        covered = int(np.sum(counts > 0))
                        current_cost = cost(size, covered)
                        if size < best_size and covered == N:
                            best_size = size
                            best_code = list(code)
                            best_covered = covered
                            best_cost = current_cost
                        break

            move_type = random.random()

            if move_type < 0.55:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                c_remove = code[idx_remove]
                idx_add = random.choice(all_indices)

                if idx_add in code_set:
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue

                nbs_remove = cover_indices[c_remove]
                nbs_add = cover_indices[idx_add]

                mask_add = np.zeros(N, dtype=bool)
                mask_add[nbs_add] = True

                cnt_rm = counts[nbs_remove]
                is_in_add = mask_add[nbs_remove]
                losing = np.sum((cnt_rm == 1) & (~is_in_add))

                cnt_add = counts[nbs_add]
                gaining = np.sum(cnt_add == 0)

                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost

                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / max(T, 1e-9)):
                    # Accept
                    counts[nbs_remove] -= 1
                    counts[nbs_add] += 1
                    code[idx_remove] = idx_add
                    code_set.remove(c_remove)
                    code_set.add(idx_add)
                    covered = new_covered
                    current_cost = new_cost

                    if size < best_size and covered == N:
                        best_size = size
                        best_code = list(code)
                        best_covered = covered
                        best_cost = current_cost

            elif move_type < 0.85:
                # Add
                idx_add = random.choice(all_indices)
                if idx_add in code_set:
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue

                nbs_add = cover_indices[idx_add]
                cnt_add = counts[nbs_add]
                gaining = np.sum(cnt_add == 0)

                new_size = size + 1
                new_covered = covered + gaining
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost

                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / max(T, 1e-9)):
                    # Accept
                    counts[nbs_add] += 1
                    code.append(idx_add)
                    code_set.add(idx_add)
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost

                    if size < best_size and covered == N:
                        best_size = size
                        best_code = list(code)
                        best_covered = covered
                        best_cost = current_cost

            else:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                c_remove = code[idx_remove]

                nbs_remove = cover_indices[c_remove]
                cnt_rm = counts[nbs_remove]
                losing = np.sum(cnt_rm == 1)

                new_size = size - 1
                new_covered = covered - losing
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost

                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / max(T, 1e-9)):
                    # Accept
                    counts[nbs_remove] -= 1
                    code.pop(idx_remove)
                    code_set.remove(c_remove)
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost

                    if size < best_size and covered == N:
                        best_size = size
                        best_code = list(code)
                        best_covered = covered
                        best_cost = current_cost

            T *= cooling
            if T < T_min:
                T = T_min

        return best_code

    # Run multiple randomized greedy restarts
    best_code = None
    best_size = N + 1

    num_restarts = 5
    time_per_restart = (deadline - start_time - 10.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        # Greedy with randomization
        selected, uncovered = greedy_cover(randomize=True, seed=restart * 12345 + 67890)

        if not selected:
            continue

        # Local search: remove redundant
        code = local_search_remove(selected, deadline)

        # Local search: swaps
        code = local_search_swap(code, deadline)

        # Simulated annealing
        sa_code = simulated_annealing(code, deadline)

        if len(sa_code) < best_size:
            best_size = len(sa_code)
            best_code = sa_code

        if time.time() > deadline - 5.0:
            break

    if best_code is None:
        # Fallback: pure greedy
        selected, _ = greedy_cover(randomize=False)
        best_code = local_search_remove(selected, deadline)

    # Convert indices to codewords
    result = []
    for i in best_code:
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)

    return result


def _construct_general(q, n, radius, deadline, start_time):
    """General fallback using the original vectorized approach."""
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    candidates = all_words
    num_candidates = len(candidates)

    coverage = np.zeros((num_candidates, total_words), dtype=bool)

    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        diff = candidates[i_start:i_end, None, :].astype(np.int8) != all_words[None, :, :].astype(np.int8)
        dists = np.sum(diff, axis=2, dtype=np.int16)
        coverage[i_start:i_end, :] = dists <= radius

    candidate_cover_counts = np.sum(coverage, axis=1, dtype=np.int32)

    def greedy_set_cover(coverage, candidate_cover_counts, deadline):
        remaining = np.ones(coverage.shape[1], dtype=np.int8)
        remaining_count = coverage.shape[1]
        selected_indices = []

        while remaining_count > 0:
            if time.time() > deadline - 20.0:
                break
            counts = coverage.dot(remaining)
            best_idx = np.argmax(counts)
            best_count = counts[best_idx]
            if best_count == 0:
                break
            selected_indices.append(best_idx)
            remaining[coverage[best_idx]] = 0
            remaining_count = np.count_nonzero(remaining)

        return selected_indices

    def local_search(selected_indices, coverage, candidate_cover_counts, deadline):
        num_candidates = coverage.shape[0]
        total_words = coverage.shape[1]

        in_set = np.zeros(num_candidates, dtype=bool)
        in_set[selected_indices] = True

        cover_counts = np.zeros(total_words, dtype=np.int16)
        for idx in selected_indices:
            cover_counts[coverage[idx]] += 1

        best_size = len(selected_indices)
        best_in_set = in_set.copy()

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            to_remove = []
            for i in range(num_candidates):
                if not in_set[i]:
                    continue
                if not np.any(coverage[i] & (cover_counts == 1)):
                    to_remove.append(i)

            if to_remove:
                for i in to_remove:
                    in_set[i] = False
                    cover_counts[coverage[i]] -= 1
                current_size = np.sum(in_set)
                if current_size < best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                improved = True
                continue

            critical_indices = []
            for i in range(num_candidates):
                if in_set[i]:
                    if np.any(coverage[i] & (cover_counts == 1)):
                        critical_indices.append(i)

            if not critical_indices:
                break

            random.shuffle(critical_indices)

            swap_found = False
            for i in critical_indices:
                if time.time() > deadline - 5.0:
                    break

                essential_words = np.where(coverage[i] & (cover_counts == 1))[0]

                if len(essential_words) == 0:
                    continue

                potential_j = np.where(coverage[:, essential_words].all(axis=1) & ~in_set)[0]

                if len(potential_j) == 0:
                    continue

                best_j = -1
                best_score = -np.inf

                cov_counts_potential = candidate_cover_counts[potential_j]
                sorted_idx = np.argsort(-cov_counts_potential)

                for idx in sorted_idx[:50]:
                    j = potential_j[idx]

                    new_cover_counts = cover_counts.copy()
                    new_cover_counts[coverage[i]] -= 1
                    new_cover_counts[coverage[j]] += 1

                    num_essentials = np.sum(new_cover_counts == 1)
                    score = -num_essentials + 0.1 * candidate_cover_counts[j]

                    if score > best_score:
                        best_score = score
                        best_j = j

                if best_j != -1:
                    in_set[i] = False
                    in_set[best_j] = True
                    cover_counts[coverage[i]] -= 1
                    cover_counts[coverage[best_j]] += 1
                    current_size = np.sum(in_set)
                    if current_size < best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    improved = True
                    swap_found = True
                    break

            if not swap_found:
                break

        return best_in_set, best_size

    initial_solution = greedy_set_cover(coverage, candidate_cover_counts, deadline)
    best_in_set, best_size = local_search(initial_solution, coverage, candidate_cover_counts, deadline)

    final_indices = np.where(best_in_set)[0]
    selected_codewords = [candidates[i].tolist() for i in final_indices]

    return selected_codewords
