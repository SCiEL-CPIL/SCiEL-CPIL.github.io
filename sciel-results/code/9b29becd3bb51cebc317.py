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
    Vectorized greedy + local search for q=3, n=7, radius=1.
    Key improvements:
    - Vectorized gain computation in greedy (numpy dot product)
    - Multiple restarts enabled by speedup
    - Efficient deletion/swap local search
    - Final polish pass
    """
    N = 2187  # 3^7
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball masks as numpy arrays of covered indices
    # ball_indices[i] = list of indices covered by codeword i
    ball_indices = []
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        covered = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    covered.append(new_i)
        ball_indices.append(covered)

    # Convert to numpy arrays for vectorized operations
    # cover_matrix[i, j] = 1 if word j is covered by codeword i
    cover_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in ball_indices[i]:
            cover_matrix[i, j] = 1

    # Precompute cover counts per candidate
    cover_counts = np.sum(cover_matrix, axis=1, dtype=np.int32)

    def vectorized_greedy(seed=None, randomize=True):
        """Vectorized greedy set cover using numpy."""
        if seed is not None:
            random.seed(seed)

        uncovered = np.ones(N, dtype=np.int8)
        selected = []
        selected_set = set()

        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break

            # Vectorized gain computation: count uncovered words covered by each candidate
            gains = np.dot(cover_matrix, uncovered)

            # Mask out already selected
            for s in selected:
                gains[s] = 0

            if randomize:
                # Add small random noise to break ties
                noise = np.random.random(N) * 0.1
                gains = gains + noise

            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]

            if best_gain == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[cover_matrix[best_idx] == 1] = 0

        return selected

    def local_search(code, deadline):
        """Deletion and swap local search."""
        code_set = set(code)

        # Build coverage counts
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[cover_matrix[c] == 1] += 1

        best_code = list(code)
        best_size = len(code)

        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False

            # Phase 1: Deletion - remove non-essential codewords
            to_remove = []
            for c in code:
                covered_words = np.where(cover_matrix[c] == 1)[0]
                if not np.any(cnt[covered_words] == 1):
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    cnt[cover_matrix[c] == 1] -= 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                improved = True
                continue

            # Phase 2: Swap - try to replace essential codewords
            swaps_done = 0
            max_swaps = 30
            random.shuffle(code)

            for c in code:
                if swaps_done >= max_swaps or time.time() > deadline - 10.0:
                    break

                # Find words uniquely covered by c
                covered_words = np.where(cover_matrix[c] == 1)[0]
                unique_words = covered_words[cnt[covered_words] == 1]

                if len(unique_words) == 0:
                    continue

                # Find candidates that cover all unique words
                # A candidate j covers all unique words if cover_matrix[j, unique_words].all()
                potential = np.where(cover_matrix[:, unique_words].all(axis=1))[0]
                # Exclude already selected
                potential = [j for j in potential if j not in code_set]

                if not potential:
                    continue

                # Pick best candidate by coverage count
                best_j = max(potential, key=lambda j: cover_counts[j])

                # Perform swap
                code.remove(c)
                code_set.remove(c)
                cnt[cover_matrix[c] == 1] -= 1

                code.append(best_j)
                code_set.add(best_j)
                cnt[cover_matrix[best_j] == 1] += 1

                swaps_done += 1
                improved = True

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

            # After swaps, try deletion again
            if improved:
                to_remove = []
                for c in code:
                    covered_words = np.where(cover_matrix[c] == 1)[0]
                    if not np.any(cnt[covered_words] == 1):
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code.remove(c)
                        code_set.remove(c)
                        cnt[cover_matrix[c] == 1] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    improved = True

        return best_code

    def polish(code, deadline):
        """Final high-precision polish pass with aggressive removal."""
        code_set = set(code)
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[cover_matrix[c] == 1] += 1

        # Repeatedly try to remove any non-essential codeword
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for c in list(code):
                if c not in code_set:
                    continue
                covered_words = np.where(cover_matrix[c] == 1)[0]
                if not np.any(cnt[covered_words] == 1):
                    code.remove(c)
                    code_set.remove(c)
                    cnt[cover_matrix[c] == 1] -= 1
                    changed = True

        # Try single swaps one more time
        random.shuffle(code)
        for c in code:
            if time.time() > deadline - 5.0:
                break
            covered_words = np.where(cover_matrix[c] == 1)[0]
            unique_words = covered_words[cnt[covered_words] == 1]
            if len(unique_words) == 0:
                continue

            potential = np.where(cover_matrix[:, unique_words].all(axis=1))[0]
            potential = [j for j in potential if j not in code_set]

            if not potential:
                continue

            # Try each potential swap
            best_j = -1
            best_score = -1
            for j in potential:
                # Score: how many additional words j covers that aren't uniquely covered by others
                j_covered = np.where(cover_matrix[j] == 1)[0]
                score = np.sum((cnt[j_covered] == 0) | (cnt[j_covered] == 1))
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j != -1:
                code.remove(c)
                code_set.remove(c)
                cnt[cover_matrix[c] == 1] -= 1
                code.append(best_j)
                code_set.add(best_j)
                cnt[cover_matrix[best_j] == 1] += 1

                # Try removal again
                for c2 in list(code):
                    if c2 not in code_set:
                        continue
                    covered_words2 = np.where(cover_matrix[c2] == 1)[0]
                    if not np.any(cnt[covered_words2] == 1):
                        code.remove(c2)
                        code_set.remove(c2)
                        cnt[cover_matrix[c2] == 1] -= 1

        return code

    # Run multiple restarts
    best_code = None
    best_size = N + 1

    # Determine number of restarts based on available time
    elapsed = time.time() - start_time
    remaining = deadline - 5.0 - elapsed
    num_restarts = max(3, min(15, int(remaining / 5.0)))

    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        # Vectorized greedy initialization
        selected = vectorized_greedy(seed=restart * 7919 + 12345, randomize=True)

        if not selected:
            continue

        # Local search
        code = local_search(selected, deadline)

        # Polish
        code = polish(code, deadline)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code

        # Early exit if we hit the known optimum
        if best_size <= 186:
            break

    if best_code is None:
        # Fallback
        selected = vectorized_greedy(seed=42, randomize=False)
        best_code = local_search(selected, deadline)

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
                    cover_counts[coverage[j]] += 1
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
