import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # 1. Generate all space words
    space_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    Q = space_words.shape[0]

    # Precompute distance matrix
    dist_space = (space_words[:, None, :] != space_words[None, :, :]).sum(axis=2)

    # 2. Initialize with Tetracode direct sum (81 words)
    tetracode = np.array([[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)], dtype=np.int8)
    tails = np.array(list(itertools.product(range(q), repeat=n-4)), dtype=np.int8)

    heads_expanded = np.repeat(tetracode, tails.shape[0], axis=0)
    tails_expanded = np.tile(tails, (tetracode.shape[0], 1))
    code = np.hstack([heads_expanded, tails_expanded])

    # Map code words to space indices
    space_index_map = {}
    for i, w in enumerate(space_words):
        space_index_map[tuple(w)] = i

    code_indices = np.array([space_index_map[tuple(cw)] for cw in code], dtype=np.int32)
    M = code.shape[0]
    dist_matrix = dist_space[code_indices, :]

    def is_valid_removal(dist_mat, k):
        covered_by_k = dist_mat[k] <= radius
        if not np.any(covered_by_k):
            return True
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        critical_words = covered_by_k
        if np.any(critical_words):
            if np.any(min_excl_k[critical_words] > radius):
                return False
        return True

    def is_valid_swap(dist_mat, k, new_code_idx):
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        critical_mask = (dist_mat[k] <= radius) & (min_excl_k > radius)
        if not np.any(critical_mask):
            return True
        dist_new = dist_space[new_code_idx, :]
        return np.all(dist_new[critical_mask] <= radius)

    def get_critical_words(dist_mat, k):
        """Return mask of words that are ONLY covered by codeword k."""
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        return (dist_mat[k] <= radius) & (min_excl_k > radius)

    def score_swap_candidate(dist_mat, k, new_code_idx):
        """Score how good a swap is: number of critical words covered by new word."""
        critical = get_critical_words(dist_mat, k)
        if not np.any(critical):
            return 0
        dist_new = dist_space[new_code_idx, :]
        return np.sum(dist_new[critical] <= radius)

    best_code = code.copy()
    best_M = M

    def try_remove_and_optimize(code_local, code_indices_local, dist_mat_local, time_limit):
        """Try to remove words and do swaps until stuck. Returns (code, indices, dist_mat, M)."""
        M_local = code_local.shape[0]
        removed_any = True
        while removed_any and time.time() < time_limit:
            removed_any = False
            M_local = code_local.shape[0]
            for k in range(M_local - 1, -1, -1):
                if time.time() > time_limit:
                    break
                if is_valid_removal(dist_mat_local, k):
                    code_local = np.delete(code_local, k, axis=0)
                    code_indices_local = np.delete(code_indices_local, k, axis=0)
                    dist_mat_local = np.delete(dist_mat_local, k, axis=0)
                    M_local -= 1
                    removed_any = True
                    break

        # Try swaps to escape local minima
        swap_found = True
        while swap_found and time.time() < time_limit:
            swap_found = False
            M_local = code_local.shape[0]
            for k in range(M_local):
                if time.time() > time_limit:
                    break

                # Find critical words for this codeword
                critical = get_critical_words(dist_mat_local, k)
                if not np.any(critical):
                    # This codeword covers no unique words - try to remove it
                    if is_valid_removal(dist_mat_local, k):
                        code_local = np.delete(code_local, k, axis=0)
                        code_indices_local = np.delete(code_indices_local, k, axis=0)
                        dist_mat_local = np.delete(dist_mat_local, k, axis=0)
                        M_local -= 1
                        swap_found = True
                        break
                    continue

                # Try to find a better position
                # Score candidates: prefer words that cover many critical words
                critical_count = np.sum(critical)

                # Try all words that are close to at least one critical word
                # For efficiency, sample candidates
                best_candidate = -1
                best_score = 0

                # First try: words at distance <= radius from some critical word
                critical_indices = np.where(critical)[0]
                # For each critical word, find nearby words
                candidate_set = set()
                for ci in critical_indices[:5]:  # Limit to 5 critical words for speed
                    # Find words within radius of this critical word
                    nearby = np.where(dist_space[ci, :] <= radius)[0]
                    candidate_set.update(nearby.tolist())

                # Add some random candidates
                for _ in range(10):
                    candidate_set.add(random.randint(0, Q - 1))

                # Remove current position from candidates
                candidate_list = [c for c in candidate_set if c != code_indices_local[k]]

                # Score each candidate
                for cand in candidate_list:
                    if time.time() > time_limit:
                        break
                    sc = score_swap_candidate(dist_mat_local, k, cand)
                    if sc > best_score:
                        best_score = sc
                        best_candidate = cand

                # Accept if it covers all critical words or improves
                if best_candidate >= 0 and best_score == critical_count:
                    # Perfect swap
                    code_local[k] = space_words[best_candidate]
                    code_indices_local[k] = best_candidate
                    dist_mat_local[k, :] = dist_space[best_candidate, :]
                    swap_found = True
                    break
                elif best_candidate >= 0 and best_score > critical_count * 0.7:
                    # Good enough swap
                    code_local[k] = space_words[best_candidate]
                    code_indices_local[k] = best_candidate
                    dist_mat_local[k, :] = dist_space[best_candidate, :]
                    swap_found = True
                    break

        return code_local, code_indices_local, dist_mat_local, M_local

    # Main optimization loop with restarts
    stuck_count = 0
    num_restarts = 0
    max_restarts = 20

    while time.time() < deadline and num_restarts < max_restarts:
        # Try to optimize current code
        code, code_indices, dist_matrix, M = try_remove_and_optimize(
            code, code_indices, dist_matrix, deadline
        )

        if M < best_M:
            best_M = M
            best_code = code.copy()
            stuck_count = 0
            num_restarts = 0
        else:
            stuck_count += 1

        if stuck_count >= 3:
            # Restart with a different initial code
            num_restarts += 1

            # Create a new initial code by perturbing the current best
            code = best_code.copy()
            code_indices = np.array([space_index_map[tuple(cw)] for cw in code], dtype=np.int32)
            dist_matrix = dist_space[code_indices, :]
            M = code.shape[0]

            # Randomly swap some codewords to escape local minimum
            num_swaps = random.randint(3, 8)
            for _ in range(num_swaps):
                if time.time() > deadline:
                    break
                k = random.randint(0, M - 1)
                new_idx = random.randint(0, Q - 1)
                code[k] = space_words[new_idx]
                code_indices[k] = new_idx
                dist_matrix[k, :] = dist_space[new_idx, :]

            # Try to optimize the perturbed code
            code, code_indices, dist_matrix, M = try_remove_and_optimize(
                code, code_indices, dist_matrix, deadline
            )

            if M < best_M:
                best_M = M
                best_code = code.copy()
                stuck_count = 0

    # Final cleanup: try one more round of removals on the best code
    if time.time() < deadline:
        code = best_code.copy()
        code_indices = np.array([space_index_map[tuple(cw)] for cw in code], dtype=np.int32)
        dist_matrix = dist_space[code_indices, :]
        M = code.shape[0]

        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for k in range(M - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_valid_removal(dist_matrix, k):
                    code = np.delete(code, k, axis=0)
                    code_indices = np.delete(code_indices, k, axis=0)
                    dist_matrix = np.delete(dist_matrix, k, axis=0)
                    M -= 1
                    removed_any = True
                    break

        if M < best_M:
            best_M = M
            best_code = code.copy()

    return [list(cw) for cw in best_code]
