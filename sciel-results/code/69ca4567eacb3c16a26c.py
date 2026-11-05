"""
Optimized ternary covering code construction for K_3(6,1).
Uses greedy initialization + removal local search + swap-based local search.
"""
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_construct(q, n, radius, deadline - (time.time() - start_time))

    return _construct_k3_6_1(deadline - (time.time() - start_time))


def _greedy_construct(q, n, radius, time_limit):
    """Simple greedy fallback for non-standard parameters."""
    total_words = q ** n
    if total_words > 500000:
        return []

    # Precompute neighbors for radius 1
    words = np.array([[(i // (q ** (n - 1 - pos))) % q for pos in range(n)] for i in range(total_words)])
    neighbor_lists = []
    for i in range(total_words):
        w = words[i]
        nbrs = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_w[j]
                    nbrs.append(idx)
        neighbor_lists.append(nbrs)

    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    sphere = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        sphere[i, neighbor_lists[i]] = True

    while time.time() < time.time() + time_limit:
        gains = (sphere & (~covered)[np.newaxis, :]).sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        code_indices.append(best_idx)
        covered[sphere[best_idx]] = True

    result = []
    for idx in code_indices:
        w = [(idx // (q ** (n - 1 - pos))) % q for pos in range(n)]
        result.append(w)
    return result


def _construct_k3_6_1(time_limit):
    """
    Specialized constructor for K_3(6,1).
    Greedy init + removal LS + swap LS with random kicks.
    """
    n = 6
    q = 3
    total_words = 729

    # Precompute words and neighbor lists
    words = np.array([[(i // (3 ** (5 - pos))) % 3 for pos in range(6)] for i in range(total_words)])

    neighbor_lists = []
    for i in range(total_words):
        w = words[i]
        nbrs = [i]
        for pos in range(6):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = 0
                    for j in range(6):
                        idx = idx * 3 + new_w[j]
                    nbrs.append(idx)
        neighbor_lists.append(nbrs)

    # Sphere matrix: sphere[i, j] = True if j is within Hamming distance 1 of i
    sphere = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        sphere[i, neighbor_lists[i]] = True

    # Greedy initialization
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []

    while time.time() < time.time() + time_limit - 10.0:
        gains = (sphere & (~covered)[np.newaxis, :]).sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        code_indices.append(best_idx)
        covered[sphere[best_idx]] = True

    current_code = code_indices[:]
    best_code = current_code[:]
    best_size = len(best_code)

    # Build count array
    count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        count[sphere[c]] += 1

    # in_code boolean array for O(1) lookup
    in_code = np.zeros(total_words, dtype=bool)
    for c in current_code:
        in_code[c] = True

    search_time_limit = time.time() + time_limit

    # Local search: removal + swap with random kicks
    stuck_count = 0
    max_stuck = 20

    while time.time() < search_time_limit:
        modified = False

        # Shuffle for diversity
        random.shuffle(current_code)

        # Pass 1: Try removals
        i = 0
        while i < len(current_code):
            c = current_code[i]
            # Check if removing c is safe: min count in sphere[c] > 1
            can_remove = True
            for j in neighbor_lists[c]:
                if count[j] <= 1:
                    can_remove = False
                    break

            if can_remove:
                # Remove c
                count[sphere[c]] -= 1
                in_code[c] = False
                current_code.pop(i)
                modified = True
                # Don't increment i
            else:
                i += 1

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]

        # Pass 2: Try swaps (remove one, add another)
        if not modified:
            # Find all codewords that cannot be removed
            # For each such codeword c_out, find uncovered words in its sphere
            # Then try to find a c_in not in code that covers all those uncovered words
            for c_out in current_code[:]:
                if not in_code[c_out]:
                    continue
                # Find words in sphere[c_out] with count == 1 (would become uncovered)
                critical_words = []
                for j in neighbor_lists[c_out]:
                    if count[j] == 1:
                        critical_words.append(j)

                if not critical_words:
                    continue  # Can be removed, handled in pass 1

                # Try to find a replacement
                # The replacement must cover all critical_words
                # Candidate set: words that are in the union of spheres of critical_words
                # and not already in code
                candidate_set = set()
                for j in critical_words:
                    for k in neighbor_lists[j]:
                        if not in_code[k]:
                            candidate_set.add(k)

                # Shuffle candidates
                candidates = list(candidate_set)
                random.shuffle(candidates)

                found_swap = False
                for c_in in candidates:
                    # Check if c_in covers all critical_words
                    covers_all = True
                    for u in critical_words:
                        if not sphere[c_in, u]:
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        count[sphere[c_out]] -= 1
                        in_code[c_out] = False
                        current_code.remove(c_out)

                        count[sphere[c_in]] += 1
                        in_code[c_in] = True
                        current_code.append(c_in)

                        modified = True
                        found_swap = True
                        break

                if found_swap:
                    break

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]

        if modified:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                break

            # Random kick: add a random codeword to escape local minimum
            rand_idx = random.randint(0, total_words - 1)
            if not in_code[rand_idx]:
                in_code[rand_idx] = True
                count[sphere[rand_idx]] += 1
                current_code.append(rand_idx)

    # Convert best_code indices back to words
    result = []
    for idx in best_code:
        w = words[idx].tolist()
        result.append(w)

    return result
