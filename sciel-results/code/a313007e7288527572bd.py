import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    total_space = q ** n

    # Precompute all words as a 2D numpy array: (total_space, n)
    # words[i] is the i-th word in base-q representation
    words = np.zeros((total_space, n), dtype=np.int8)
    for i in range(total_space):
        idx = i
        for j in range(n):
            words[i, j] = idx % q
            idx //= q

    # Build the cover matrix: cover_matrix[i, j] = 1 if word j is within Hamming distance <= radius of word i
    # For radius 1: word j is covered by i if they differ in at most 1 position
    # cover_matrix[i] includes i itself and all words at distance 1 from i
    # Size: total_space x total_space = 2187 x 2187 for q=3, n=7
    # Each row has 1 + n*(q-1) = 1 + 7*2 = 15 ones

    cover_matrix = np.zeros((total_space, total_space), dtype=np.int8)
    for i in range(total_space):
        cover_matrix[i, i] = 1
        if radius >= 1:
            for pos in range(n):
                orig_val = words[i, pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Compute index of word with pos-th digit changed to val
                    # words are stored in base-q with least significant digit last in the list
                    # idx = sum(words[i,k] * q^(n-1-k)) for k in 0..n-1
                    # Actually, let's use the direct computation
                    new_idx = 0
                    for k in range(n):
                        v = words[i, k] if k != pos else val
                        new_idx = new_idx * q + v
                    cover_matrix[i, new_idx] = 1

    # Precompute row sums (each row has the same number of 1s for radius 1)
    # For greedy: we need to compute gains quickly
    # gain[i] = number of uncovered words covered by codeword i = sum(cover_matrix[i, :][uncovered])

    # Use a different approach for speed:
    # Represent uncovered set as a boolean array
    # gain[i] = np.dot(cover_matrix[i], uncovered_int)

    # Greedy set cover
    def greedy_cover():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        while uncovered.any():
            if time.time() > deadline:
                break
            # Compute gains for all candidates
            # gain[i] = sum of cover_matrix[i, j] for j where uncovered[j] = 1
            gains = cover_matrix @ uncovered  # (total_space,)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            code.append(best_idx)
            uncovered = uncovered & (1 - cover_matrix[best_idx])
        return code

    # Local search with numpy
    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        if max_time is None:
            max_time = deadline

        current_code = code[:]
        code_set = set(current_code)
        n_cw = len(current_code)

        # Build cover_count: cover_count[w] = number of codewords covering word w
        # cover_count = sum of cover_matrix[code[i], :] for all i
        cover_count = np.zeros(total_space, dtype=np.int16)
        for idx in current_code:
            cover_count += cover_matrix[idx]

        improved = True
        while improved:
            improved = False
            if time.time() > max_time:
                break
            random.shuffle(current_code)
            code_set = set(current_code)
            n_cw = len(current_code)

            for i in range(n_cw):
                if time.time() > max_time:
                    return current_code

                idx = current_code[i]
                mask = cover_matrix[idx]

                # Check if removable: all words covered by idx are covered by at least one other
                # removable if min(cover_count[mask == 1]) >= 2
                covered_by_idx = mask == 1
                if np.min(cover_count[covered_by_idx]) >= 2:
                    # Remove it
                    cover_count -= mask
                    current_code.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break
                else:
                    # Find uniquely covered words
                    unique_mask = (cover_count == 1) & covered_by_idx
                    if unique_mask.any():
                        # Find candidate j not in code_set that covers all unique words
                        # candidate covers unique words if cover_matrix[j, unique_mask] == 1 for all
                        # i.e., cover_matrix[j][unique_mask].all()
                        # For speed, check: sum(cover_matrix[j][unique_mask]) == sum(unique_mask)
                        n_unique = int(np.sum(unique_mask))
                        # Vectorized check: for all j, does cover_matrix[j][unique_mask] have all 1s?
                        # This is expensive. Instead, iterate over candidates.
                        # Optimize: only check candidates not in code_set
                        candidates = []
                        for j in range(total_space):
                            if j in code_set:
                                continue
                            # Check if j covers all unique words
                            if np.all(cover_matrix[j, unique_mask]):
                                candidates.append(j)
                        if candidates:
                            new_idx = random.choice(candidates)
                            # Swap
                            cover_count -= mask
                            cover_count += cover_matrix[new_idx]
                            code_set.remove(idx)
                            code_set.add(new_idx)
                            current_code[i] = new_idx
                            improved = True
                            break
        return current_code

    # Initial greedy
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code, max_time=deadline - 5.0)
    best_len = len(best_code)

    # Randomized restarts
    while time.time() < deadline - 5.0:
        cand_code = greedy_cover()
        cand_code = local_search_reduce(cand_code, max_time=deadline - 5.0)
        if len(cand_code) < best_len:
            best_code = cand_code
            best_len = len(cand_code)

    # Convert indices back to words
    return [words[i].tolist() for i in best_code]
