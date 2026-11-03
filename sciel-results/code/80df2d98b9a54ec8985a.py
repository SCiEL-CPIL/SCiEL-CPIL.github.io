import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using greedy set cover, trimming, and
    simulated annealing local search with swap moves.
    Uses inverse-index (reverse ball) for O(1) candidate enumeration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls using a tight Python loop
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)
    ball_sizes = np.array([len(b) for b in balls_list], dtype=np.int32)

    # Precompute inverse index: for each word, list of centers whose ball contains it
    # This enables O(1) candidate enumeration for covering a set of unique words
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        # Compute gains: for each center, how many currently-uncovered words does it cover?
        gains = np.zeros(total_words, dtype=np.int32)
        for c in range(total_words):
            b = balls[c]
            gains[c] = np.sum(uncovered[b])
        
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        uncovered[balls[best_center]] = False
        cover_count[balls[best_center]] += 1

    # Trimming: remove redundant codewords
    def trim_code():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    trim_code()

    # Helper: get unique words (covered only by this center)
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    # Fast exact gain computation for swap i -> j
    # gain = |{w in ball[j] : w not in ball[i] and cover_count[w]==0}| + |{w in ball[i]∩ball[j] : cover_count[w]==1}|
    #       - |{w in ball[i]∩ball[j] : cover_count[w]==1}|  [these become uncovered if we remove i, but j covers them]
    # Actually: after removing i and adding j:
    #   Words in ball[i] \ ball[j] with cover_count==1 become uncovered -> INVALID unless covered by other
    #   Words in ball[i]∩ball[j] with cover_count==1: still covered by j -> OK, gain 0
    #   Words in ball[j] \ ball[i] with cover_count==0: newly covered -> gain +1
    #   Words in ball[j] \ ball[i] with cover_count>=1: no change -> gain 0
    #   Words in ball[i]∩ball[j] with cover_count>=2: cover_count decreases by 1, no effect -> gain 0
    # So valid swap requires: for all w in ball[i]\ball[j], cover_count[w] >= 2
    # And gain = |{w in ball[j]\ball[i] : cover_count[w]==0}|
    def compute_swap_gain(i, j):
        """Returns (is_valid, gain) for swapping center i with center j."""
        bi = balls[i]
        bj = balls[j]
        bi_set = set(bi.tolist())
        bj_set = set(bj.tolist())
        
        # Check validity: all words in ball[i] not in ball[j] must have cover_count >= 2
        only_in_i = bi_set - bj_set
        for w in only_in_i:
            if cover_count[w] < 2:
                return (False, 0)
        
        # Compute gain: words in ball[j] not in ball[i] that are currently uncovered
        gain = 0
        only_in_j = bj_set - bi_set
        for w in only_in_j:
            if cover_count[w] == 0:
                gain += 1
        
        return (True, gain)

    # Local search with exact scoring
    def local_search(seed, max_rounds=200):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        
        improved = True
        swap_round = 0

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps using inverse-index for fast candidate enumeration
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                # Use inverse index to find all centers that cover ALL unique words
                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                # Get candidates from the first unique word
                candidate_set = set(inv_ball[unique_list[0]])
                
                # Intersect with candidates from remaining unique words
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                # Remove current code words from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Evaluate candidates with exact scoring
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(80, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    is_valid, gain = compute_swap_gain(i, j)
                    if is_valid and gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with different seeds
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()

    # Final trimming
    trim_code()

    # High-precision polish pass: more thorough exact local search
    def polish_pass():
        nonlocal code, cover_count
        if time.time() > deadline:
            return
        code_set = set(code)
        improved = True
        polish_round = 0
        max_polish_rounds = 100

        while improved and time.time() < deadline and polish_round < max_polish_rounds:
            improved = False
            polish_round += 1

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps with exact scoring - evaluate more candidates
            for ci in range(len(code)):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                # Get candidates from the first unique word
                candidate_set = set(inv_ball[unique_list[0]])
                
                # Intersect with candidates from remaining unique words
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                # Remove current code words from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Evaluate more candidates with exact scoring
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(200, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    is_valid, gain = compute_swap_gain(i, j)
                    if is_valid and gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    polish_pass()
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
