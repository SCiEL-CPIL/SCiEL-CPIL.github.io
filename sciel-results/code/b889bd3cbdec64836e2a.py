import time
import numpy as np
import random
import heapq

def construct(q, n, radius):
    """
    Constructs a covering code using heap-based greedy set cover, trimming, and
    local search with swap moves. Uses inverse-index for O(1) candidate enumeration.
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

    # Precompute balls
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

    # Precompute inverse index
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Heap-based Greedy Set Cover: O(M log M) instead of O(M^2)
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    # Initial gains for all centers
    gains = uncovered[balls].sum(axis=1)
    # Max-heap: negate for min-heap behavior
    heap = [(-int(g), i) for i, g in enumerate(gains)]
    heapq.heapify(heap)

    while np.any(uncovered) and time.time() < deadline:
        # Pop until we find a valid center
        while heap:
            neg_gain, center = heapq.heappop(heap)
            gain = -neg_gain
            # Recompute actual gain (may be stale)
            actual_gain = int(uncovered[balls[center]].sum())
            if actual_gain == gain and gain > 0:
                break
        else:
            break

        if gain == 0:
            break

        code.append(center)
        uncovered[balls[center]] = False
        cover_count[balls[center]] += 1

        # Update gains for affected centers: those whose balls intersect with balls[center]
        # A center c's gain changes if its ball shares words with balls[center]
        # We can find affected centers via the words in balls[center]
        for w in balls_list[center]:
            for c in inv_ball[w]:
                if c in code:
                    continue
                new_gain = int(uncovered[balls[c]].sum())
                heapq.heappush(heap, (-new_gain, c))

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

    # Local search with inverse-index candidate enumeration
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

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

                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(50, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    bi = balls[i]
                    in_bi = np.zeros(total_words, dtype=bool)
                    in_bi[bi] = True
                    gain = int(np.sum(cover_count[bj] == in_bi[bj]))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with different seeds
    for seed in [42, 123, 456, 789, 1024, 2048, 4096]:
        if time.time() < deadline:
            local_search(seed, max_rounds=200)
            trim_code()

    # Final trimming
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
