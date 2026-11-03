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
        # Use: gain[c] = |ball[c] ∩ uncovered| = ball_size[c] - |ball[c] ∩ covered|
        # But covered = ~uncovered, so we need |ball[c] ∩ uncovered| directly.
        # Efficient approach: for each center, count uncovered in its ball.
        # We can vectorize: gains = sum over each ball of the uncovered indicators.
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

    # Simulated annealing local search with inverse-index candidate enumeration
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    def compute_gain(i, j):
        """Compute the gain of swapping codeword i for codeword j.
        Gain = (# words in ball[j] uncovered or uniquely covered by i)
               - (# words in ball[i]∩ball[j] uniquely covered by i)
        Vectorized using numpy."""
        bi = balls[i]
        bj = balls[j]
        # Create a boolean mask for words in ball[i]
        in_bi = np.zeros(total_words, dtype=bool)
        in_bi[bi] = True
        # Words in ball[j] that are in ball[i]
        inter_mask = in_bi[bj]
        # Words in ball[j] that are NOT in ball[i]
        not_in_bi_mask = ~inter_mask
        # Gain from words in ball[j]∩ball[i] with cover_count==1
        gain_inter = np.sum(cover_count[bj[inter_mask]] == 1)
        # Gain from words in ball[j]\ball[i] with cover_count==0
        gain_not_inter = np.sum(cover_count[bj[not_in_bi_mask]] == 0)
        return gain_not_inter  # net gain: we lose the inter ones but they were already counted
        # Actually: removing i loses cover on ball[i] words that were unique to i.
        # Adding j gains cover on ball[j] words that were uncovered.
        # The correct gain = (# uncovered in ball[j]) - (# uniquely covered by i in ball[i])
        # But since all unique words of i must be in ball[j] (by candidate constraint),
        # gain = (# uncovered in ball[j]\ball[i]) + (# uniquely covered by i in ball[i]∩ball[j]) 
        #       - (# uniquely covered by i in ball[i]∩ball[j])
        #       = (# uncovered in ball[j]\ball[i])
        # So the simplified gain is just the uncovered words in ball[j] not in ball[i].

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

                # Try more candidates (shuffled for diversity)
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(100, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                bi_words = balls[i]
                bi_set = set(bi_words.tolist())

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    bj_set = set(bj.tolist())
                    not_in_bi = bj_set - bi_set
                    gain = 0
                    for w in not_in_bi:
                        if cover_count[w] == 0:
                            gain += 1
                    
                    if gain > best_gain:
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

    # Second round of local search with different seeds to escape local minima
    for seed in [999, 2345, 5678, 9012, 3456, 7890, 11111, 22222]:
        if time.time() < deadline:
            local_search(seed, max_rounds=100)
            trim_code()

    # Final polish: repeated trim + swap passes until convergence
    final_passes = 0
    max_final_passes = 30
    while time.time() < deadline and final_passes < max_final_passes:
        final_passes += 1
        before = len(code)
        trim_code()
        if len(code) < before:
            continue
        # Try one more round of swaps
        code_set = set(code)
        improved = False
        for ci in range(len(code) - 1, -1, -1):
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
        # Quick swap attempt
        for ci in range(len(code)):
            if time.time() > deadline:
                break
            i = code[ci]
            unique = get_unique_words(i)
            if len(unique) == 0:
                continue
            unique_list = unique.tolist()
            if not unique_list:
                continue
            candidate_set = set(inv_ball[unique_list[0]])
            for w in unique_list[1:]:
                candidate_set &= set(inv_ball[w])
                if not candidate_set:
                    break
            candidate_set -= code_set
            if not candidate_set:
                continue
            candidates = list(candidate_set)
            random.shuffle(candidates)
            candidates = candidates[:50]
            bi_set = set(balls[i].tolist())
            best_swap = None
            best_gain = -1
            for j in candidates:
                bj_set = set(balls[j].tolist())
                not_in_bi = bj_set - bi_set
                gain = sum(1 for w in not_in_bi if cover_count[w] == 0)
                if gain > best_gain:
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
        if not improved:
            break
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
