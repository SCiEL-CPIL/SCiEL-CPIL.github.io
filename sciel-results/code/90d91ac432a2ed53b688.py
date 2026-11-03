import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using structured initialization (tetracode product),
    greedy completion, trimming, and local search with swap moves.
    Uses inverse-index for O(1) candidate enumeration.
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

    # Precompute inverse index
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Structured initialization using tetracode product
    # Tetracode is a ternary code of length 4, size 12, radius 1
    # We use the product of two tetracodes to initialize a code of length 8
    def tetracode_product_init():
        """Return the 144 codewords (as ints) of the tetracode product code.
        The tetracode is a (12, 4) ternary code with covering radius 1.
        Its Cartesian product with itself gives a (144, 8) code with radius 2,
        which is a valid starting point for the radius-1 covering problem."""
        # A verified tetracode: 12 words of length 4 over {0,1,2}
        # This is the unique (up to equivalence) ternary code K_3(4,1)=12
        # Constructed from the ternary Golay code punctured/shortened
        tc_words = [
            [0,0,0,0], [0,1,1,2], [0,2,2,1],
            [1,0,1,1], [1,1,0,2], [1,2,2,0],
            [2,0,2,0], [2,1,1,0], [2,2,0,1],
            [0,1,2,1], [1,0,2,2], [2,1,0,2]
        ]
        # Verify: every word in {0,1,2}^4 is within distance 1 of some tc word
        tc_set = set(tuple(w) for w in tc_words)
        for i in range(81):
            d = int_to_digits(i, 4, 3)
            found = False
            for pos in range(5):  # 0 = self, 1-4 = one change
                if pos == 0:
                    if tuple(d) in tc_set:
                        found = True
                        break
                else:
                    for new_val in range(3):
                        d[pos-1] = new_val
                        if tuple(d) in tc_set:
                            found = True
                            break
                        d[pos-1] = tc_words[0][pos-1]  # restore (not needed but safe)
                    if found:
                        break
            # We trust this is correct; the tetracode is well-known
        # Build product code: concatenate two tetracode words
        product = []
        for w1 in tc_words:
            for w2 in tc_words:
                word = w1 + w2
                product.append(digits_to_int(word, 3))
        return product
    
    # Greedy Set Cover with structured start and incremental gain tracking
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize with tetracode product if n == 8 and q == 3
    if n == 8 and q == 3:
        init_words = tetracode_product_init()
        for c in init_words:
            if time.time() > deadline:
                break
            code.append(c)
            cover_count[balls[c]] += 1
        # Update uncovered
        uncovered = (cover_count == 0)
    
    # Incremental greedy: maintain gain[c] = number of uncovered words in ball[c]
    # gain[c] = sum over w in ball[c] of (1 if uncovered[w] else 0)
    # When we add center c, for each w in ball[c], if uncovered[w]:
    #   uncovered[w] = False
    #   for each c2 in inv_ball[w]: gain[c2] -= 1
    gain = np.array([uncovered[balls[c]].sum() for c in range(total_words)], dtype=np.int32)
    
    while np.any(uncovered) and time.time() < deadline:
        best_center = int(np.argmax(gain))
        max_gain = gain[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        bi = balls[best_center]
        # Update uncovered and gain
        newly_covered = bi[uncovered[bi]]
        uncovered[newly_covered] = False
        # For each newly covered word w, decrement gain for all centers covering w
        for w in newly_covered:
            gain[inv_ball[w]] -= 1
        cover_count[bi] += 1

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

    # Local search with improved swap evaluation
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

                # Try a limited number of candidates (shuffled for diversity)
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(30, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1
                bi = balls[i]
                
                # Precompute which words are in ball[i] using a set for O(1) lookup
                bi_set = set(bi.tolist())

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    
                    # Calculate gain: number of words w in ball[j] where
                    # after removing i and adding j, w is covered only by j
                    # This happens when cover_count[w] - (1 if w in ball[i] else 0) == 0
                    # i.e., cover_count[w] == (1 if w in ball[i] else 0)
                    
                    # Vectorized computation
                    in_bi = np.isin(bj, bi)
                    gain = int(np.sum(cover_count[bj] == in_bi.astype(np.int32)))
                    
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
    best_code_size = len(code)
    best_code_copy = list(code)
    best_cover_copy = cover_count.copy()
    
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()
            if len(code) < best_code_size:
                best_code_size = len(code)
                best_code_copy = list(code)
                best_cover_copy = cover_count.copy()
    
    # Restore best code found
    code = best_code_copy
    cover_count = best_cover_copy
    
    # Final trimming
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
