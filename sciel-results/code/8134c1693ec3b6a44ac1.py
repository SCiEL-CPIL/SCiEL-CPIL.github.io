import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls
    balls = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]

    best_size = N
    best_set = None

    # Build adjacency: adj[u] = list of v such that balls[u] intersects balls[v]
    # For a 1-codeword swap to be valid, the new codeword must be within distance 1 of all
    # critical words (words covered only by the old codeword). The critical words are in balls[c].
    # So the new codeword must be within distance 1 of every critical word.
    # If there's exactly 1 critical word, new codeword must be in balls[critical_word].
    # If there are 2 critical words at distance 2 from each other, new codeword must be in
    # balls[cw1] ∩ balls[cw2]. If distance > 2, no valid swap.
    # We can precompute for each pair of words at distance 2, the intersection of their balls.

    # For efficiency, precompute ball_membership as a bitset using Python ints
    # ball_bits[w] is a big int with bit v set iff v in balls[w]
    ball_bits = np.zeros(N, dtype=object)
    for w in range(N):
        bits = 0
        for v in balls[w]:
            bits |= (1 << v)
        ball_bits[w] = bits

    # Also precompute for each word, its 16 neighbors (excluding itself) as a list
    # neighbors[w] = list of 16 words at distance 1 from w
    neighbors = np.zeros((N, 16), dtype=np.int32)
    for w in range(N):
        for j in range(16):
            neighbors[w, j] = balls[w, 1 + j]

    # Main loop: multiple restarts
    while time.time() < deadline - 10.0:

        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline - 15.0:
                break

            counts = uncovered_mask[balls].sum(axis=1)
            if np.max(counts) == 0:
                break
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        if np.any(uncovered_mask):
            continue

        # 2. Local Search Phase
        # Phase 2a: Initial Pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # Phase 2b: Targeted 1-Swaps
        # For each codeword c, find its critical words (covered only by c).
        # A valid replacement c' must cover all critical words.
        # If |critical| == 1: c' must be in balls[critical_word] \ {c}
        # If |critical| == 2: c' must be in balls[cw1] ∩ balls[cw2] \ {c}
        # If |critical| >= 3: harder, but still possible if they're all within distance 1 of some c'
        
        # Precompute intersection of balls for pairs at distance 2
        # For words u, v at Hamming distance 2, balls[u] ∩ balls[v] has exactly 1 element
        # (the word at distance 1 from both). For distance 0, intersection is balls[u].
        # For distance 1, intersection has 7 elements.
        
        # We'll use the bitset approach for intersection
        # ball_bits[u] & ball_bits[v] gives the intersection as a bitset
        
        # To get the actual word index from a bitset, we need to find the set bit.
        # For small bitsets (at most 17 bits set), we can iterate.
        
        # For efficiency in the swap phase, let's precompute for each word w,
        # a list of all words that could replace it (i.e., words c' such that
        # if c' covers all critical words of c, the swap is valid).
        # This depends on the critical words, so we can't fully precompute.
        
        # Instead, during the swap phase, for each codeword c:
        # 1. Find critical words
        # 2. If |critical| <= 2, compute the candidate set via bitset intersection
        # 3. If |critical| >= 3, skip (or use a more expensive check)
        
        improved_in_pass = True
        passes = 0
        max_passes = 80

        while improved_in_pass and passes < max_passes and time.time() < deadline - 5.0:
            passes += 1
            improved_in_pass = False
            new_selected = []

            indices = list(range(len(selected)))
            np.random.shuffle(indices)

            for idx in indices:
                if time.time() > deadline - 5.0:
                    break

                c = selected[idx]

                # Check if c is removable
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved_in_pass = True
                    continue

                # Find critical words
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved_in_pass = True
                    continue

                # Find valid candidates
                # Candidate c' must be in balls[cw] for all critical words cw
                # Use bitset intersection
                if len(critical_words) == 1:
                    cw = critical_words[0]
                    cand_bits = ball_bits[cw]
                elif len(critical_words) == 2:
                    cand_bits = ball_bits[critical_words[0]] & ball_bits[critical_words[1]]
                elif len(critical_words) == 3:
                    cand_bits = ball_bits[critical_words[0]] & ball_bits[critical_words[1]] & ball_bits[critical_words[2]]
                else:
                    # Too many critical words, skip swap for this codeword
                    new_selected.append(c)
                    continue

                # Exclude c itself
                cand_bits &= ~(1 << c)

                if cand_bits == 0:
                    new_selected.append(c)
                    continue

                # Extract valid candidates from bitset
                # Find all set bits
                valid_candidates = []
                temp_bits = cand_bits
                while temp_bits:
                    # Find lowest set bit
                    lsb = temp_bits & (-temp_bits)
                    pos = lsb.bit_length() - 1
                    valid_candidates.append(pos)
                    temp_bits ^= lsb

                if len(valid_candidates) > 0:
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]

                    # Perform swap
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved_in_pass = True
                else:
                    new_selected.append(c)

            selected = new_selected

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    if best_set is None:
        return [idx_to_list(w) for w in range(N)]

    return [idx_to_list(w) for w in best_set]
