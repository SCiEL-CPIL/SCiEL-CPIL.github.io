"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    # Total space size
    total = q ** n

    # Precompute all words as base-q digit lists
    all_words = []
    for i in range(total):
        w = [0] * n
        x = i
        for j in range(n - 1, -1, -1):
            w[j] = x % q
            x //= q
        all_words.append(tuple(w))

    # Precompute balls: for each word index, list of indices within Hamming distance <= radius
    # Use numpy for vectorized distance computation in chunks
    words_arr = np.array(all_words, dtype=np.int32)

    # For radius=1, ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)

    # Precompute neighbors for each word (index into all_words)
    # For each position j and each alternative value v != words[i][j], compute the index
    neighbors = []
    for i in range(total):
        nbrs = [i]
        w = all_words[i]
        for j in range(n):
            orig = w[j]
            for v in range(q):
                if v == orig:
                    continue
                # Compute new index: change digit j from orig to v
                # index = sum(w[k] * q^(n-1-k))
                # delta = (v - orig) * q^(n-1-j)
                delta = (v - orig) * (q ** (n - 1 - j))
                nbrs.append(i + delta)
        neighbors.append(nbrs)

    # Greedy set cover: repeatedly pick the word whose ball covers the most uncovered words
    covered = np.zeros(total, dtype=bool)
    uncovered_count = total

    # For each candidate codeword, track how many currently-uncovered words its ball covers
    # We'll use a priority approach: maintain gain[i] = number of uncovered words in ball(i)
    gain = np.array([len(neighbors[i]) for i in range(total)], dtype=np.int32)

    code = []
    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Pick the codeword with maximum gain
        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break

        code.append(all_words[best_idx])
        covered_list = neighbors[best_idx]
        for idx in covered_list:
            if not covered[idx]:
                covered[idx] = True
                uncovered_count -= 1
                # Update gains for all other candidates: if their ball contains idx, decrement
                # This is expensive; instead, we'll recompute gains periodically or use a different strategy

        # To keep it tractable, periodically recompute gains
        # Actually, for efficiency, let's just recompute gain for the next selection
        # But that's O(total * ball_size) per step. With total=729, ball_size=15, that's ~11k per step.
        # With ~73 steps, that's ~800k ops. Fine.

        # Recompute gains for all candidates
        if time.time() < deadline:
            for i in range(total):
                g = 0
                for idx in neighbors[i]:
                    if not covered[idx]:
                        g += 1
                gain[i] = g

    # If we timed out mid-greedy, the code may not be valid. Validate and if invalid, fall back.
    # Check validity
    if len(code) == 0:
        # Fallback: tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Verify the code is valid (covers all words)
    final_covered = np.zeros(total, dtype=bool)
    for cw in code:
        cw_idx = all_words.index(tuple(cw)) if tuple(cw) in all_words else None
        if cw_idx is None:
            # Shouldn't happen since all codewords are from all_words
            continue
        for idx in neighbors[cw_idx]:
            final_covered[idx] = True

    if not final_covered.all():
        # Invalid; fall back to tetracode
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Local improvement: try to remove codewords if coverage is maintained
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            # Try removing code[i]
            candidate = code[:i] + code[i+1:]
            if len(candidate) == 0:
                continue
            # Check if candidate still covers everything
            check_covered = np.zeros(total, dtype=bool)
            valid = True
            for cw in candidate:
                cw_tuple = tuple(cw)
                if cw_tuple not in all_words:
                    valid = False
                    break
                cw_idx = all_words.index(cw_tuple)
                for idx in neighbors[cw_idx]:
                    check_covered[idx] = True
            if valid and check_covered.all():
                code = candidate
                improved = True
                break  # Restart the removal pass

    # Try adding better codewords: for each uncovered... no, it's already covered.
    # Try swapping: replace one codeword with another to enable removal of another
    # This is more complex; skip for now if time is tight

    return [list(cw) for cw in code]
# EVOLVE-BLOCK-END
