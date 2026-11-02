"""Covering code K_3(7,1) via greedy set cover with bitset acceleration and time guard."""
# EVOLVE-BLOCK-START
import time
import itertools

# Deadline: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # For the specific task: q=3, n=7, radius=1
    # Total space: 3^7 = 2187 words
    # Ball size (radius 1): 1 + 7*(3-1) = 1 + 14 = 15 words per ball

    # Precompute all words as integers 0..q^n - 1
    total = q ** n
    # Map integer -> tuple for output
    # We'll work with integers for speed

    # Precompute the ball (set of words within distance <= radius) for each possible center
    # Ball size: sum_{i=0}^{radius} C(n,i) * (q-1)^i
    # For q=3, n=7, radius=1: 1 + 7*2 = 15

    # Represent each word as a tuple of digits
    # Precompute digit arrays
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute balls: for each center index, list of indices within distance <= radius
    # Use bitsets (Python ints as bitsets) for fast intersection
    # total = 2187, so a bitset is an integer with 2187 bits

    # Precompute for each position and each alternative digit, the mask of words that differ
    # Actually, let's precompute the ball as a set of indices for each center.
    # For greedy set cover, we need:
    # 1. Uncovered set (bitset)
    # 2. For each candidate center, the set of uncovered words it covers

    # Let's precompute ball membership: ball_masks[center_idx] = bitset of words within distance <= radius
    ball_masks = [0] * total

    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 0
        # All words within Hamming distance <= radius
        # For radius 1: the center itself + all words differing in exactly 1 position
        # General approach: iterate over all words and check distance (too slow for 2187^2)
        # Better: enumerate the ball directly
        # For radius 1:
        #   - The center itself
        #   - For each position i, for each alt digit d != center[i], the word with that change
        if radius == 1:
            # Center itself
            mask |= (1 << center_idx)
            for pos in range(n):
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    # Compute the index of the word with position pos changed to alt
                    # We need a fast way to compute this
                    # Precompute: for each position, digit, and base index, the resulting index
                    # Actually, let's precompute a 3D array: change_idx[pos][alt][center_idx] -> new_idx
                    # But that's 7 * 3 * 2187 = 45927 entries, fine
                    pass
        break  # We'll restructure

    # Better approach: precompute transition tables
    # trans[pos][alt_digit][current_idx] -> new_idx after setting position pos to alt_digit
    # But we need the index of the word that has all digits same as center except pos has alt.
    # Let's precompute digit contributions:
    # idx = sum_{i=0}^{n-1} digit[i] * q^(n-1-i)
    # So changing digit at position pos from old to new changes idx by (new - old) * q^(n-1-pos)

    # Precompute powers
    powers = [q ** (n - 1 - i) for i in range(n)]

    # For each center, compute ball mask
    ball_masks = [0] * total
    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    new_idx = center_idx + (alt - center[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        if radius >= 2:
            # For radius 2, we'd need pairs of positions - skip for now, task is radius 1
            pass
        ball_masks[center_idx] = mask

    # Greedy set cover
    # uncovered: bitset of all words
    all_mask = (1 << total) - 1
    uncovered = all_mask

    # For each candidate center, precompute its ball mask (done above)
    # Greedy: repeatedly pick the center that covers the most uncovered words

    # To speed up, we can precompute ball_masks and use bit operations
    # But 2187 candidates * 2187 bits = manageable

    # We'll use a list of (covered_count, center_idx) and update
    # Actually, for 2187 candidates, we can just recompute the count each iteration
    # Number of iterations: at most 2187/15 ≈ 146 (lower bound), best known 186
    # Each iteration: 2187 bitset ANDs of 2187 bits = 2187 * (2187/64) word ops ≈ 74k ops
    # 186 iterations * 74k = ~14M ops, very fast in Python with big ints

    selected = []
    remaining_candidates = list(range(total))

    while uncovered != 0:
        if time.time() > deadline - 5:  # Safety margin
            break

        best_count = -1
        best_idx = -1
        for c in remaining_candidates:
            # Count uncovered words covered by center c
            covered = ball_masks[c] & uncovered
            # Count bits
            cnt = bin(covered).count('1')
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                if cnt == 0:
                    break

        if best_idx == -1 or best_count == 0:
            # Shouldn't happen if we have a valid covering
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        remaining_candidates.remove(best_idx)

    # Check if we have a valid covering
    if uncovered != 0:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert selected indices back to lists of digits
    result = []
    for idx in selected:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
