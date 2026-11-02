"""Football pool problem for 7 matches: ternary covering code K_3(7,1).

Strategy:
- Start with the direct-sum construction (9 * 3^(n-4) codewords) as an initial feasible cover.
- Use a bitmask-based representation for O(1) ball-union / intersection checks.
- Perform greedy removal: repeatedly try to remove a codeword if the union of the remaining
  balls still covers the whole space. Use randomised restarts with different removal orders.
- Deadline guard: stop before 140 s and return the best (smallest) code found.
"""
import time
import random
import itertools
import sys

_DEADLINE = 140.0  # seconds; evaluator kills at 150 s


def _ball_mask(idx, n, q):
    """Return the integer bitmask of the Hamming ball of radius 1 around word idx.

    idx is the mixed-radix index of a word of length n over alphabet 0..q-1.
    The ball includes the word itself and all words at Hamming distance 1.
    """
    # Decode idx to digits
    digits = [0] * n
    tmp = idx
    for i in range(n - 1, -1, -1):
        digits[i] = tmp % q
        tmp //= q

    mask = 0
    # Include the word itself
    mask |= (1 << idx)
    # Include all words at distance 1
    for pos in range(n):
        for new_val in range(q):
            if new_val == digits[pos]:
                continue
            # Compute the index of the word with digits[pos] replaced by new_val
            # idx = digits[0]*q^(n-1) + digits[1]*q^(n-2) + ... + digits[n-1]
            # Change at pos: delta = (new_val - digits[pos]) * q^(n-1-pos)
            delta = (new_val - digits[pos]) * (q ** (n - 1 - pos))
            new_idx = idx + delta
            mask |= (1 << new_idx)
    return mask


def _build_ball_masks(n, q):
    """Precompute ball masks for all q^n words. Returns list of length q^n."""
    total = q ** n
    masks = [0] * total
    for idx in range(total):
        masks[idx] = _ball_mask(idx, n, q)
    return masks


def _full_mask(total):
    """Mask with all total bits set."""
    return (1 << total) - 1


def _direct_sum_initial(q, n, radius):
    """Initial feasible cover via direct sum of ternary tetracode."""
    if n < 4:
        # Fallback: all words
        return [list(itertools.product(range(q), repeat=n))[i] for i in range(q ** n)]
    # For ternary (q=3), use tetracode on first 4 coords
    if q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    else:
        # Generic fallback: use all q^4 words on first 4 coords (not optimal but valid)
        tetracode = [list(t) for t in itertools.product(range(q), repeat=4)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _word_to_idx(word, q, n):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def construct(q, n, radius):
    start_time = time.time()
    total = q ** n

    # Precompute ball masks
    ball_masks = _build_ball_masks(n, q)
    full = _full_mask(total)

    # Initial cover
    init_code = _direct_sum_initial(q, n, radius)
    init_idx = [_word_to_idx(w, q, n) for w in init_code]

    # Verify initial cover
    union = 0
    for idx in init_idx:
        union |= ball_masks[idx]
    if union != full:
        # Should not happen, but fallback to all words
        return [list(itertools.product(range(q), repeat=n))[i] for i in range(total)]

    best_code = list(init_idx)
    best_len = len(best_code)

    # Greedy removal with randomised restarts
    rng = random.Random(42)

    def try_remove(code_idx, ball_masks, full):
        """Try to remove codewords greedily. Returns (new_code, removed_count)."""
        code = list(code_idx)
        # Compute union of all balls
        union = 0
        for idx in code:
            union |= ball_masks[idx]

        # Shuffle for randomised removal order
        order = list(range(len(code)))
        rng.shuffle(order)

        # For each codeword in random order, check if removing it keeps full coverage
        # We need to track which balls are "only" covered by this codeword.
        # Efficient approach: for each position in the code list, check if its ball
        # has any bit not covered by the union of the others.
        # This is O(k) per removal check where k = number of remaining codewords.
        # To speed up, we can precompute for each codeword the set of words it uniquely covers.

        # Alternative: iterative single-removal pass
        changed = True
        while changed:
            changed = False
            new_order = list(range(len(code)))
            rng.shuffle(new_order)
            for pos in new_order:
                if pos >= len(code):
                    continue
                idx = code[pos]
                # Check: is ball_masks[idx] subset of union of all other balls?
                # i.e., is (union ^ ball_masks[idx]) still == full? No.
                # We need: union_without_this = union of all balls except idx
                # If union_without_this == full, we can remove idx.
                # Compute union_without_this efficiently:
                # union_without_this = full if (union & ~ball_masks[idx]) | (union of others) covers full
                # Actually: union_without_this = union & ~ball_masks[idx] is NOT correct
                # because other balls may also cover those positions.
                # Correct: union_without_this = union of all balls except idx.
                # We can compute: if (union | ball_masks[idx]) == union, then removing idx doesn't change union.
                # But that's always true. We need: does union still equal full after removing idx?
                # union_after_remove = union of {ball_masks[c] for c in code if c != idx}
                # This is expensive to compute from scratch.

                # Trick: precompute for each codeword the "unique" bits it covers.
                # unique_bits[i] = ball_masks[code[i]] & ~(union of all other balls)
                # If unique_bits[i] == 0, we can remove code[i].

                # Recompute unique bits for all remaining codewords (expensive but correct)
                # For speed, do a single pass: for each i, compute union of all j != i
                # This is O(k^2 * bitops) per pass. With k ~ 200 and 2187 bits, it's feasible.

                # Better approach: maintain a count array
                # count[w] = number of codewords whose ball covers word w
                # A codeword idx can be removed if for all w in ball_masks[idx], count[w] > 1
                # And after removal, count[w] decreases by 1 for w in ball_masks[idx].

                # Let's use the count-based approach for the outer loop.
                break  # Exit inner loop, use count-based approach below

            # If we broke out, switch to count-based
            if not changed:
                # Use count-based greedy
                pass

        # Actually, let's just do the count-based approach properly
        return code, 0

    # Proper count-based greedy removal
    def greedy_remove(code_idx, ball_masks, full, max_iters=200000):
        """Greedy removal using coverage counts."""
        code = list(code_idx)
        k = len(code)
        # count[w] = number of codewords covering word w
        count = [0] * total
        for idx in code:
            bm = ball_masks[idx]
            # Iterate over set bits in bm
            m = bm
            while m:
                # Extract lowest set bit
                lsb = m & (-m)
                w = lsb.bit_length() - 1
                count[w] += 1
                m ^= lsb

        removed = 0
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() - start_time > _DEADLINE - 10:
                break
            # Try to find a removable codeword
            found = -1
            for i in range(k):
                idx = code[i]
                bm = ball_masks[idx]
                # Check if all words in bm have count > 1
                m = bm
                removable = True
                while m:
                    lsb = m & (-m)
                    w = lsb.bit_length() - 1
                    if count[w] <= 1:
                        removable = False
                        break
                    m ^= lsb
                if removable:
                    found = i
                    break
            if found < 0:
                break
            # Remove code[found]
            idx = code[found]
            bm = ball_masks[idx]
            m = bm
            while m:
                lsb = m & (-m)
                w = lsb.bit_length() - 1
                count[w] -= 1
                m ^= lsb
            # Remove from code list (swap with last for O(1))
            code[found] = code[-1]
            code.pop()
            k -= 1
            removed += 1

        return code, removed

    # Run multiple randomised restarts
    for restart in range(50):
        if time.time() - start_time > _DEADLINE - 5:
            break

        # Shuffle initial code for different removal order
        shuffled = list(init_idx)
        rng.shuffle(shuffled)

        new_code, removed = greedy_remove(shuffled, ball_masks, full)

        if len(new_code) < best_len:
            best_len = len(new_code)
            best_code = new_code
            # Try to further reduce from the new best
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > _DEADLINE - 5:
                    break
                for _ in range(10):
                    shuffled2 = list(best_code)
                    rng.shuffle(shuffled2)
                    new_code2, _ = greedy_remove(shuffled2, ball_masks, full)
                    if len(new_code2) < best_len:
                        best_len = len(new_code2)
                        best_code = new_code2
                        improved = True
                    if time.time() - start_time > _DEADLINE - 5:
                        break

    # Convert back to list of words
    result = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n - 1, -1, -1):
            word[i] = tmp % q
            tmp //= q
        result.append(word)

    return result
