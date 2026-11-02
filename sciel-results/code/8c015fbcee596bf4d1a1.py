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

    # Precompute powers for index calculation
    powers = [q ** (n - 1 - i) for i in range(n)]

    # Precompute ball masks for each center
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
        ball_masks[center_idx] = mask

    # Greedy set cover
    all_mask = (1 << total) - 1
    uncovered = all_mask

    selected = []
    remaining_candidates = list(range(total))

    while uncovered != 0:
        if time.time() > deadline - 5:  # Safety margin
            break

        best_count = -1
        best_idx = -1
        # Iterate over remaining candidates
        for c in remaining_candidates:
            covered = ball_masks[c] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                # If we cover everything, we are done
                if cnt == uncovered.bit_count():
                    break

        if best_idx == -1 or best_count == 0:
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        try:
            remaining_candidates.remove(best_idx)
        except ValueError:
            pass

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

    # Local search: try to remove redundant codewords
    # For each codeword, check if all words it covers are also covered by other codewords
    # If so, remove it
    # This is a simple reduction step that can improve the solution further
    
    if time.time() < deadline - 10:
        # Build a list of codeword indices
        cw_indices = selected[:]
        
        # For each codeword, check if it's redundant
        # A codeword is redundant if all words in its ball are covered by other codewords
        # We need to track which codewords cover which words
        
        # Precompute for each word, which codewords cover it
        # Actually, let's just try removing each codeword one at a time
        # and check if the remaining set still covers everything
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cw_indices)):
                if time.time() > deadline - 5:
                    break
                # Try removing codeword i
                test_indices = cw_indices[:i] + cw_indices[i+1:]
                # Check if test_indices covers all words
                # Build a bitset of covered words
                covered_mask = 0
                for idx in test_indices:
                    covered_mask |= ball_masks[idx]
                if covered_mask == all_mask:
                    # Codeword i is redundant, remove it
                    del cw_indices[i]
                    improved = True
                    break
        
        # Update selected with the reduced set
        selected = cw_indices
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
