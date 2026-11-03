"""Rewritten program: greedy set-cover with numpy bitsets + local-search refinement."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    total = q ** n
    words = np.arange(total, dtype=np.int64)
    word_digits = np.zeros((total, n), dtype=np.int8)
    for pos in range(n):
        # digits at pos: (w // q^pos) % q
        qpos = q ** pos
        word_digits[:, pos] = (words // qpos) % q

    # Precompute neighbor masks for each codeword: a boolean array of length total
    # True if the word is within Hamming distance <= radius of this codeword.
    # We'll build these on the fly for each candidate to save memory, but cache a few.

    def hamming_mask(codeword):
        """Return boolean mask of all words within radius of codeword."""
        # codeword: list of n ints
        # Compute Hamming distance to all words
        cd = np.array(codeword, dtype=np.int8)
        dist = np.sum(word_digits != cd, axis=1)
        return dist <= radius

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    code = []

    while covered_count > 0:
        if time.time() > deadline - 10:
            break

        best_idx = -1
        best_newly = -1
        # Sample candidates: we need to pick the word that covers the most uncovered words.
        # Full scan over all total words is too slow for large total.
        # Instead, sample from uncovered words themselves (a word is always within distance 0 of itself,
        # so it covers at least itself). We also try some random words.
        # Actually, the optimal greedy choice is the word maximizing |N(c) & uncovered|.
        # We approximate by sampling.

        # Strategy: sample up to ~2000 candidates from uncovered words + random
        sample_size = min(total, 2000)
        # Get indices of uncovered words
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) > sample_size:
            # Random sample from uncovered
            sample = np.random.choice(uncov_indices, size=sample_size, replace=False)
        else:
            sample = uncov_indices

        # Also add some random words from the whole space
        if total > sample_size:
            extra = np.random.choice(total, size=min(500, total - sample_size), replace=False)
            sample = np.concatenate([sample, extra])

        for idx in sample:
            cw = words[idx]
            # Convert to list of digits
            cw_digits = [(cw // (q ** pos)) % q for pos in range(n)]
            mask = hamming_mask(cw_digits)
            newly = np.sum(mask & uncovered)
            if newly > best_newly:
                best_newly = newly
                best_idx = idx

        if best_idx == -1:
            # Fallback: pick first uncovered
            best_idx = uncov_indices[0] if len(uncov_indices) > 0 else 0

        cw = words[best_idx]
        cw_digits = [(cw // (q ** pos)) % q for pos in range(n)]
        mask = hamming_mask(cw_digits)
        uncovered &= ~mask
        covered_count = np.sum(uncovered)
        code.append(cw_digits)

    # Local search: try to remove codewords or swap to reduce count
    # We'll try a few rounds of: for each codeword, check if removing it still covers all.
    # Also try swapping a codeword with another to improve coverage.

    def is_covering(codewords_list):
        """Check if codewords_list covers the entire space."""
        if not codewords_list:
            return total == 0
        covered = np.zeros(total, dtype=bool)
        for cw in codewords_list:
            mask = hamming_mask(cw)
            covered |= mask
        return np.all(covered)

    # Round 1: try removing each codeword
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            # Temporarily remove code[i]
            remaining = code[:i] + code[i+1:]
            if is_covering(remaining):
                code = remaining
                improved = True
                break  # restart loop since code changed

    # Round 2: try swapping one codeword with another from the space
    if time.time() < deadline - 10:
        # For each codeword, try replacing it with a random word that might cover more
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            orig = code[i]
            # Try a few random replacements
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                cand_idx = np.random.randint(total)
                cand = [(cand_idx // (q ** pos)) % q for pos in range(n)]
                new_code = code[:i] + [cand] + code[i+1:]
                if is_covering(new_code):
                    # Check if this helps (it doesn't reduce count, but might allow future removals)
                    # Only accept if it enables a removal
                    code = new_code
                    # Try removing another codeword
                    for j in range(len(code)):
                        if j == i:
                            continue
                        remaining = code[:j] + code[j+1:]
                        if is_covering(remaining):
                            code = remaining
                            break
                    break

    # Final check
    if not is_covering(code):
        # Emergency: add back uncovered words
        covered = np.zeros(total, dtype=bool)
        for cw in code:
            covered |= hamming_mask(cw)
        for idx in np.where(~covered)[0]:
            cw = [(idx // (q ** pos)) % q for pos in range(n)]
            code.append(cw)

    return code
# EVOLVE-BLOCK-END
