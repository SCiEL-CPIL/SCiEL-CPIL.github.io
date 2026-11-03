"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _ball_mask(q, n, radius, word):
    """Return a list of indices of all words within Hamming distance <= radius of word."""
    mask = []
    # Generate all positions to change (combinations of size 0..radius)
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            # For each position, choose a different symbol
            for deltas in itertools.product(range(1, q), repeat=r):
                w = list(word)
                for pos, d in zip(positions, deltas):
                    w[pos] = (w[pos] + d) % q
                # Convert to index
                idx = 0
                for sym in w:
                    idx = idx * q + sym
                mask.append(idx)
    return mask


def _verify(q, n, radius, codewords):
    """Check that every word is within distance radius of some codeword."""
    total = q ** n
    covered = [False] * total
    for cw in codewords:
        for idx in _ball_mask(q, n, radius, cw):
            covered[idx] = True
    return all(covered)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Precompute all words and their balls
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    total = q ** n

    # Precompute ball masks for all words (speeds up greedy and local search)
    all_balls = [_ball_mask(q, n, radius, w) for w in all_words]

    # Build index mapping
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx

    # Start from the tetracode construction (81 words for n=6)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    initial_indices = [word_to_idx[tuple(cw)] for cw in initial_code]

    best_code = initial_code[:]
    best_size = len(best_code)

    def _coverage_count(code_indices):
        """Return array of coverage counts for each word."""
        counts = [0] * total
        for ci in code_indices:
            for idx in all_balls[ci]:
                counts[idx] += 1
        return counts

    def _remove_redundant(code_indices, counts):
        """Remove codewords that are redundant (all their covered words are covered by others)."""
        changed = True
        while changed:
            changed = False
            new_counts = counts[:]
            for i, ci in enumerate(code_indices):
                # Check if removing ci leaves all its words still covered
                is_redundant = True
                for idx in all_balls[ci]:
                    if new_counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in all_balls[ci]:
                        new_counts[idx] -= 1
                    code_indices.pop(i)
                    counts = new_counts
                    changed = True
                    break
        return code_indices, counts

    def _try_swap(code_indices, counts):
        """Try to swap a codeword with another to maintain coverage while enabling removal."""
        # This is a heuristic; try random swaps and check if we can then remove redundant words
        if len(code_indices) <= 1:
            return code_indices, counts, False

        # Try adding a random word and removing a redundant one
        for _ in range(50):  # limited attempts per call
            if time.time() > deadline:
                break
            add_idx = random.randint(0, total - 1)
            if add_idx in code_indices:
                continue
            # Tentatively add
            new_counts = counts[:]
            for idx in all_balls[add_idx]:
                new_counts[idx] += 1
            # Now try to remove a redundant word
            removable = -1
            for i, ci in enumerate(code_indices):
                is_redundant = True
                for idx in all_balls[ci]:
                    if new_counts[idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    removable = i
                    break
            if removable >= 0:
                # Perform the swap
                for idx in all_balls[removable]:
                    new_counts[idx] -= 1
                code_indices[removable] = add_idx
                counts = new_counts
                return code_indices, counts, True
            # Revert
        return code_indices, counts, False

    # Main optimization loop
    for restart in range(200):
        if time.time() > deadline:
            break

        if restart == 0:
            code_indices = initial_indices[:]
        else:
            # Perturb: start from best and add some random words, then clean up
            code_indices = best_code_indices[:]
            num_add = random.randint(1, 5)
            for _ in range(num_add):
                add_idx = random.randint(0, total - 1)
                if add_idx not in code_indices:
                    code_indices.append(add_idx)

        counts = _coverage_count(code_indices)
        code_indices, counts = _remove_redundant(code_indices, counts)

        # Local improvement: try swaps
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                code_indices, counts, did_improve = _try_swap(code_indices, counts)
                if did_improve:
                    improved = True
                    code_indices, counts = _remove_redundant(code_indices, counts)

        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code_indices = code_indices[:]
            best_code = [all_words[ci] for ci in best_code_indices]

    # Final verification
    if not _verify(q, n, radius, best_code):
        # Fallback to initial
        return initial_code

    return best_code
# EVOLVE-BLOCK-END
