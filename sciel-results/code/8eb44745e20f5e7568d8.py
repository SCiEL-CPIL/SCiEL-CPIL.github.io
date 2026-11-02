"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build base solution: direct sum of tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if radius != 1:
        return base_solution

    # Precompute ball masks for all 2187 words using numpy bitsets
    import numpy as np

    total_words = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    # ball_masks[i] is a numpy bool array of length total_words, True where word j is within dist 1 of word i
    # We'll use integer encoding: word = sum of digits * 3^position
    # Precompute powers of 3
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i - 1] * 3

    # Convert all words to integer indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in word_to_idx.items()}

    # Precompute ball membership as a list of lists (each ball has ball_size entries)
    # For efficiency, use numpy arrays
    ball_masks = np.zeros((total_words, total_words), dtype=np.uint8)

    for i, w in enumerate(all_words):
        # The center itself
        ball_masks[i, i] = 1
        # All neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_word = list(w)
                    new_word[pos] = val
                    j = word_to_idx[tuple(new_word)]
                    ball_masks[i, j] = 1

    # Lazy greedy set cover
    # uncovered: numpy bool array, True if word is still uncovered
    uncovered = np.ones(total_words, dtype=np.uint8)
    # cover_count: how many selected codewords cover each word
    cover_count = np.zeros(total_words, dtype=np.int16)

    # Lazy evaluation: for each candidate codeword, track its marginal gain
    # We'll use a heap with lazy deletion
    # gain[i] = number of currently uncovered words in ball(i)
    gain = np.array([ball_masks[i].dot(uncovered) for i in range(total_words)], dtype=np.int32)

    # Heap entries: (-gain, codeword_index)
    heap = [(-gain[i], i) for i in range(total_words)]
    heapq.heapify(heap)

    # Stale gains tracking
    stale_gain = gain.copy()
    selected = []
    selected_set = set()

    greedy_steps = 0
    while np.any(uncovered):
        if time.time() > deadline - 30.0:
            break

        # Pop until we find a valid (non-stale, non-selected) entry
        while heap:
            neg_g, cw_idx = heapq.heappop(heap)
            g = -neg_g
            if cw_idx in selected_set:
                continue
            # Check if gain is stale
            if g != stale_gain[cw_idx]:
                continue
            # Recompute actual gain
            actual_gain = int(ball_masks[cw_idx].dot(uncovered))
            if actual_gain != g:
                # Stale - update and re-push
                stale_gain[cw_idx] = actual_gain
                heapq.heappush(heap, (-actual_gain, cw_idx))
                continue
            break
        else:
            break

        if g <= 0:
            break

        # Select this codeword
        selected.append(cw_idx)
        selected_set.add(cw_idx)
        greedy_steps += 1

        # Update uncovered and cover_count
        ball = ball_masks[cw_idx]
        newly_covered = (ball & uncovered).astype(bool)
        uncovered[ball == 1] = 0
        cover_count[ball == 1] += 1

        # Update gains for all candidates affected by newly covered words
        # For each newly covered word w, any candidate that covers w has its gain reduced by 1
        # We'll do this lazily: just mark that gains need recomputation for affected candidates
        # For efficiency with lazy heap, we update stale_gain for candidates that cover newly covered words
        # This is expensive if done naively. Instead, we'll use a simpler approach:
        # Periodically rebuild the heap, or just let lazy deletion handle it.
        # For now, we'll update stale_gain for all candidates (expensive but correct)
        # Actually, let's just let the lazy check handle it by not updating stale_gain
        # The recompute-on-pop will handle correctness.

        # To keep the heap from growing too large, periodically rebuild
        if len(heap) > 3 * total_words:
            heap = [(-stale_gain[i], i) for i in range(total_words) if i not in selected_set]
            heapq.heapify(heap)

    # Check if we have full coverage
    if not np.all(uncovered == 0):
        # Fallback: add more codewords from base solution to cover remaining
        for cw in base_solution:
            idx = word_to_idx[tuple(cw)]
            if idx not in selected_set:
                ball = ball_masks[idx]
                if np.any(ball & uncovered):
                    selected.append(idx)
                    selected_set.add(idx)
                    uncovered[ball == 1] = 0
                    cover_count[ball == 1] += 1
            if not np.any(uncovered):
                break

    if not np.any(uncovered):
        # Local search phase: try to remove redundant codewords
        if time.time() < deadline - 15.0:
            selected = _local_search_removal(selected, ball_masks, cover_count, total_words, deadline)

        # Local search phase: try swaps (remove one, add one that covers more)
        if time.time() < deadline - 5.0:
            selected = _local_search_swap(selected, ball_masks, cover_count, total_words, deadline)

        # Convert back to list of lists
        result = [list(idx_to_word[i]) for i in selected]
        # Deduplicate
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        return unique_result

    # Fallback to base solution
    return base_solution


def _local_search_removal(selected, ball_masks, cover_count, total_words, deadline):
    """Try to remove redundant codewords (those whose removal doesn't uncover any word)."""
    selected_set = set(selected)
    # Recompute cover_count from scratch for correctness
    cover_count = np.zeros(total_words, dtype=np.int16)
    for idx in selected:
        cover_count[ball_masks[idx] == 1] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each codeword
        for i in range(len(selected)):
            idx = selected[i]
            ball = ball_masks[idx]
            # Check if any word is covered only by this codeword
            only_covered = (ball == 1) & (cover_count == 1)
            if not np.any(only_covered):
                # Safe to remove
                cover_count[ball == 1] -= 1
                selected.pop(i)
                selected_set.discard(idx)
                improved = True
                break  # Restart from beginning after modification
    return selected


def _local_search_swap(selected, ball_masks, cover_count, total_words, deadline):
    """Try to swap: remove one codeword, add another that covers the newly uncovered words."""
    selected_set = set(selected)
    # Recompute cover_count
    cover_count = np.zeros(total_words, dtype=np.int16)
    for idx in selected:
        cover_count[ball_masks[idx] == 1] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # For each codeword, check if we can replace it with a better one
        # A swap is beneficial if we can remove codeword A and add codeword B such that
        # B covers all words that A uniquely covers, and B is not already selected.
        # Actually, for minimization, we want to check if removing A leaves some words uncovered,
        # and if there's a single B (not in selected) that covers all those uncovered words.
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            idx_a = selected[i]
            ball_a = ball_masks[idx_a]
            # Words that would become uncovered if we remove A
            uniquely_covered = (ball_a == 1) & (cover_count == 1)
            if not np.any(uniquely_covered):
                # A is redundant - handled in removal phase
                continue

            # Find a candidate B that covers all uniquely_covered words
            # B must cover all words in uniquely_covered
            # We can check: for each candidate B not in selected, does ball_B cover uniquely_covered?
            # This is expensive. Let's use a heuristic: pick the word in uniquely_covered
            # that is covered by the fewest candidates, and try candidates covering it.

            # Alternative: just try a few random candidates
            # Get the set of uniquely covered words
            uncovered_if_removed = np.where(uniquely_covered)[0]
            if len(uncovered_if_removed) == 0:
                continue

            # Try to find a B that covers all these words
            # Check candidates that are neighbors of the uniquely covered words
            candidates_to_try = set()
            for w_idx in uncovered_if_removed:
                # Words within distance 1 of w_idx are potential centers that cover w_idx
                # Actually, we need B such that dist(B, w_idx) <= 1 for all w_idx in uncovered_if_removed
                # This means B must be within distance 1 of ALL these words
                # The intersection of their balls
                pass

            # Simpler: try the uniquely covered words themselves as potential new centers
            # If we add w_idx as a codeword, it covers itself and its neighbors
            for w_idx in uncovered_if_removed[:5]:  # Limit tries
                if w_idx in selected_set:
                    continue
                ball_b = ball_masks[w_idx]
                # Check if ball_b covers all uniquely_covered words
                if np.all(ball_b[uniquely_covered] == 1):
                    # Check if adding B and removing A doesn't leave gaps
                    # After removing A: cover_count' = cover_count - ball_a
                    # After adding B: cover_count'' = cover_count' + ball_b
                    # Need cover_count'' >= 1 everywhere
                    new_cover = cover_count.copy()
                    new_cover[ball_a == 1] -= 1
                    new_cover[ball_b == 1] += 1
                    if np.all(new_cover >= 1):
                        # Do the swap
                        cover_count = new_cover
                        selected[i] = w_idx
                        selected_set.discard(idx_a)
                        selected_set.add(w_idx)
                        improved = True
                        break

            if improved:
                break

    return selected

# EVOLVE-BLOCK-END
