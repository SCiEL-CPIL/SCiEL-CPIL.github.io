"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import heapq
import random


def construct(q, n, radius):
    """
    Lazy greedy set-cover for covering code K_q(n, radius) with swap-based local search.
    Uses a priority queue for efficient greedy selection and iterative swap+remove
    local search to minimize codeword count.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute ball indices for all possible codewords
    # For radius=1: ball = {word} ∪ {word with one position changed to each other value}
    ball_arrays = []
    for idx in range(total_words):
        word = index_to_word(idx, q, n)
        covered = [idx]
        for pos in range(n):
            digit = word[pos]
            weight = q ** (n - 1 - pos)
            for val in range(q):
                if val != digit:
                    new_idx = idx + (val - digit) * weight
                    covered.append(new_idx)
        ball_arrays.append(np.array(covered, dtype=np.int64))

    # Lazy greedy set cover using a max-heap (negated for min-heap)
    uncovered = np.ones(total_words, dtype=np.uint8)
    codeword_indices = []

    # Build initial heap: (-gain, idx) for each candidate
    heap = []
    for idx in range(total_words):
        gain = np.count_nonzero(uncovered[ball_arrays[idx]])
        heapq.heappush(heap, (-gain, idx))

    # Track selected set for O(1) membership
    selected_set = set()

    while uncovered.any():
        if time.time() > deadline - 12:
            break

        # Pop best candidate from heap
        best_gain = 0
        best_idx = -1
        while heap:
            neg_gain, idx = heapq.heappop(heap)
            if idx in selected_set:
                continue
            # Recompute actual gain (lazy evaluation)
            actual_gain = np.count_nonzero(uncovered[ball_arrays[idx]])
            if actual_gain > best_gain:
                best_gain = actual_gain
                best_idx = idx
            if -neg_gain != actual_gain:
                # Stale entry, push updated one
                heapq.heappush(heap, (-actual_gain, idx))
            if actual_gain == 0:
                break

        if best_idx == -1 or best_gain == 0:
            break

        codeword_indices.append(best_idx)
        selected_set.add(best_idx)
        uncovered[ball_arrays[best_idx]] = 0

        # Update heap entries for candidates affected by this selection
        # For efficiency, we'll just let lazy evaluation handle staleness
        # But push updated gains for all candidates that might have changed
        # Actually, just let the lazy mechanism handle it naturally

    # Check validity
    if uncovered.any():
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution

    # Local search: iterative removal + swap
    def try_removals(codeword_indices, cover_count):
        """Try to remove redundant codewords. Returns True if any removal made."""
        changed = False
        i = 0
        while i < len(codeword_indices) and time.time() < deadline - 15:
            ci = codeword_indices[i]
            ball = ball_arrays[ci]
            if not (cover_count[ball] == 1).any():
                cover_count[ball] -= 1
                codeword_indices.pop(i)
                selected_set.discard(ci)
                changed = True
            else:
                i += 1
        return changed

    def try_swaps(codeword_indices, cover_count, max_rounds=5):
        """Try swap-based local search to enable further removals."""
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(codeword_indices)):
                if time.time() > deadline - 10:
                    break
                ci = codeword_indices[i]
                ball_i = ball_arrays[ci]
                
                # Find words uniquely covered by ci
                unique_mask = (cover_count[ball_i] == 1)
                unique_count = np.count_nonzero(unique_mask)
                
                # If no unique words, this is just a removal candidate (handled by try_removals)
                # Try swapping with candidates that cover the unique words
                # and provide maximum additional coverage
                
                # Sample a limited set of candidates for speed
                num_candidates = min(30, total_words - len(codeword_indices))
                best_cand = -1
                best_new_gain = -1
                
                for _ in range(num_candidates):
                    cand = random.randint(0, total_words - 1)
                    if cand in selected_set:
                        continue
                    
                    ball_cand = ball_arrays[cand]
                    
                    # Check if cand covers all uniquely-covered words of ci
                    if unique_count > 0:
                        # Words uniquely covered by ci must be in ball_cand
                        # i.e., for each w in ball_i with cover_count[w]==1, w must be in ball_cand
                        # Efficient: check intersection of unique words with ball_cand
                        unique_words = ball_i[unique_mask]
                        # Check if all unique_words are in ball_cand
                        # Use: create a set from ball_cand for O(1) lookup
                        # But that's expensive. Instead: count how many unique words are NOT in ball_cand
                        # Use numpy: mark ball_cand membership
                        # Alternative: check if len(set(unique_words) & set(ball_cand)) == unique_count
                        # For small balls (~14 elements), this is fast
                        if not np.all(np.isin(unique_words, ball_cand)):
                            continue
                    
                    # Compute gain: new words covered by cand that aren't already covered
                    # After removing ci and adding cand:
                    # Words in ball_cand that had cover_count==0 after removing ci become covered
                    # Words in ball_i not in ball_cand that had cover_count==1 become uncovered (bad)
                    # But we already checked unique words are in ball_cand, so no new uncovered
                    
                    # Gain = number of words in ball_cand with cover_count==0 (before swap)
                    # that are NOT in ball_i (since those already covered by ci)
                    # Actually: after removing ci, words in ball_i have count-1
                    # Words in ball_cand with count==0 (before any change) will have count==1 after adding cand
                    # But words in both ball_i and ball_cand: count was >=1, after removing ci count>=0, after adding cand count>=1
                    # New coverage = words in ball_cand where (cover_count - [in ball_i]) == 0
                    # i.e., cover_count[w] == 1 and w in ball_i, OR cover_count[w] == 0 and w not in ball_i
                    # Simpler: just compute net gain as new words covered
                    in_both = np.intersect1d(ball_i, ball_cand, assume_unique=True)
                    only_cand = np.setdiff1d(ball_cand, ball_i, assume_unique=True)
                    # Words in only_cand that are currently uncovered (cover_count==0)
                    new_covered = np.count_nonzero(cover_count[only_cand] == 0)
                    
                    if new_covered > best_new_gain:
                        best_new_gain = new_covered
                        best_cand = cand
                
                if best_cand != -1 and best_new_gain > 0:
                    # Perform the swap
                    cover_count[ball_i] -= 1
                    cover_count[ball_arrays[best_cand]] += 1
                    if not np.any(cover_count == 0):
                        codeword_indices[i] = best_cand
                        selected_set.discard(ci)
                        selected_set.add(best_cand)
                        improved = True
                        # Try removals immediately after a successful swap
                        try_removals(codeword_indices, cover_count)
                        break
                    else:
                        # Revert
                        cover_count[ball_arrays[best_cand]] -= 1
                        cover_count[ball_i] += 1
            
            if not improved:
                break

    # Initial removal pass
    cover_count = np.zeros(total_words, dtype=np.int32)
    for ci in codeword_indices:
        cover_count[ball_arrays[ci]] += 1

    # Iterate: removals, then swaps, then removals again
    for _ in range(10):
        if time.time() > deadline - 8:
            break
        changed = try_removals(codeword_indices, cover_count)
        if time.time() > deadline - 8:
            break
        improved = try_swaps(codeword_indices, cover_count, max_rounds=3)
        if time.time() > deadline - 8:
            break
        if not changed and not improved:
            break
        # After swaps, try more removals
        if time.time() > deadline - 8:
            break
        try_removals(codeword_indices, cover_count)

    # Final removal pass
    if time.time() < deadline - 3:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = try_removals(codeword_indices, cover_count)

    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in codeword_indices]

    # Final validity check
    final_cover = np.zeros(total_words, dtype=np.uint8)
    for ci in codeword_indices:
        final_cover[ball_arrays[ci]] = 1

    if np.any(final_cover == 0):
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution

    return codewords

# EVOLVE-BLOCK-END
