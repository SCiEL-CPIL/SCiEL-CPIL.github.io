"""Football pool K_3(7,1): minimize codewords covering all of {0,1,2}^7 within Hamming distance 1."""
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill (150s wall but keep margin)
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction for general case (may not be optimal but valid-ish)
        # For the specific task this won't be called, but handle gracefully
        return _direct_sum(q, n, radius)
    
    # Precompute all 2187 words as tuples of 7 digits
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 3 ** 7  # 2187
    
    # Precompute the "ball" (closed Hamming ball of radius 1) for each word as a Python int bitmask
    # Ball of word w: w itself + all words at Hamming distance 1
    balls = [0] * N
    for idx, w in enumerate(all_words):
        mask = 1 << idx
        # Distance 1 neighbors: change one coordinate to each of the other 2 values
        for pos in range(7):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                # Build neighbor
                neighbor = list(w)
                neighbor[pos] = val
                nidx = word_to_idx[tuple(neighbor)]
                mask |= (1 << nidx)
        balls[idx] = mask
    
    full_mask = (1 << N) - 1
    
    # Initial solution: direct sum of ternary tetracode with all tails
    # Tetracode: 9 words of length 4 that perfectly cover {0,1,2}^4 with radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices
    initial_indices = [word_to_idx[tuple(w)] for w in initial_codewords]
    
    # Track coverage as a set of selected indices and the OR of their balls
    selected = set(initial_indices)
    coverage = 0
    for idx in selected:
        coverage |= balls[idx]
    
    best_count = len(selected)
    best_selected = set(selected)
    
    # Phase 1: Greedy removal of redundant codewords
    # A codeword is redundant if its ball is fully covered by the union of the others
    def is_redundant(idx, current_coverage, selected_set):
        # Coverage without idx = current_coverage & ~balls[idx]
        # If current_coverage == (current_coverage & ~balls[idx]) then idx is redundant
        # i.e., balls[idx] & current_coverage == balls[idx] (all bits of idx's ball are in coverage)
        # Actually: removing idx is safe iff balls[idx] is a subset of (coverage_without_idx)
        # coverage_without_idx = coverage & ~balls[idx] ... no that's wrong
        # coverage_without_idx = OR of balls[j] for j in selected, j != idx
        # We can't easily compute that without recomputing. Instead:
        # idx is redundant iff for every word x in balls[idx], there exists j != idx in selected with x in balls[j]
        # Equivalent: balls[idx] & ~(coverage_without_idx) == 0
        # But we don't have coverage_without_idx directly.
        # Alternative: compute coverage_without_idx = coverage & ~balls[idx] is WRONG because
        # other words may also cover those bits.
        # Correct approach: recompute OR excluding idx. For 243 words this is O(243) big-int ops, acceptable.
        cov_without = 0
        for j in selected_set:
            if j != idx:
                cov_without |= balls[j]
        return (balls[idx] & cov_without) == balls[idx]
    
    # Do multiple passes of removal
    for _pass in range(10):
        if time.time() > deadline:
            break
        removed_any = False
        for idx in list(selected):
            if time.time() > deadline:
                break
            if is_redundant(idx, coverage, selected):
                selected.discard(idx)
                coverage = 0
                for j in selected:
                    coverage |= balls[j]
                removed_any = True
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = set(selected)
        if not removed_any:
            break
    
    # Phase 2: Swap-based local search
    # Try to replace a codeword with another to maintain coverage while potentially enabling further removals
    # We use a candidate pool: all 2187 words
    # Strategy: for each selected word, try swapping it with a random unselected word.
    # If after swap the coverage is still full, accept and try to remove more.
    
    random.seed(42)
    
    # Precompute for speed: for each word, which other words' balls intersect its ball
    # Actually, let's do a simpler but effective approach:
    # 1. Try to find a word to remove (redundant check)
    # 2. If none, try to add a word that helps, then remove two
    # This is complex. Instead, do: try replacing each selected word with each candidate,
    # check if coverage remains full. If yes, accept (count same) and retry removal.
    
    # More efficient: use a "move" based approach
    # For the current selected set, try all possible single-word replacements
    # selected -> (selected - {i}) + {j} for i in selected, j not in selected
    # Check if new coverage == full_mask
    
    # This is O(|selected| * (N - |selected|)) which is ~243 * 2187 ≈ 530k checks per iteration
    # Each check: compute coverage of (selected - {i}) + {j}
    # = (coverage_without_i) | balls[j]
    # where coverage_without_i = OR of balls[k] for k in selected, k != i
    
    # Precompute coverage_without for each i in selected
    # This is O(|selected|^2) big-int ops per iteration
    
    swap_accepted = 0
    for iteration in range(200):
        if time.time() > deadline:
            break
        if len(selected) == 0:
            break
        
        # Compute coverage_without for each selected word
        sel_list = list(selected)
        m = len(sel_list)
        cov_without = [0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    cov_without[i] |= balls[sel_list[j]]
        
        # Try swaps: for each i in selected, try adding j not in selected
        # New coverage = cov_without[i] | balls[j]
        # Accept if new coverage == full_mask
        # We want to find a swap that enables further reduction, so we accept any valid swap
        # and then try to remove redundant words again
        
        found_swap = False
        for i in range(m):
            if time.time() > deadline:
                break
            if found_swap:
                break
            # Try random candidates for j
            # To be efficient, try a limited number of random j's per i
            for _trial in range(50):
                if time.time() > deadline:
                    break
                j = random.randint(0, N - 1)
                if j in selected:
                    continue
                new_cov = cov_without[i] | balls[j]
                if new_cov == full_mask:
                    # Accept swap
                    selected.discard(sel_list[i])
                    selected.add(j)
                    coverage = new_cov
                    found_swap = True
                    swap_accepted += 1
                    break
        
        if not found_swap:
            # Try removals again
            removed_any = False
            for idx in list(selected):
                if time.time() > deadline:
                    break
                if is_redundant(idx, coverage, selected):
                    selected.discard(idx)
                    coverage = 0
                    for j in selected:
                        coverage |= balls[j]
                    removed_any = True
                    if len(selected) < best_count:
                        best_count = len(selected)
                        best_selected = set(selected)
            if not removed_any:
                # No progress, break
                break
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = set(selected)
    
    # Final verification
    final_coverage = 0
    for idx in best_selected:
        final_coverage |= balls[idx]
    if final_coverage != full_mask:
        # Fallback to initial
        best_selected = set(initial_indices)
    
    return [list(all_words[idx]) for idx in sorted(best_selected)]


def _direct_sum(q, n, radius):
    """Fallback for non-standard parameters."""
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Very slow fallback: just return all words (valid but large)
    return [list(w) for w in itertools.product(range(q), repeat=n)]
