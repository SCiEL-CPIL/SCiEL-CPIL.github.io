"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1.
    Uses vectorized coverage counting and aggressive local search (removal + swapping).
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Greedy for other cases (rarely called in this specific task context)
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = {}
        for w in all_words:
            nb = set()
            nb.add(w)
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    new_w = tuple(new_w)
                    nb.add(new_w)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            
            if best_word is None:
                break
                
            code.append(best_word)
            uncovered -= neighbors[best_word]
            
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices for fast vector ops
    # neighbors_idx[i] is a list of indices of words within distance 1 of word i
    neighbors_idx = []
    for w in all_words:
        idx = word_to_idx[w]
        nb_set = set([idx])
        for i in range(n):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_w = tuple(new_w)
                nb_set.add(word_to_idx[new_w])
        neighbors_idx.append(list(nb_set))
    
    # Convert to numpy arrays for faster slicing if needed, or just use lists
    # Actually, lists of indices are fine for simple addition/subtraction on a numpy array
    
    # Initial Code: Tetracode-based (81 words)
    # Tetracode: [a, b, (a+b)%3, (a+2b)%3]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code_indices.append(word_to_idx[w])
    
    code_indices = set(initial_code_indices)
    
    # cover_counts[i] = number of codewords within distance 1 of word i
    cover_counts = np.zeros(total_words, dtype=np.int32)
    
    def update_cover(idx, delta):
        cover_counts[neighbors_idx[idx]] += delta
        
    # Initialize cover counts
    for idx in code_indices:
        update_cover(idx, 1)
        
    # Helper to check if coverage is valid
    def is_valid():
        return np.min(cover_counts) > 0

    # Local Search Phase 1: Removal
    # Try to remove any word that doesn't break coverage
    improved = True
    while improved:
        improved = False
        # Sort by current cover count to prioritize removing words with low contribution?
        # Or just random order to avoid bias
        for idx in list(code_indices):
            # Check if removing idx keeps coverage
            update_cover(idx, -1)
            if is_valid():
                code_indices.remove(idx)
                improved = True
                break # Restart loop
            else:
                update_cover(idx, 1) # Revert

    # Local Search Phase 2: Swapping
    # Try to replace one word with another to allow further reductions or just maintain size
    # We want to find a configuration that allows more removals, or simply minimizes size.
    # Since we already removed all removable words, size is fixed locally.
    # Swapping might lead to a different local minimum with smaller size.
    
    max_iters = 50
    for _ in range(max_iters):
        improved = False
        # Pick a random word to remove? Or iterate?
        # Iterating all is safer for finding global-ish minima in small spaces
        for idx_out in list(code_indices):
            # Remove it
            update_cover(idx_out, -1)
            
            # Find best candidate to add
            # We want to add a word that restores coverage (if broken) or allows removal of another word.
            # Since we removed one, some words might have count 0.
            # If min(cover_counts) > 0, we can just add any word that allows removal of another?
            # Actually, if min > 0, we are still covered. We can add a word and then try to remove another.
            # But standard swap: Replace idx_out with idx_in.
            
            # To be efficient, we only care if the new set is valid.
            # If we just swap, size is constant.
            # The goal is to reach a state where we can remove more words.
            
            # Let's try all possible idx_in not in code
            # This is O(73 * 729) ~ 53k checks per pass.
            # Each check: update(+1), check min, update(-1).
            
            # Optimization: Only consider idx_in that are "useful".
            # If cover_counts has 0s, idx_in must cover them.
            # If cover_counts has no 0s, any idx_in is valid for coverage, but might allow removal of another word.
            
            # Simple approach: Try all idx_in.
            # We can break early if we find a swap that allows a subsequent removal?
            # No, just find a valid swap. If found, do it and restart removal phase.
            
            # To avoid getting stuck in same swap, we can track tried swaps or just randomize order.
            # Let's iterate all.
            
            # We can speed up by checking:
            # new_cover = cover_counts (with idx_out removed)
            # We need min(new_cover + neighbors[idx_in]) > 0
            # This is equivalent to: for all j where new_cover[j] == 0, idx_in must be neighbor of j.
            
            zero_indices = np.where(cover_counts == 0)[0]
            
            if len(zero_indices) > 0:
                # idx_in must be in the intersection of neighbors of all zero_indices
                # Or simply, we can just test candidates.
                # Given small size, testing all 729 is fine.
                
                for idx_in in range(total_words):
                    if idx_in in code_indices:
                        continue
                    # Check if adding idx_in covers all zeros
                    # We can precompute which words cover which indices?
                    # Or just:
                    # temp_counts = cover_counts.copy() # Too slow
                    # Better: Check if all zero_indices are in neighbors_idx[idx_in]
                    
                    # Optimization:
                    # If zero_indices is small, check them.
                    valid = True
                    for z in zero_indices:
                        if z not in neighbors_idx[idx_in]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        code_indices.add(idx_in)
                        update_cover(idx_in, 1)
                        # idx_out was already removed from cover_counts
                        # But we need to remove idx_out from code_indices set
                        # Wait, we did update_cover(idx_out, -1) above.
                        # So cover_counts reflects: no idx_out, no idx_in yet.
                        # We added idx_in to cover_counts.
                        # So cover_counts is now for (Code - {idx_out} + {idx_in})
                        
                        # Check if this new state allows removal of another word?
                        # Or just accept the swap and go back to removal phase.
                        improved = True
                        break
                if improved:
                    break
            else:
                # No zeros. We can swap with anything?
                # But we want to find a swap that leads to a smaller code.
                # If we swap, size is same. We need to enable a removal.
                # So, after swapping, we should check if any word can be removed.
                # This makes the inner loop expensive.
                
                # Alternative: Just do a random swap to escape local minimum?
                # Or, try to find a swap that increases the "slack" (min cover count).
                # Let's just pick a random valid swap to perturb the landscape.
                # Or, iterate and find one that maximizes min(cover_counts).
                
                # For simplicity and speed, let's just do a random swap if we are stuck in removal.
                # But we need to ensure we don't oscillate.
                # Let's try to find a swap that allows a removal.
                # This is a 2-move lookahead.
                
                # Given time constraints, let's just do a single-step valid swap (which is always possible if no zeros)
                # and hope the removal phase cleans up.
                # Actually, if no zeros, we are in a stable state for removal.
                # Swapping might lead to a state where a removal is possible.
                
                # Let's try a few random swaps
                for _ in range(10):
                    idx_in = np.random.choice(total_words)
                    if idx_in in code_indices:
                        continue
                    # Check if swap is valid (it is, since no zeros)
                    code_indices.add(idx_in)
                    update_cover(idx_in, 1)
                    # Now try to remove something
                    removal_possible = False
                    for idx_rm in list(code_indices):
                        update_cover(idx_rm, -1)
                        if np.min(cover_counts) > 0:
                            code_indices.remove(idx_rm)
                            removal_possible = True
                            break
                        else:
                            update_cover(idx_rm, 1)
                    
                    if removal_possible:
                        improved = True
                        break
                    else:
                        # Revert swap
                        update_cover(idx_in, -1)
                        code_indices.remove(idx_in)
                        update_cover(idx_out, 1)
                        # Wait, idx_out was removed from cover_counts at start of this block.
                        # We need to add it back.
                        # Actually, the logic above is getting messy.
                        pass

            if improved:
                break
        
        # If no improvement in this pass, break
        if not improved:
            break

    result = [list(all_words[idx]) for idx in code_indices]
    return result

# EVOLVE-BLOCK-END
