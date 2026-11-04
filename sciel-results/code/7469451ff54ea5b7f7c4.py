"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Uses greedy set cover with removal-based optimization for the ternary case.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction for other parameters
        # For general case, use a simple greedy approach
        return _greedy_construct_general(q, n, radius, deadline)
    
    # Specialized for K_3(6,1)
    return _construct_k3_6_1(deadline)


def _greedy_construct_general(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    
    # Precompute neighborhoods
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    # Greedy: pick codeword that covers most uncovered words
    uncovered = set(all_words)
    codewords = []
    
    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_count = -1
        for w in uncovered:
            count = 0
            for d in range(radius + 1):
                # Count words at distance d from w that are uncovered
                # This is expensive, so use sampling for large spaces
                pass
            # Simplified: just count neighbors
            for v in uncovered:
                if hamming_dist(w, v) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            break
        
        codewords.append(best_word)
        # Remove covered words
        to_remove = set()
        for v in uncovered:
            if hamming_dist(best_word, v) <= radius:
                to_remove.add(v)
        uncovered -= to_remove
    
    return [list(w) for w in codewords]


def _construct_k3_6_1(deadline):
    """Specialized construction for K_3(6,1) using bitset-based greedy removal and swap local search."""
    
    # 1. Precompute all 729 words and map them to indices
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N_WORDS = len(all_words)
    
    # 2. Precompute coverage bitmask for every possible word in the space
    # mask[w] is an integer where bit i is 1 if all_words[i] is within distance 1 of w
    # We need this for any potential codeword, so we compute for all 729 words
    cover_masks = [0] * N_WORDS
    for idx, w in enumerate(all_words):
        mask = 0
        # The word itself
        mask |= (1 << idx)
        # Words at distance 1
        for pos in range(6):
            for val in range(3):
                if w[pos] == val:
                    continue
                new_word = list(w)
                new_word[pos] = val
                new_idx = word_to_idx[tuple(new_word)]
                mask |= (1 << new_idx)
        cover_masks[idx] = mask
    
    # 3. Initialize with direct sum construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    # Convert to indices
    current_indices = [word_to_idx[tuple(cw)] for cw in initial_codewords]
    
    # 4. Greedy Removal Loop
    # We maintain a list of active indices
    active = list(current_indices)
    
    def is_valid_cover(indices):
        """Check if the set of indices covers all words."""
        union = 0
        for idx in indices:
            union |= cover_masks[idx]
        # Check if union is all 1s (N_WORDS bits)
        return (union == (1 << N_WORDS) - 1)
    
    # Optimization: Instead of recalculating union from scratch for every removal check,
    # we can just check if removing index i leaves any word uncovered.
    # A word is uncovered if it is covered ONLY by i.
    # So, if (Union of all) == (Union of all except i), then i is removable.
    # Union_all = U
    # Union_except_i = U ^ cover_masks[i]  (only if cover_masks[i] is subset of U, which it is)
    # Wait, U = Union_{j in active} cover_masks[j].
    # If we remove i, new_union = U ^ cover_masks[i] ONLY IF cover_masks[i] is a subset of U (always true)
    # AND NO other codeword covers any part of cover_masks[i] that is not covered by others?
    # No, that's not right.
    # Let U be the union of all active masks.
    # If we remove i, the new union is U \ cover_masks[i].
    # This is valid if (U \ cover_masks[i]) == U, which implies cover_masks[i] is empty? No.
    # It is valid if U \ cover_masks[i] == (1 << N_WORDS) - 1.
    # Which means U == (1 << N_WORDS) - 1 and (U \ cover_masks[i]) == (1 << N_WORDS) - 1.
    # This is equivalent to: Every word covered by i is also covered by some other active codeword.
    
    def try_remove(active_list):
        """Attempt to remove redundant codewords."""
        removed_any = True
        while removed_any and time.time() < deadline - 20:
            removed_any = False
            current_union = 0
            for idx in active_list:
                current_union |= cover_masks[idx]
            
            new_active = []
            for idx in active_list:
                # Check if removing idx leaves a valid cover
                # new_union = current_union without cover_masks[idx]?
                # No, we need to recalculate union of the rest.
                # Optimization: If cover_masks[idx] is a subset of (current_union ^ cover_masks[idx])? No.
                # Let's just compute union of others.
                union_others = 0
                for jdx in active_list:
                    if jdx != idx:
                        union_others |= cover_masks[jdx]
                
                if union_others == (1 << N_WORDS) - 1:
                    # We can remove idx
                    removed_any = True
                    continue
                else:
                    new_active.append(idx)
            
            if removed_any:
                active_list = new_active
        
        return active_list

    active = try_remove(active)
    
    # 5. Swap-based Local Search
    # Try to replace a codeword with another word to potentially enable further removals.
    # Or just to find a smaller cover directly.
    # We iterate through active codewords and try swapping them with any word in the space.
    # If the new set is a valid cover, we accept it (size remains same, but structure changes).
    # Then we try to remove again.
    
    # To speed up, we can prioritize swapping with words that are "far" from current codewords
    # or just randomize.
    
    # Let's do a few rounds of: Swap -> Remove
    best_size = len(active)
    best_active = list(active)
    
    for round_num in range(50): # Limit rounds to ensure time safety
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Shuffle active to break symmetry
        random.shuffle(active)
        
        for i in range(len(active)):
            if time.time() > deadline - 10:
                break
            current_idx = active[i]
            # Try swapping with a random subset of other words to save time
            # Or iterate through all if fast enough? 729 * 80 * 50 = ~3M checks.
            # Each check is O(N) union calculation. 3M * 80 ops = 240M ops. Might be too slow in Python.
            # We need a faster validity check.
            
            # Faster check:
            # Let U be union of all active.
            # Let U_except_i be union of active except i.
            # If we swap i with k, new_union = U_except_i | cover_masks[k].
            # We need new_union == FULL_MASK.
            
            # Precompute U_except_i for each i?
            # Let's compute U_total and U_except_i for the current active list once.
            
            U_total = 0
            for idx in active:
                U_total |= cover_masks[idx]
            
            # If U_total is not full, something is wrong (shouldn't happen)
            # Calculate U_except for each i
            # This is O(N^2) per round. 80^2 = 6400 union ops. Fast.
            
            U_except = [0] * len(active)
            for i_idx in range(len(active)):
                u = 0
                for j_idx in range(len(active)):
                    if i_idx != j_idx:
                        u |= cover_masks[active[j_idx]]
                U_except[i_idx] = u
            
            # Now for each i, try swapping with k
            # new_union = U_except[i] | cover_masks[k]
            # We want new_union == FULL_MASK.
            
            FULL_MASK = (1 << N_WORDS) - 1
            
            # We can try all k? 729 * 80 = 58k checks per round.
            # 58k * 50 rounds = 2.9M checks.
            # Each check is one OR and one comparison. Very fast in Python?
            # 2.9M iterations is fine.
            
            for i_idx in range(len(active)):
                if time.time() > deadline - 10:
                    break
                u_except = U_except[i_idx]
                # If u_except is already full, we don't need to swap, we can just remove?
                # But try_remove should have handled that.
                # So u_except is likely not full.
                
                # Try to find a k such that u_except | cover_masks[k] == FULL_MASK
                # This means cover_masks[k] must cover the missing bits of u_except.
                # missing = FULL_MASK ^ u_except
                # We need (cover_masks[k] & missing) == missing.
                
                missing = FULL_MASK ^ u_except
                if missing == 0:
                    # Can remove active[i_idx]
                    active.pop(i_idx)
                    improved = True
                    break # Restart loop with smaller active
                
                # Find a k that covers missing
                # We can iterate k
                # Optimization: Precompute for each missing mask? No, missing varies.
                # Just iterate k.
                found_swap = False
                for k in range(N_WORDS):
                    if k == active[i_idx]:
                        continue
                    if (cover_masks[k] & missing) == missing:
                        # Swap!
                        active[i_idx] = k
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break # Restart removal/swap loop with new structure
        
        if len(active) < best_size:
            best_size = len(active)
            best_active = list(active)
        
        if not improved:
            # No swaps found, try random perturbations?
            # Or just break if no improvement in a few rounds
            pass

    # Final removal pass
    best_active = try_remove(best_active)
    
    # Convert back to words
    result = [list(all_words[idx]) for idx in best_active]
    
    # Verify
    if not is_valid_cover(best_active):
        # Fallback to initial
        result = [list(all_words[idx]) for idx in current_indices]
        
    return result
# EVOLVE-BLOCK-END
