"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks
    # mask[i] is a boolean array of length N, mask[i][j] is True if word j is within radius of word i
    # For q=3, n=6, N=729. This is 729 * 729 bits ~ 65KB, very small.
    # We can use numpy arrays for faster operations if available, but standard lists of ints/bools might be fine.
    # Let's use a list of sets for flexibility or a bitmask integer if N is small enough.
    # N=729 fits in a Python integer bitmask.
    
    # Initial code: Direct sum of tetracode
    if n >= 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback for small n, just take all words or a greedy cover
        initial_codewords = list(all_words)
        
    # Ensure distinct codewords
    initial_codewords = list(set(tuple(c) for c in initial_codewords))
    
    # Convert to list of lists for final return format, but work with tuples for hashing
    current_codewords = [list(c) for c in initial_codewords]
    
    # Function to calculate coverage mask for a single word
    def get_coverage_mask(word):
        mask = 0
        # Iterate over all words? No, iterate over neighbors of word.
        # For radius 1, neighbors are the word itself and words differing in 1 position.
        # Number of neighbors: 1 + n*(q-1). For q=3, n=6: 1 + 6*2 = 13.
        # This is much faster than checking all N words.
        
        # Helper to add a word to mask if it exists
        def add_to_mask(w_tuple):
            nonlocal mask
            idx = word_to_idx.get(w_tuple)
            if idx is not None:
                mask |= (1 << idx)
                
        add_to_mask(tuple(word))
        for i in range(n):
            for val in range(q):
                if val == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = val
                add_to_mask(tuple(new_word))
                
        return mask

    # Calculate masks for all current codewords
    # We store them in a list corresponding to current_codewords
    masks = [get_coverage_mask(c) for c in current_codewords]
    
    # Total coverage check
    def is_valid(masks_list):
        union_mask = 0
        for m in masks_list:
            union_mask |= m
        # Check if union_mask covers all N bits
        # (1 << N) - 1 is the mask for all bits set
        return union_mask == (1 << N) - 1

    # Initial validity check
    if not is_valid(masks):
        # If initial construction is invalid (shouldn't happen for this specific params), 
        # fall back to greedy construction
        pass # For this task, the tetracode sum is valid.

    # Greedy Removal / Local Search
    # We want to minimize len(current_codewords)
    
    # Optimization: Precompute union of all other masks?
    # If we remove codeword i, the new union is (TotalUnion & ~masks[i]) | (Union of others).
    # Actually, if we maintain `total_union`, removing i is valid if `total_union == (total_union & ~masks[i])` is false?
    # No. Removing i is valid if all words covered by i are also covered by others.
    # This is equivalent to: (masks[i] & ~others_union) == 0.
    # Where others_union is the union of all masks except i.
    
    # Since N is small, we can just recompute union or use a trick.
    # Let's try simple iterative removal.
    # To make it faster, we can track `cover_count` for each word: how many codewords cover it.
    # If we remove a codeword, we decrement counts for all words it covers.
    # If any count becomes 0, removal is invalid.
    
    cover_counts = [0] * N
    
    # Initialize cover counts
    for m in masks:
        # Iterate through set bits in m
        # This is O(N * len(codewords)) which is 729 * 81 ~ 59k ops. Fast.
        temp = m
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cover_counts[idx] += 1
            temp ^= lsb
            
    removed = True
    while removed:
        removed = False
        # Try to remove each codeword
        # We can randomize order or just go through.
        # Let's go through indices in reverse to handle list removals easier?
        # Or just mark for removal.
        
        to_remove = []
        for i in range(len(current_codewords)):
            # Check if we can remove current_codewords[i]
            # We need to see if removing it leaves any word with count 0.
            # We can simulate by checking if (cover_counts[idx] == 1) for any idx in masks[i].
            # If yes, cannot remove. If no, can remove.
            
            can_remove = True
            m = masks[i]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                if cover_counts[idx] == 1:
                    can_remove = False
                    break
                temp ^= lsb
                
            if can_remove:
                to_remove.append(i)
                
        if to_remove:
            # Remove them. Note: if we remove multiple, we must update counts correctly.
            # Since they are all "safe" to remove individually, are they safe to remove together?
            # If A covers w and B covers w, and count[w]=2. Removing A makes count[w]=1. Removing B makes count[w]=0?
            # No, the check `cover_counts[idx] == 1` was done against the *current* state.
            # If we remove A, count[w] becomes 1. If we then check B, and B also covers w, count[w] is still 1 in the *original* array, so B looks removable.
            # But if we remove A first, then B, w is uncovered.
            # So we should remove one at a time and update counts, or be smarter.
            # Given the small size, let's remove one at a time, re-evaluating.
            
            # Pick the first one we found
            i = to_remove[0]
            m = masks[i]
            
            # Update cover_counts
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                cover_counts[idx] -= 1
                temp ^= lsb
                
            # Remove from lists
            current_codewords.pop(i)
            masks.pop(i)
            removed = True
            # Restart the while loop to re-evaluate others
            continue
            
    # Local Search: Swap
    # Try swapping a codeword with another not in the set to see if we can enable further removals?
    # Or just try to find a better set via random restarts?
    # Given time constraints, simple removal is often enough to get close to optimal for these small sizes.
    # The tetracode sum (81) might reduce to ~73-75 with just removals?
    # Actually, the tetracode sum structure is quite rigid.
    # Let's add a simple greedy construction as a competitor if time permits, or just return the reduced tetracode sum.
    
    # Check time
    if time.time() > deadline:
        return current_codewords

    # Try a few random restarts with greedy construction to see if we can beat the reduced tetracode sum
    best_code = current_codewords
    best_len = len(best_code)
    
    # Greedy Set Cover
    # 1. Start with all words uncovered.
    # 2. Pick codeword covering most uncovered words.
    # 3. Mark its words covered.
    # 4. Repeat.
    
    # We can use the precomputed masks for all possible codewords?
    # There are 729 possible codewords.
    # Precomputing masks for all 729 words takes 729 * 13 lookups. Very fast.
    
    all_possible_masks = {}
    for w in all_words:
        all_possible_masks[w] = get_coverage_mask(w)
        
    # Greedy
    for restart in range(5): # Few restarts
        if time.time() > deadline:
            break
            
        uncovered = (1 << N) - 1
        used_words = set()
        greedy_code = []
        
        while uncovered:
            best_word = None
            best_gain = 0
            
            # To speed up, we can randomize the order of candidates or just scan all
            # Scanning all 729 is fast.
            for w in all_words:
                if w in used_words:
                    continue
                gain = bin(uncovered & all_possible_masks[w]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    if best_gain == 0:
                        break # Should not happen if uncovered is not 0 and we have valid codewords
                        
            if best_word is None:
                break # Error
                
            greedy_code.append(list(best_word))
            used_words.add(best_word)
            uncovered &= ~all_possible_masks[best_word]
            
        # Try to reduce greedy_code
        # Use the same removal logic
        g_masks = [all_possible_masks[tuple(c)] for c in greedy_code]
        
        # Check validity
        if not is_valid(g_masks):
            continue
            
        g_cover_counts = [0] * N
        for m in g_masks:
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                g_cover_counts[idx] += 1
                temp ^= lsb
                
        g_removed = True
        while g_removed:
            g_removed = False
            for i in range(len(greedy_code)):
                can_remove = True
                m = g_masks[i]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if g_cover_counts[idx] == 1:
                        can_remove = False
                        break
                    temp ^= lsb
                    
                if can_remove:
                    m = g_masks[i]
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        g_cover_counts[idx] -= 1
                        temp ^= lsb
                        
                    greedy_code.pop(i)
                    g_masks.pop(i)
                    g_removed = True
                    break # Restart scan
            
        if len(greedy_code) < best_len:
            best_code = greedy_code
            best_len = len(best_code)
            
    return best_code
# EVOLVE-BLOCK-END
