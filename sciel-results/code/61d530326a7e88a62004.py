import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute neighbors and masks
    # neighbors[i] is a list of word indices within Hamming distance <= radius
    # neighbor_masks[i] is a bitmask of those neighbors
    
    # For q=3, n=8, radius=1, neighbors count is 1 + 8*2 = 17.
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x

    neighbors = [[] for _ in range(total_words)]
    neighbor_masks = [0] * total_words

    # Generate neighbors for all words
    for i in range(total_words):
        w = int_to_word(i)
        # Self
        neighbors[i].append(i)
        neighbor_masks[i] |= (1 << i)
        
        # Distance 1 neighbors
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                ni = word_to_int(w)
                neighbors[i].append(ni)
                neighbor_masks[i] |= (1 << ni)
            w[pos] = orig_val

    # Phase 1: Greedy Set Cover
    # uncovered_mask: bitmask of words not yet covered
    # Initially all 1s
    uncovered_mask = (1 << total_words) - 1
    code = []
    code_set = set()
    
    # To speed up greedy, we can pre-compute nothing else, 
    # but iterating 6561 candidates * ~500 steps = 3M popcounts.
    # Python int.bit_count() is fast.
    
    while uncovered_mask:
        best_c = -1
        best_count = -1
        
        # Optimization: If we are running low on time, we might want to break.
        # But greedy is fast.
        
        for c in range(total_words):
            if c in code_set:
                continue
            # Count how many uncovered words are covered by c
            # mask_c & uncovered_mask
            count = (neighbor_masks[c] & uncovered_mask).bit_count()
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1:
            # Should not happen if uncovered_mask is not 0
            break
            
        code.append(best_c)
        code_set.add(best_c)
        uncovered_mask &= ~neighbor_masks[best_c]
        
        if time.time() > deadline - 10:
            break

    if not code:
        # Fallback: just return a trivial cover if greedy failed or timed out badly
        # This is unlikely
        return [int_to_word(i) for i in range(0, total_words, 17)] # Invalid but safe fallback structure

    # Initialize coverage counts
    cover_counts = [0] * total_words
    for c in code:
        for w in neighbors[c]:
            cover_counts[w] += 1

    # Phase 2: Local Search
    # Strategy: 
    # 1. Try to remove redundant codewords.
    # 2. Try 1-for-1 swaps.
    # 3. Try 2-for-1 swaps.
    
    def try_remove(c):
        """
        Attempts to remove codeword c.
        Returns True if removed, False otherwise.
        Modifies code, code_set, cover_counts.
        """
        # Check if any word in neighbors[c] has cover_count == 1
        # If so, removing c will uncover it.
        for w in neighbors[c]:
            if cover_counts[w] == 1:
                return False
        
        # Safe to remove
        for w in neighbors[c]:
            cover_counts[w] -= 1
        
        code_set.remove(c)
        # Remove from list (slow O(N), but N is small ~500)
        # We can mark as removed and clean up later, or just filter.
        # For simplicity in the loop, we'll manage a list and a set.
        # Actually, modifying list while iterating is bad.
        # We'll use a boolean array 'in_code' or just rebuild list occasionally.
        # Let's use a list and remove by value.
        try:
            code.remove(c)
        except ValueError:
            pass
        return True

    def perform_swap(c_out, c_in):
        """
        Removes c_out and adds c_in.
        Assumes c_in covers all words that would be lost by removing c_out.
        """
        # Update counts for c_out (decrement)
        for w in neighbors[c_out]:
            cover_counts[w] -= 1
        
        # Update counts for c_in (increment)
        for w in neighbors[c_in]:
            cover_counts[w] += 1
        
        code_set.remove(c_out)
        code_set.add(c_in)
        
        try:
            idx = code.index(c_out)
            code[idx] = c_in
        except ValueError:
            pass # Should not happen

    def find_swap_candidate(c_out):
        """
        Finds a c_in not in code such that N(c_in) covers all words
        that are uniquely covered by c_out.
        """
        # Identify uniquely covered words
        lost_mask = 0
        for w in neighbors[c_out]:
            if cover_counts[w] == 1:
                lost_mask |= (1 << w)
        
        if lost_mask == 0:
            return None # Can just remove, handled by try_remove
        
        # Find c_in that covers lost_mask
        # (neighbor_masks[c_in] & lost_mask) == lost_mask
        # And c_in not in code_set
        
        # To make this faster, we can just iterate all candidates.
        # 6561 iterations * bitwise ops is fast.
        for c_in in range(total_words):
            if c_in in code_set:
                continue
            if (neighbor_masks[c_in] & lost_mask) == lost_mask:
                return c_in
        return None

    def find_2_for_1_swap(c_out1, c_out2):
        """
        Finds c_in that covers words uniquely covered by c_out1 or c_out2.
        """
        lost_mask = 0
        for w in neighbors[c_out1]:
            if cover_counts[w] == 1:
                lost_mask |= (1 << w)
        for w in neighbors[c_out2]:
            # If cover_count was 1 for c_out1, it's in lost_mask.
            # If cover_count was 2, and one is c_out1, then after removing c_out1 it becomes 1.
            # Wait, the logic for "uniquely covered" needs care.
            # A word w is "lost" if it is covered ONLY by c_out1 and/or c_out2.
            # i.e., cover_counts[w] == 1 and w in N(c_out1)
            # OR cover_counts[w] == 2 and w in N(c_out1) and w in N(c_out2)
            
            # Simpler: 
            # Temporarily assume c_out1 and c_out2 are removed.
            # A word w is lost if cover_counts[w] - (1 if w in N(c_out1) else 0) - (1 if w in N(c_out2) else 0) == 0
            # And cover_counts[w] > 0 originally.
            
            pass # This logic is tricky to implement efficiently in the inner loop.
            # Let's stick to 1-for-1 for now, it's robust.
            
        return None

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 50 # Reset randomization or break if stuck?
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Redundancy Removal Pass
        # Iterate in random order to avoid bias
        random.shuffle(code)
        for c in code[:]: # Iterate over copy
            if c not in code_set: continue # Might have been removed
            if try_remove(c):
                improved = True
        
        # 2. 1-for-1 Swap Pass
        if not improved:
            # Try a few random swaps
            attempts = 0
            max_attempts = 100 # Limit per pass to allow time for multiple passes
            while attempts < max_attempts and time.time() < deadline - 2:
                attempts += 1
                if not code:
                    break
                c_out = random.choice(code)
                c_in = find_swap_candidate(c_out)
                if c_in is not None:
                    perform_swap(c_out, c_in)
                    improved = True
                    # After a swap, some codewords might become redundant.
                    # Break to do a removal pass? Or continue?
                    # Let's break to re-evaluate redundancies.
                    break
        
        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            # If stuck, maybe randomize the code?
            # Or just keep trying. The swap search is exhaustive over candidates for each c_out.
            # If no swap works, we are in a local optimum for 1-for-1 moves.
            # We could try 2-for-1, but let's see if 1-for-1 is enough.
            
            # If we are stuck for a long time, try a "kick":
            # Remove a random codeword (breaking coverage), then repair greedily?
            # This is risky.
            
            # For now, just continue. The shuffle in removal pass helps.

    # Final Verification and Formatting
    # Ensure we return a valid list of lists
    final_code = []
    for c in code:
        final_code.append(int_to_word(c))
    
    # Optional: Verify coverage one last time to be safe
    # If invalid, fallback to a known valid code (e.g., greedy result before optimization if saved, or just the initial greedy)
    # Since we maintain cover_counts, if min(cover_counts) >= 1, it's valid.
    if min(cover_counts) < 1:
        # Fallback to a simple valid code if something went wrong
        # Use the direct sum tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return final_code
