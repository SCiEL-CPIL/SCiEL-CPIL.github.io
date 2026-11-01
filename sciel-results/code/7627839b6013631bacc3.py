"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)

    # Build coverage mask: cov[cw_idx, word_idx] = 1 if cw covers word
    # Each codeword covers itself + 2*n neighbors = 15 words for q=3, n=7
    cov = np.zeros((total_words, total_words), dtype=np.int8)
    
    # Efficient coverage computation using broadcasting
    for pos in range(n):
        # For each position, compute which words differ
        for val in range(q):
            # Create mask for words that have a different value at position pos
            mask = all_words[:, pos] != val
            # For each such word, the codeword formed by setting pos to val covers it
            # This is complex to vectorize directly, so we use a different approach
            
    # Alternative: compute coverage using distance matrix
    # For radius=1, a word w is covered by codeword c if dist(w,c) <= 1
    # dist <= 1 means they are equal or differ in exactly one position
    
    # Let's build it more carefully
    # For each word i, find all codewords that cover it
    # A codeword c covers word i if:
    # 1. c == i (distance 0)
    # 2. c differs from i in exactly one position (distance 1)
    
    cov = np.zeros((total_words, total_words), dtype=np.int8)
    
    # Self-coverage
    idx = np.arange(total_words)
    cov[idx, idx] = 1
    
    # For each position, for each pair of different values, the words that differ only at that position
    for pos in range(n):
        for val1 in range(q):
            for val2 in range(q):
                if val1 == val2:
                    continue
                # Words with value val1 at position pos
                mask1 = all_words[:, pos] == val1
                # Words with value val2 at position pos  
                mask2 = all_words[:, pos] == val2
                
                # For words in mask1, the codeword formed by changing pos to val2 covers them
                # But we need to be careful: the codeword must be exactly the word with pos changed
                # This means: for word i in mask1, codeword j covers i if j == i except at pos where j has val2
                
                # Actually, let's just compute distances directly for radius 1
                pass
    
    # Simpler approach: compute the full distance matrix for radius 1
    # dist(i,j) <= 1 iff sum(all_words[i] != all_words[j]) <= 1
    
    # Chunked computation to manage memory
    chunk_size = 200
    for i_start in range(0, total_words, chunk_size):
        i_end = min(i_start + chunk_size, total_words)
        # Shape: (chunk, total_words, n)
        diff = all_words[i_start:i_end, None, :] != all_words[None, :, :]
        dists = np.sum(diff, axis=2)
        cov[i_start:i_end, :] = (dists <= radius).astype(np.int8)

    # Phase 1: Greedy Set Cover
    remaining = np.ones(total_words, dtype=bool)
    remaining_count = total_words
    selected_indices = []
    
    # Use int8 for faster dot product
    remaining_int8 = remaining.astype(np.int8)
    
    while remaining_count > 0:
        if time.time() > deadline - 40.0:
            break
        
        # Calculate coverage count for all candidates against remaining words
        counts = cov.dot(remaining_int8)
        
        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
        
        selected_indices.append(best_idx)
        
        # Update remaining
        remaining &= ~cov[best_idx].astype(bool)
        remaining_int8 = remaining.astype(np.int8)
        remaining_count = np.count_nonzero(remaining)

    # Convert indices to codewords
    selected_codewords = [all_words[i].tolist() for i in selected_indices]

    # Phase 2: Local search with swap and delete moves (inspired by the better program)
    # Represent current solution as a set of codeword indices
    current = list(selected_indices)
    
    # Compute coverage counts
    cur_cov = np.zeros(total_words, dtype=np.int32)
    for cw in current:
        cur_cov += cov[cw]
    
    best_solution = list(current)
    best_size = len(current)
    
    # Helper: try to remove redundant codewords
    def try_remove_redundant(sol, cov_counts, deadline):
        """Remove any codeword whose removal doesn't uncover words."""
        removed = 0
        i = 0
        while i < len(sol):
            if time.time() > deadline:
                break
            cw = sol[i]
            new_cov = cov_counts - cov[cw]
            if (new_cov == 0).sum() == 0:
                sol.pop(i)
                cov_counts = new_cov
                removed += 1
            else:
                i += 1
        return sol, cov_counts, removed

    # Apply initial redundancy removal
    current, cur_cov, _ = try_remove_redundant(current, cur_cov, deadline - 30)
    if len(current) < best_size:
        best_size = len(current)
        best_solution = list(current)
    
    # Local search: try swaps (replace one codeword with another) and deletions
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 15:
        improved = False
        cur_size = len(current)
        
        # Try deletions first
        for i in range(cur_size):
            if time.time() > deadline - 15:
                break
            cw = current[i]
            new_cov = cur_cov - cov[cw]
            if (new_cov == 0).sum() == 0:
                # Safe deletion
                current.pop(i)
                cur_cov = new_cov
                improved = True
                no_improvement_count = 0
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = list(current)
                break  # restart loop after modification
        
        if improved:
            continue
        
        # Try swaps: replace current[i] with some other codeword j
        # A swap is beneficial if the new codeword covers words that were only
        # covered by the old one, OR if it allows further reductions
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            old_cw = current[i]
            old_cov_mask = cov[old_cw]
            
            # Find words that are ONLY covered by old_cw (critical words)
            critical = (cur_cov == 1) & old_cov_mask.astype(bool)
            if not critical.any():
                continue
            
            # We need a replacement that covers all critical words
            # Find candidates that cover all critical words
            # For speed, try a few random candidates
            for attempt in range(15):
                if time.time() > deadline - 15:
                    break
                new_cw = random.randint(0, total_words - 1)
                if new_cw == old_cw or new_cw in current:
                    continue
                new_cov_mask = cov[new_cw]
                # Check if new_cw covers all critical words
                if new_cov_mask[critical].all():
                    # Do the swap
                    current[i] = new_cw
                    cur_cov = cur_cov - old_cov_mask + new_cov_mask
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    current, cur_cov, rem = try_remove_redundant(current, cur_cov, deadline - 15)
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Perturb: randomly add a codeword to escape local minimum
                if len(current) < total_words:
                    rand_cw = random.randint(0, total_words - 1)
                    if rand_cw not in current:
                        current.append(rand_cw)
                        cur_cov += cov[rand_cw]
                        no_improvement_count = 0
                else:
                    break
    
    # Final cleanup
    final = list(best_solution)
    final_cov = np.zeros(total_words, dtype=np.int32)
    for cw in final:
        final_cov += cov[cw]
    final, final_cov, _ = try_remove_redundant(final, final_cov, deadline - 5)
    
    # Verify coverage
    if final_cov.min() >= 1:
        return [list(all_words[idx]) for idx in final]
    
    # Fallback: return the greedy solution if it's valid
    greedy_cov = np.zeros(total_words, dtype=np.int32)
    for cw in selected_indices:
        greedy_cov += cov[cw]
    if greedy_cov.min() >= 1:
        return [list(all_words[idx]) for idx in selected_indices]
    
    # Last resort: return first few words (shouldn't happen)
    return [all_words[i].tolist() for i in range(min(200, total_words))]
# EVOLVE-BLOCK-END
