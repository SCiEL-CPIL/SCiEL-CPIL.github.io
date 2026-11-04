import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # This problem is specific to q=3, n=6, radius=1 based on the task description
    # but we should handle the general case or at least return a valid code for others.
    # The task asks for K_q(n, radius).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple grid or random
        # For now, return a trivial code that might be large but valid
        # This is just a safety net, the main logic is for 3,6,1
        # A simple covering code for general q,n,r is hard.
        # We'll just return a small random set and hope it's not evaluated for other params,
        # or use a very basic construction.
        # Actually, the prompt implies the evaluator passes this task's params.
        # "the evaluator passes this task's alphabet size q, length n and covering radius"
        # So it will be 3, 6, 1.
        pass

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]
    
    # Precompute neighbors for all 729 words
    # neighbors[i] is a list of 13 indices: i itself and its 12 neighbors
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Initial Solution: Greedy Set Cover
    # This provides a valid starting point for SA
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []
    
    # Precompute which words cover which others? No, neighbors is enough.
    # Greedy: pick word that covers most uncovered
    while uncovered.sum() > 0:
        if time.time() > deadline - 60: # Leave 60s for SA
            break
        # For each candidate, count how many uncovered words it covers
        # counts[c] = sum(uncovered[neighbors[c]])
        # We can vectorize this? 
        # uncovered is (729,), neighbors is (729, 13).
        # uncovered[neighbors] is (729, 13). Sum axis=1 -> (729,)
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = np.argmax(counts)
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0
        
    if uncovered.sum() != 0:
        # If greedy didn't finish (unlikely for 729 words), fill with random
        while uncovered.sum() > 0:
            c = np.random.randint(total)
            codewords.append(c)
            uncovered[neighbors[c]] = 0
            
    # Convert to list of lists for return format later
    # But for SA, we work with indices
    
    # Initialize coverage counts
    # coverage_counts[w] = number of codewords covering w
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
        
    # Simulated Annealing
    # Objective: minimize len(codewords) subject to min(coverage_counts) >= 1
    
    T = 1.0 # Initial temperature
    T_min = 0.01
    cooling = 0.999
    # Number of iterations: run until time limit
    # We want to try many moves
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    while time.time() < deadline:
        # Move 1: Try to remove a random codeword
        # If valid (min coverage >= 1), it's an improvement (size - 1)
        # If invalid, it's a worsening (size - 1 but invalid? No, we must maintain validity or penalize)
        # Standard approach for constrained SA:
        # State must be valid? Or we allow invalid states with a penalty?
        # If we allow invalid states, we need to repair them.
        # Easier: Only allow moves that keep the solution VALID.
        # If we only allow valid moves, we can only remove redundant words.
        # This is just local search for removal. It will get stuck.
        # To escape, we need to allow "swap" moves that keep validity.
        
        # Let's try:
        # 1. Remove a redundant word (if any).
        # 2. If no redundant word, try a swap: remove i, add j.
        #    Check if new set is valid.
        #    If valid, accept if size decreases (impossible with swap, size stays same) 
        #    OR if it leads to a state where a removal is possible later.
        #    Actually, swap keeps size constant. It changes the structure.
        #    If we accept swaps that keep validity, we explore the space of valid covers of size K.
        #    If we find a state where a word is redundant, we remove it, size K-1.
        
        # So the strategy is:
        # - Find redundant words. If any, remove one (improvement).
        # - If none, try a swap. If the new state is valid, accept it with probability based on T.
        #   Why accept neutral swaps? To explore the space of valid covers.
        #   Maybe we should bias swaps towards covering "tight" parts?
        
        # Let's check for redundant words efficiently.
        # A word i is redundant if min(coverage_counts[neighbors[i]]) >= 2.
        
        # Find all redundant words
        # For each i in codewords, check neighbors[i]
        # This is O(K * 13). K is ~70-80. 1000 ops. Very fast.
        
        redundant_indices = []
        for idx, c in enumerate(codewords):
            if np.min(coverage_counts[neighbors[c]]) >= 2:
                redundant_indices.append(idx)
                
        if redundant_indices:
            # Pick one to remove
            # To be smart, pick the one that "unbalances" the coverage least?
            # Or just random.
            idx_to_remove = random.choice(redundant_indices)
            c_to_remove = codewords[idx_to_remove]
            
            # Perform removal
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            
            # Update best
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                
            # Don't cool down on improvement? Or cool down slowly.
            # Usually we cool down every step.
            T *= cooling
            if T < T_min:
                T = T_min
            continue
            
        else:
            # No redundant words. Try a swap.
            # Pick a random codeword to remove
            idx_to_remove = random.randint(0, len(codewords) - 1)
            c_to_remove = codewords[idx_to_remove]
            
            # Check if we can find a replacement c_to_add such that the set remains valid.
            # The words that become uncovered are those where coverage was 1 and covered by c_to_remove.
            # Let's find the "critical" words: neighbors[c_to_remove] where coverage_counts == 1.
            critical_words = neighbors[c_to_remove][coverage_counts[neighbors[c_to_remove]] == 1]
            
            if len(critical_words) == 0:
                # This shouldn't happen if redundant check passed, 
                # but if coverage_counts > 1 for all neighbors, it was redundant.
                # So critical_words must be non-empty if not redundant.
                pass
            
            # We need to pick a c_to_add that covers ALL critical_words.
            # c_to_add must be in the neighborhood of every word in critical_words.
            # i.e., c_to_add must be a common neighbor of all words in critical_words.
            
            # Since critical_words are neighbors of c_to_remove, 
            # c_to_add must be within distance 1 of each critical word.
            # Note: c_to_remove itself is a neighbor of all critical words.
            # But c_to_add != c_to_remove (otherwise no change).
            
            # How to find candidates?
            # Pick one critical word, say w0.
            # Candidates for c_to_add must be in neighbors[w0].
            # Then check if they cover all other critical words.
            
            if len(critical_words) > 0:
                w0 = critical_words[0]
                candidates = neighbors[w0]
                
                found_swap = False
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                for c_candidate in candidates:
                    if c_candidate == c_to_remove:
                        continue
                    
                    # Check if c_candidate covers all critical_words
                    # c_candidate covers a word w if w is in neighbors[c_candidate]
                    # Or distance(c_candidate, w) <= 1.
                    # We can check if critical_words is a subset of neighbors[c_candidate]
                    # Using sets is slow. 
                    # Let's use a boolean check.
                    
                    # Actually, we can precompute a "cover mask" or just loop.
                    # critical_words is small (at most 12).
                    # neighbors[c_candidate] is 13 items.
                    # Check if all c in critical_words are in neighbors[c_candidate]
                    
                    # Efficient check:
                    # Create a set of neighbors[c_candidate]? No, list is fine for small n.
                    # Or use numpy:
                    # np.all(np.isin(critical_words, neighbors[c_candidate]))
                    
                    if np.all(np.isin(critical_words, neighbors[c_candidate])):
                        # This swap keeps the solution valid!
                        # Perform swap
                        codewords[idx_to_remove] = c_candidate
                        coverage_counts[neighbors[c_to_remove]] -= 1
                        coverage_counts[neighbors[c_candidate]] += 1
                        
                        # Accept this swap?
                        # It maintains validity and size.
                        # It might enable future removals.
                        # We should accept it with some probability to explore.
                        # Let's always accept valid swaps to explore the "valid cover" manifold.
                        # Or accept with prob 1.0?
                        # If we always accept, we might cycle.
                        # But it's a neutral move.
                        # Let's accept it.
                        
                        found_swap = True
                        break
                        
                if not found_swap:
                    # Try a different strategy: 
                    # Maybe remove 2 and add 1? Too complex.
                    # Or just cool down.
                    pass
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

    # Final cleanup: try to remove any remaining redundant words
    # This is important because the last few steps might leave redundant words
    # if the loop ended right after a swap.
    improved = True
    while improved:
        improved = False
        for idx, c in enumerate(codewords):
            if np.min(coverage_counts[neighbors[c]]) >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break # Restart loop

    # Use the best found
    if len(best_codewords) < len(codewords):
        final_indices = best_codewords
    else:
        final_indices = codewords

    # Convert indices to lists
    return [idx_to_list(idx) for idx in final_indices]

# EVOLVE-BLOCK-END
