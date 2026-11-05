import numpy as np
import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code for q=3, n=8, radius=1.
    Starts from the 486-word Tetracode product construction and greedily removes redundant codewords.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task specifies 3, 8, 1
        # Simple greedy construction for general case
        # But we focus on the specific case for performance
        # For now, return empty or raise, but task says "evaluator passes this task's..."
        # Assuming the evaluator only calls with 3, 8, 1 based on the task description.
        # If not, we should handle it gracefully.
        # Let's just return a trivial solution if parameters don't match to avoid errors,
        # or better, implement a basic greedy for any q, n, r if time permits.
        # However, the prompt says "evaluator passes this task's alphabet size...".
        # I will assume q=3, n=8, radius=1.
        pass

    # Generate the initial 486 codewords
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Product with all 81 words of length 4
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            initial_codewords.append(h + list(tail))
            
    codewords = initial_codewords
    num_words = 3 ** 8 # 6561
    
    # Map codeword to its index in the 6561 space for fast lookup
    # A codeword c = [c0, c1, ..., c7] maps to idx = c0*3^7 + c1*3^6 + ... + c7
    powers = [3 ** i for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[n - 1 - i]
        return idx
        
    # Precompute neighbors for each possible word? 
    # Or just compute on the fly?
    # There are 6561 words. Each has 1 + 8*2 = 17 neighbors (self + dist 1).
    # Total entries in neighbor list: 6561 * 17 ~ 111k. This is small.
    
    # Let's precompute the list of indices covered by each possible word index.
    # covered_by[i] = list of indices j such that dist(i, j) <= 1
    covered_by = [[] for _ in range(num_words)]
    
    # We need to map indices back to words to compute distances?
    # Or compute neighbors directly in index space?
    # It's easier to generate neighbors in word space then map to index.
    
    idx_to_word = [[0]*n for _ in range(num_words)]
    for i in range(num_words):
        temp = i
        for j in range(n-1, -1, -1):
            idx_to_word[i][j] = temp % 3
            temp //= 3
            
    for i in range(num_words):
        word = idx_to_word[i]
        # Self
        covered_by[i].append(i)
        # Neighbors
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_word = word.copy()
                    new_word[pos] = new_val
                    covered_by[i].append(word_to_idx(new_word))
                    
    # Initialize coverage counts
    coverage = np.zeros(num_words, dtype=np.int32)
    
    # Add all initial codewords to coverage
    for c in codewords:
        idx = word_to_idx(c)
        for neighbor_idx in covered_by[idx]:
            coverage[neighbor_idx] += 1
            
    # Greedy removal
    # We maintain a list of current codewords.
    # We repeatedly scan for removable codewords.
    # A codeword c at index idx is removable if coverage[j] > 1 for all j in covered_by[idx].
    
    # Optimization: Instead of scanning all codewords every time, 
    # we can just loop through the list. If a removal happens, we restart the scan 
    # or continue with the modified list.
    # Since 486 is small, a simple loop is fine.
    
    # To be efficient, we can check codewords in an order that might find removals faster.
    # But simple iteration is O(N * 17) per pass.
    
    while True:
        if time.time() > deadline:
            break
            
        removed_any = False
        # We iterate over a copy or indices because we modify the list
        # Actually, we can just iterate and skip if already removed?
        # Better: iterate over the current list.
        
        # We want to remove as many as possible.
        # Let's try to find one removal, then restart? 
        # Or find all removable in one pass?
        # If we remove c1, c2 might become non-removable.
        # So it's safer to find one, remove it, and restart the scan from the beginning?
        # Or just scan and if we remove one, break and restart the while loop?
        # This ensures we always check against the current state.
        
        # Let's scan for the first removable codeword.
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
                
            c = codewords[i]
            idx = word_to_idx(c)
            
            # Check if all neighbors have coverage > 1
            # This is a quick check
            removable = True
            for neighbor_idx in covered_by[idx]:
                if coverage[neighbor_idx] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove this codeword
                # Decrement coverage
                for neighbor_idx in covered_by[idx]:
                    coverage[neighbor_idx] -= 1
                
                # Remove from list
                codewords.pop(i)
                removed_any = True
                break # Restart the scan from the beginning to catch new redundancies
        
        if not removed_any:
            break
            
    # The result is the list of codewords
    return codewords
