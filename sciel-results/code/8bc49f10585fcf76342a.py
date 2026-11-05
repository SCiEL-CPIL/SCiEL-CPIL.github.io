import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Base solution: Direct sum of ternary tetracode and full space of remaining coords
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate the 81 codewords
    base_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            base_codewords.append(h + list(tail))
    
    # If n < 4, we can't use tetracode directly. 
    # However, the task specifies n=6. 
    # For robustness, if n is small, we might need a different base, but let's assume n>=4.
    # If n < 4, the direct sum approach fails. 
    # Given the task is specifically about n=6, we proceed with the tetracode base.
    
    if n < 4:
        # Fallback for small n (not expected for this specific task instance)
        # Just return a simple greedy or all words if very small
        # For n=1, q=3, r=1, we need 1 word? No, distance 1 covers everything in n=1.
        # Actually, for n=1, any single word covers all 3 words.
        # Let's just return the first word if n==1.
        # But the problem is fixed to n=6.
        pass

    # Convert codewords to integer representation for faster processing
    # Word w -> int(w[0]*3^5 + w[1]*3^4 + ... + w[5])
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute neighborhoods for all possible words? 
    # Space is 3^6 = 729. This is small.
    # We only need neighborhoods for the codewords we have, but removing them might require checking coverage.
    # It is more efficient to maintain a coverage count array.
    
    all_words = list(range(q ** n))
    
    # Map word int to list of neighbors (including self) within radius 1
    # Since q=3, n=6, radius=1, neighbor count is 1 + n*(q-1) = 1 + 6*2 = 13.
    
    def get_neighbors(word_int):
        word = to_word(word_int)
        neighbors = [word_int]
        for i in range(n):
            orig = word[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word[i] = new_val
                neighbors.append(to_int(word))
            word[i] = orig
        return neighbors

    # Initialize coverage counts
    coverage = [0] * (q ** n)
    active_codewords = [to_int(c) for c in base_codewords]
    
    # We need to know which codeword covers which word to update counts efficiently?
    # Or just recompute? 
    # With 81 codewords and 13 neighbors each, total updates ~ 1000.
    # Removing one codeword requires checking if any of its 13 neighbors have coverage 1.
    # If yes, it's not removable. If no, remove it and decrement counts.
    
    # To make removal efficient, we should track for each word, how many codewords cover it.
    # We also need to be able to quickly check if a codeword is redundant.
    # A codeword c is redundant if for all w in N(c), coverage[w] > 1.
    
    # Let's initialize coverage
    for c in active_codewords:
        for w in get_neighbors(c):
            coverage[w] += 1
            
    # Greedy peeling
    # We iterate through the list of active codewords.
    # If a codeword is redundant, we remove it.
    # We might need multiple passes or a queue of "potentially redundant" codewords.
    # Since removing a codeword decreases coverage, it might make *other* codewords redundant.
    # So we should use a queue or just loop until no more changes.
    
    # Optimization: Use a set for active codewords for O(1) removal? 
    # But we need to iterate. A list is fine for 81 items.
    
    changed = True
    while changed:
        changed = False
        current_time = time.time()
        if current_time > deadline:
            break
            
        # Iterate over a copy of the list to allow modification
        # Or iterate by index and skip if removed.
        # Let's use a set of active indices or just a list and filter.
        # Filtering 81 items is fast.
        
        new_active = []
        for c in active_codewords:
            # Check if c is redundant
            redundant = True
            for w in get_neighbors(c):
                if coverage[w] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                for w in get_neighbors(c):
                    coverage[w] -= 1
                changed = True
            else:
                new_active.append(c)
                
        active_codewords = new_active
        
        # If we removed any, loop again because removal might make others redundant
        # The 'changed' flag handles this.
        
    # Convert back to list of lists
    final_codewords = [to_word(c) for c in active_codewords]
    
    # Ensure distinctness (should be guaranteed by construction and removal logic)
    # But just in case, deduplicate
    seen = set()
    unique_codewords = []
    for c in final_codewords:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_codewords.append(c)
            
    return unique_codewords
