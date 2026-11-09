import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit

    N = 17
    MIN_DIST = 6
    
    # 1. Generate the 256-word linear code
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    # Base vector indices where bit is 1: Q U {0}
    ones_indices = list(Q | {0})
    
    # Generator matrix rows (9 rows)
    # Row i is cyclic shift of base vector by i
    # We represent each row as an integer (17 bits)
    gen_rows = []
    base = 0
    for idx in ones_indices:
        base |= (1 << idx)
        
    for i in range(9):
        # Cyclic shift right by i (or left, doesn't matter as long as consistent)
        # (base >> i) | (base << (17-i)) & ((1<<17)-1)
        shifted = (base >> i) | ((base << (17 - i)) & ((1 << 17) - 1))
        gen_rows.append(shifted)
        
    # Generate all 512 combinations
    code_words = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= gen_rows[i]
        
        # Calculate parity
        parity = word.bit_count() % 2
        if parity == 0:
            code_words.append(word)
            
    # This should give 256 words
    current_code = set(code_words)
    
    # 2. Try to expand greedily
    # We want to add words that are at distance >= 6 from all in current_code
    # Candidate pool: We can try random words, or words of specific weights.
    # Distance 6 in 17 bits.
    # Weight 6 words are "close" to each other?
    # If we have a word of weight 6, another word of weight 6 can be at distance 6 if they share 3 bits?
    # d(x,y) = wt(x) + wt(y) - 2*wt(x&y) = 6+6-2*3 = 6.
    # So words of weight 6 are good candidates if they are far apart.
    
    # Pre-compute bit counts for speed?
    # We can just use .bit_count()
    
    # Helper to check if a word is valid
    def is_valid(word):
        for w in current_code:
            if (word ^ w).bit_count() < MIN_DIST:
                return False
        return True

    # Generate candidates
    # We can iterate through all 2^17 words? 131072 words.
    # Checking each against 256 existing words: 131072 * 256 = 33M operations.
    # In Python, this might be too slow (33M * ~100ns = 3.3s).
    # Actually, bit_count is fast. 33M iterations in pure Python might take 10-20s.
    # Let's try to optimize.
    
    # We can break early in is_valid.
    # Also, we can sort candidates by weight or something?
    
    # Let's try a random sampling approach first, then full scan if time permits.
    # But full scan is deterministic and finds ALL possible additions.
    
    # Optimization: Use NumPy for vectorized distance checks?
    # Convert current_code to a NumPy array of ints?
    # XOR and bit_count on arrays is not directly vectorized in NumPy for ints easily without loops or specific tricks.
    # However, we can convert to binary matrix? 256x17 matrix.
    # Candidate: 17 vector.
    # Distance: sum of XOR.
    # This is fast in NumPy.
    
    code_arr = np.array(list(current_code), dtype=np.uint32)
    
    # Function to check a batch of candidates
    def check_candidates(candidates):
        # candidates: list of ints
        # Return list of valid candidates
        valid = []
        if not candidates:
            return valid
        
        # Convert candidates to array
        cand_arr = np.array(candidates, dtype=np.uint32)
        
        # We need d(c, w) >= 6 for all w in code_arr
        # d(c, w) = bit_count(c ^ w)
        
        # We can loop through candidates and check against code_arr
        # Or loop through code_arr and eliminate candidates?
        
        # Let's just loop through candidates.
        # For each candidate, compute distance to all code words.
        # If min distance >= 6, it's valid.
        
        for c in candidates:
            # Compute XOR with all code words
            # This creates an array of 256 ints
            # Then bit_count on each
            # This is 256 operations per candidate.
            # If we have 1000 candidates, 256000 ops. Fast.
            
            # Let's do it in a batch for speed?
            # Actually, pure Python loop with bit_count is quite fast.
            # Let's stick to Python for simplicity and use early exit.
            
            for w in current_code:
                if (c ^ w).bit_count() < MIN_DIST:
                    break
            else:
                valid.append(c)
                
        return valid

    # Generate all possible words? 131072.
    # Let's split into chunks to monitor time.
    all_words = range(2**N)
    
    # We already have the linear code words.
    # We should not re-add them.
    # is_valid will return False for them anyway (distance 0).
    
    # To speed up, we can skip words with odd weight?
    # No, distance 6 is even. If x has even weight and y has even weight, distance is even.
    # If x has even weight and y has odd weight, distance is odd.
    # Our code words have even weight (parity 0).
    # So we only need to consider candidates with even weight.
    # This halves the search space to 65536.
    
    # Let's iterate through even weight words.
    # We can generate them or just filter.
    
    # Let's try to add as many as possible.
    # We will iterate through all even weight words.
    # We will check them against current_code.
    
    # To make it faster, we can update current_code as we find valid words.
    # This makes subsequent checks harder (more words to check against) but ensures maximality.
    
    # However, checking 65k words against a growing set (up to ~300) might be slow.
    # 65k * 300 = 19.5M distance checks.
    # Each check is (c^w).bit_count().
    # In Python, 19.5M iterations might take ~5-10 seconds.
    # This is acceptable.
    
    # Let's implement the loop with time checks.
    
    added = 0
    # Iterate through all 17-bit integers
    # We can just iterate 0 to 2^17-1 and skip odd weight
    for c in range(2**N):
        if time.time() > deadline:
            break
            
        if c.bit_count() % 2 != 0:
            continue
            
        # Check against current_code
        valid = True
        for w in current_code:
            if (c ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            current_code.add(c)
            added += 1
            
    return list(current_code)
