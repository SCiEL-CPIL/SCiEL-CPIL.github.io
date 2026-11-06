import time
import random

def construct():
    # Deadline guard: 85 seconds to be safe
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 18
    MIN_DIST = 6
    
    # 1. Initial Reed-Muller Code (32 words)
    # First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # Actually, the seed code is RM(1,4) on 16 bits + 2 zeros?
    # The seed code in the prompt:
    # for slope in range(16): for intercept in range(2):
    #   word = 0
    #   for point in range(16):
    #      bit = ((slope & point).bit_count() + intercept) % 2
    #      word |= bit << point
    #   words.append(word)
    # This creates 32 words of length 16. The prompt says "extended by two zero bits".
    # The returned words are in [0, 2**18). The code above only sets bits 0-15.
    # So bits 16 and 17 are 0.
    
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
            
    # Ensure distinctness and validity
    words = list(set(words))
    
    # 2. Greedy Expansion
    # We want to add random words that have distance >= 6 from all existing words.
    # To speed up, we can check candidates in batches or use a smarter sampling.
    # 18-bit space is 262,144. Checking one candidate against 512 words takes 512 XOR+popcount.
    # 262k * 512 ops is ~134M ops. In Python, this is too slow for a full scan.
    # We will use random sampling.
    
    # Optimization: Use a list for current words.
    # If the list grows large, checking becomes expensive.
    # We can stop adding if we hit the theoretical max 512 or if time runs out.
    
    # To improve success rate, we can try to find "holes" in the code.
    # A simple random greedy might get stuck.
    # We will perform multiple passes.
    
    # Pre-generate a large set of random candidates to avoid random() overhead in the inner loop?
    # Or just use random.getrandbits.
    
    # Let's use a "best first" or "random" approach.
    # Given the 6.0 score from simple greedy, we aim for more.
    
    # Strategy: 
    # - Keep the current set.
    # - Generate random candidates.
    # - Check if valid.
    # - If valid, add.
    # - If we stall (no new words found in X attempts), try a different strategy or restart?
    # Restarting loses the current progress. Better to just keep sampling.
    
    # To be faster, we can sort the current words or use a bitset? 
    # Python integers are fast for XOR.
    
    # Let's add a timeout check inside the loop.
    
    max_attempts = 1000000  # Cap on total random tries to prevent infinite loop if time check fails
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        
        # Generate a random 18-bit word
        candidate = random.getrandbits(18)
        
        # Check if candidate is already in words
        # Using a set for O(1) lookup
        if candidate in words:
            attempts += 1
            continue
            
        # Check distance to all existing words
        valid = True
        for w in words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            words.append(candidate)
            # If we reached the known optimum, we can stop
            if len(words) >= 512:
                break
        else:
            attempts += 1
            
    return words
