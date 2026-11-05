import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Seed with the known 32-word RM(1,4) code (extended to 17 bits with a 0)
    # This ensures we never score below 1.0 (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The 17th bit is 0
            words.append(word)
    
    # Helper to check if a candidate word is valid (distance >= 6 from all existing)
    def is_valid(candidate):
        for w in words:
            if (candidate ^ w).bit_count() < 6:
                return False
        return True

    # 2. Greedy Growth Phase
    # Try to add random words that satisfy the distance constraint.
    # We iterate through random candidates. If valid, add them.
    # To speed this up, we can use a simple random strategy.
    # Since the space is large (2^17), random sampling might be slow to find valid words
    # as the code grows, but it works for the first few additions.
    
    # Better: Use a "nearest neighbor" or "complement" heuristic?
    # Actually, for small distances like 6, random sampling is often effective initially.
    # Let's try to add as many as possible via random sampling.
    
    random.seed(42)
    added = 0
    while time.time() < deadline - 10.0:  # Leave 10s for hill climbing
        candidate = random.getrandbits(17)
        if is_valid(candidate):
            words.append(candidate)
            added += 1
            # If we've added a lot, the space is getting crowded.
            # We might need to switch strategies or just keep going.
            # The theoretical max is ~258. We start with 32.
            # Random sampling probability drops as 1 - (1 - 2^-17 * volume)^(n)
            # With n=32, it's still high. With n=100, it's lower.
            
    # 3. Hill Climbing / Local Search Phase
    # Try to improve the set by swapping words or moving words to better positions.
    # A simple heuristic: Pick a random word in the set, try to move it to a nearby
    # position (flip 1 or 2 bits) that is still valid and doesn't reduce the "potential"
    # for future additions. 
    # However, a simpler and more effective move for maximizing size is to try to
    # find a word that is "more central" or allows more space.
    # Alternatively, just try to add more words using a smarter search.
    
    # Let's try a different approach: 
    # If we have time, try to find more words by checking candidates generated from
    # existing words by flipping bits.
    
    # Actually, the most effective simple heuristic for binary codes is often
    # just random addition, but it stalls. 
    # Let's try to "shake" the code: 
    # Pick a random word, remove it, and try to re-insert it or a neighbor.
    # This is complex to implement correctly within time limits without a robust
    # backtracking solver.
    
    # Given the constraints (60s), and that random addition from 32 to ~50-100 is
    # feasible, let's stick to a robust random addition loop with a tighter
    # inner check.
    
    # Optimization: Use a set for O(1) lookups? No, we need distance checks.
    # The distance check is O(N) per candidate.
    
    # Let's refine the random search to be slightly smarter:
    # Generate candidates by flipping 1 bit of an existing word.
    # This explores the local neighborhood.
    
    base_words = list(words)
    for _ in range(1000): # Limit iterations to avoid infinite loops if stuck
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random existing word
        if not words:
            continue
        base_idx = random.randint(0, len(words) - 1)
        base = words[base_idx]
        
        # Flip a random bit
        bit_pos = random.randint(0, 16)
        candidate = base ^ (1 << bit_pos)
        
        if is_valid(candidate):
            words.append(candidate)
            
    # Final check: ensure all pairs are valid (sanity check, though is_valid ensures it)
    # The function returns the list of words.
    return words
