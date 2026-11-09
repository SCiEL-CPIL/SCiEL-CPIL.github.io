import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 52.0  # Safety margin below 60s
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Start with the first-order Reed-Muller code (32 words)
    # First-order Reed-Muller RM(1,4): 16 coordinates, 32 words
    # We extend to 18 bits by adding 2 more coordinates
    # Actually, let's use RM(1,4) on 16 bits, then pad to 18 bits
    # But better: use RM(1,4) directly on 16 coordinates, and we can try to extend
    
    # Build RM(1,4) on 16 coordinates
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    # Pad to 18 bits (add 2 zero bits) - this gives distance 8
    words = [w << 0 for w in rm_words]  # 16-bit words, treat as 18-bit with zeros in top
    
    # Actually, let me be more careful. The words are 16-bit. 
    # I'll store them as 18-bit integers (they already fit since max is 2^16-1)
    
    best_words = list(words)
    best_size = len(best_words)
    
    # Try greedy extension: add more words one by one
    # Use a candidate pool and greedily add words that maintain min distance
    
    def can_add(candidate, current_words):
        for w in current_words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    # Generate candidate words and try to add them
    # Strategy: iterate through all 2^18 words in a smart order
    # Sort by weight to find good candidates
    
    # Pre-compute: we want to find words at distance >= 6 from all current words
    # Use a set for fast lookup
    word_set = set(best_words)
    
    # Try adding random and structured candidates
    rng = random.Random(42)
    
    # First, try to extend with structured additions
    # The RM(1,4) code has 32 words. We can try to add second-order functions
    # RM(2,4) has 16+32+... = 2^(1+4)+2^4 = 256 words? No, RM(2,4) is the quadratic code
    # Actually RM(2,4) has dimension 1+4+6 = 11, so 2048 words
    
    # Let me just do greedy search over all 2^18 words
    # But that's 262144 words, checking each against up to many existing words
    # We need to be efficient
    
    # Use a different approach: for each candidate, check against all current words
    # But limit the number of candidates we check
    
    # Let's try a smarter approach: 
    # 1) Start with RM(1,4) extended
    # 2) Greedily add the best candidates
    
    # Generate candidates in order of increasing weight (or some heuristic)
    # Actually, let's just iterate through all possible 18-bit words
    # and greedily add if valid
    
    # To speed up, use a bitset or just iterate
    # 2^18 = 262144 candidates, each checked against ~32+ words initially
    # That's about 262144 * 32 = ~8.4 million distance checks, each is XOR + popcount
    # This should be fast enough in Python if we're careful
    
    # Let's do it
    current_words = list(best_words)
    current_set = set(current_words)
    
    # Try all 2^18 words
    # But that might take too long. Let's limit.
    
    # Actually, let's try a few passes:
    # Pass 1: Try all words 0 to 2^18-1, add if valid
    # This is O(2^18 * K) where K is code size
    
    # Let's estimate: 2^18 * 32 * (XOR + popcount) 
    # In Python, this might be ~10-30 seconds. Let's try with a deadline check.
    
    added_any = True
    while added_any and time.time() < deadline - 5.0:
        added_any = False
        for candidate in range(1 << N):
            if candidate in current_set:
                continue
            if can_add(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                added_any = True
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
    
    # Try some random restarts if time permits
    if time.time() < deadline - 3.0:
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            # Try a different initial set
            rng_words = [rng.randint(0, (1 << N) - 1) for _ in range(8)]
            # Remove duplicates
            rng_words = list(set(rng_words))
            if len(rng_words) < 2:
                continue
            
            # Greedily extend this random seed
            greedy_words = list(rng_words)
            greedy_set = set(greedy_words)
            
            # Verify the seed is valid
            valid = True
            for i in range(len(greedy_words)):
                for j in range(i+1, len(greedy_words)):
                    if (greedy_words[i] ^ greedy_words[j]).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Now greedily add from all candidates
            for candidate in range(1 << N):
                if candidate in greedy_set:
                    continue
                if can_add(candidate, greedy_words):
                    greedy_words.append(candidate)
                    greedy_set.add(candidate)
            
            if len(greedy_words) > best_size:
                best_words = list(greedy_words)
                best_size = len(greedy_words)
    
    return best_words
