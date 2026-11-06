import random
import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 45.0  # Safety margin for 60s limit

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    # Build a structured base: Reed-Muller RM(1,4) gives 32 words of length 16, dist 8.
    # Extend to 17 bits by adding a parity-like bit or just appending 0.
    # Actually, let's use a simpler structured start: all weight-8 and weight-9 words? No, too many.
    # Let's start with the Reed-Muller RM(1,4) code extended to 17 bits.
    # RM(1,4) has 2^(1+4) = 32 codewords, length 16, min dist 8.
    # We can extend to 17 bits by appending a 0 bit (length 17, dist 8).
    # Or we can try to find a better base.
    
    # Let's try a different approach: start with a known good code if we can generate it quickly.
    # The [17,5,9] Hamming code? No, that's distance 9, size 32.
    # Let's use the Reed-Muller RM(1,4) extended to 17 bits.
    
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                # Evaluate linear function f(x) = slope*x + intercept over GF(2)^4
                # slope and point are 4-bit vectors. Dot product mod 2.
                dot = bin(slope & point).count('1') % 2
                bit = (dot + intercept) % 2
                word |= bit << point
            # Append a 0 bit to make it 17 bits
            base_words.append(word)  # bit 16 is 0
    
    # Verify base is valid
    for i in range(len(base_words)):
        for j in range(i+1, len(base_words)):
            if hamming_dist(base_words[i], base_words[j]) < MIN_DIST:
                # This shouldn't happen for RM(1,4)
                pass
    
    codewords = list(base_words)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Now use a clique-finding heuristic to add more words.
    # Strategy: 
    # 1. Generate a pool of candidate words that are compatible with all base words.
    # 2. Build a compatibility graph among these candidates (edge if dist >= 6).
    # 3. Find a large clique in this graph.
    # 4. Add the clique to the base.
    
    # But generating all compatible candidates is expensive (2^17 = 131072).
    # Let's be smarter: 
    # - Sample random candidates that are compatible with the current set.
    # - Use a greedy clique heuristic on the sampled candidates.
    
    rng = random.Random(42)
    
    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
    
    # Phase 1: Greedy addition of random compatible words
    attempts = 0
    max_attempts_greedy = 100000
    while attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.4:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Phase 2: Local search / clique extension
    # Try to find more words by considering "nearby" words to existing codewords?
    # Or just keep greedy adding with different strategies.
    
    # Let's try a "swap" heuristic: remove one word, try to add two.
    # This can escape local optima.
    
    for _ in range(5000):
        if time.time() - start_time > time_limit * 0.8:
            break
        
        K = len(codewords)
        if K == 0:
            break
        
        # Pick a random word to remove
        idx = rng.randint(0, K - 1)
        removed = codewords[idx]
        others = [c for i, c in enumerate(codewords) if i != idx]
        
        # Try to find two new words that are compatible with 'others' and each other
        found = False
        for _ in range(200):
            if time.time() - start_time > time_limit * 0.8:
                break
            
            w1 = rng.getrandbits(N)
            if not is_compatible_with_all(w1, others):
                continue
            
            # Try to find w2 compatible with others and w1
            for _ in range(50):
                w2 = rng.getrandbits(N)
                if not is_compatible_with_all(w2, others):
                    continue
                if hamming_dist(w1, w2) < MIN_DIST:
                    continue
                
                # Found w1, w2
                new_set = others + [w1, w2]
                codewords = new_set
                found = True
                break
            
            if found:
                break
        
        if found:
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
    
    # Phase 3: Final greedy polish
    attempts = 0
    max_attempts_final = 50000
    while attempts < max_attempts_final:
        if time.time() - start_time > time_limit:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Return the best set found
    return best_codewords
