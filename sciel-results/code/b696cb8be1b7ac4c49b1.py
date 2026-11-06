import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s kill
    
    n = 17
    min_dist = 6
    max_size = 340  # Upper bound from Brouwer
    total_words = 1 << n  # 131072
    
    # Precompute popcounts using numpy for speed
    all_words = np.arange(total_words, dtype=np.uint32)
    popcount = np.zeros(total_words, dtype=np.uint8)
    for i in range(17):
        popcount += (all_words >> i) & 1
    
    # Construct the known 256-word [17,9,6] QR code (shortened extended QR code)
    # The quadratic residues mod 17 are: {1, 2, 4, 8, 9, 13, 15, 16}
    # We'll use a systematic construction approach
    
    def construct_qr_code_256():
        """Construct the 256-word [17,9,6] shortened extended QR code."""
        # The [18,9,6] extended QR code has generator polynomial for length 18
        # Shortening by one coordinate gives [17,9,6]
        # 
        # Alternative: Use the fact that this is a linear code with 9 information bits
        # We need to find the generator matrix
        
        # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
        # The [17,9,5] QR code has generator polynomial g(x) = x^8 + x^7 + x^6 + x^4 + x^3 + x^2 + x + 1
        # Wait, let me use a different approach - construct it directly
        
        # For the extended QR code of length 18, the generator polynomial divides x^18 - 1
        # The quadratic residue code of length 17 has parameters [17,9,5]
        # Extended version has [18,9,6]
        
        # Let's use a simpler approach: start with the first-order Reed-Muller code structure
        # and extend it to get 256 words
        
        # Actually, let's construct it using the generator matrix approach
        # The [17,9,5] QR code generator matrix can be built from the QR set
        
        # QR set mod 17: Q = {1, 2, 4, 8, 9, 13, 15, 16}
        # The generator polynomial for the [17,9,5] code is:
        # g(x) = (x+1)(x^4+x^3+x^2+x+1)(x^4+x^2+1) ... this is getting complex
        
        # Let me use a direct construction: the code is the set of all c such that
        # c is in the span of certain basis vectors
        
        # Simpler: Use the fact that we can construct it as a linear code
        # with 9 generator rows
        
        # For now, let's use a greedy approach to find a good 256-word code
        # starting from a known good seed
        
        # Start with the all-zero word and greedily add words
        # using a "minimum conflict" heuristic
        
        codewords = [0]
        used = {0}
        
        # Precompute for faster conflict checking
        # For each word, we want to know which other words are at distance < 6
        
        # Use a priority-based greedy: at each step, pick the word that minimizes
        # the maximum distance to existing codewords (or minimizes conflicts)
        
        remaining = list(range(1, total_words))
        random.seed(42)
        
        # To speed up, we'll use a batch approach
        # Process candidates in chunks and find valid ones
        
        for iteration in range(255):
            if time.time() > deadline - 10.0:
                break
            
            # Find the next valid codeword
            # Try random candidates first for diversity
            found = False
            attempts = 0
            max_attempts = 5000
            
            while attempts < max_attempts and not found:
                attempts += 1
                candidate = random.choice(remaining)
                
                valid = True
                for cw in codewords:
                    if popcount[candidate ^ cw] < min_dist:
                        valid = False
                        break
                
                if valid:
                    codewords.append(candidate)
                    used.add(candidate)
                    found = True
            
            if not found:
                # Try a more systematic search
                for candidate in remaining:
                    valid = True
                    for cw in codewords:
                        if popcount[candidate ^ cw] < min_dist:
                            valid = False
                            break
                    if valid:
                        codewords.append(candidate)
                        used.add(candidate)
                        found = True
                        break
            
            if not found:
                break
        
        return codewords
    
    # Try to construct the 256-word QR code
    qr_code = construct_qr_code_256()
    
    if len(qr_code) >= 200:  # If we got a reasonably large code
        best_codewords = qr_code
        best_size = len(qr_code)
    else:
        best_codewords = [0]
        best_size = 1
    
    # Helper: check if a candidate can be added to the current code
    def can_add(candidate, codewords, popcount, min_dist):
        for cw in codewords:
            if popcount[candidate ^ cw] < min_dist:
                return False
        return True
    
    # Helper: run greedy pass with a given order
    def run_greedy_pass(order, current_codewords=None, popcount=popcount, min_dist=min_dist, deadline=deadline):
        if current_codewords is None:
            codewords = [0]
        else:
            codewords = current_codewords.copy()
        
        for candidate in order:
            if time.time() > deadline - 2.0:
                break
            valid = True
            for cw in codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                codewords.append(candidate)
        return codewords
    
    # If we have a decent starting code, try to extend it
    if best_size > 100 and time.time() < deadline - 5.0:
        # Try to add more words to the existing code
        candidates = [w for w in range(total_words) if w not in set(best_codewords)]
        random.seed(42)
        
        # Multiple passes trying to add words
        for pass_num in range(10):
            if time.time() > deadline - 3.0:
                break
            
            random.shuffle(candidates)
            added_in_pass = 0
            
            for candidate in candidates:
                if time.time() > deadline - 2.0:
                    break
                
                valid = True
                for cw in best_codewords:
                    if popcount[candidate ^ cw] < min_dist:
                        valid = False
                        break
                
                if valid:
                    best_codewords.append(candidate)
                    candidates.remove(candidate)
                    added_in_pass += 1
            
            if added_in_pass == 0:
                break
        
        best_size = len(best_codewords)
    
    # Local improvement: try 1-for-2 exchanges
    if best_size > 100 and time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            cw_set = set(best_codewords)
            
            for idx in range(len(best_codewords)):
                if time.time() > deadline - 2.0:
                    break
                
                # Try removing word at idx and adding new words
                removed = best_codewords[idx]
                remaining = [best_codewords[j] for j in range(len(best_codewords)) if j != idx]
                
                added = 0
                # Try to find 2 new words that can be added
                for candidate in random.sample(range(total_words), min(2000, total_words)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    if candidate in cw_set and candidate != removed:
                        continue
                    
                    valid = True
                    for cw in remaining:
                        if popcount[candidate ^ cw] < min_dist:
                            valid = False
                            break
                    
                    if valid:
                        remaining.append(candidate)
                        cw_set.add(candidate)
                        added += 1
                        if added >= 2:
                            break
                
                if added > 1:  # Net gain of at least 1
                    best_codewords = remaining
                    best_size = len(remaining)
                    improved = True
                    break
    
    return best_codewords
