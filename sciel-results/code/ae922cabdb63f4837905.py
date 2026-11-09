import random
import time

# Pre-compute bit counts for 17-bit integers
BIT_COUNTS = [0] * (1 << 17)
for i in range(1, 1 << 17):
    BIT_COUNTS[i] = BIT_COUNTS[i >> 1] + (i & 1)

def construct():
    N = 17
    MIN_DIST = 6
    ALL_WORDS = 1 << N
    
    # Generate the 32-word seed
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
    seed_set = set(seed_words)
    
    # List of all words
    all_words = list(range(ALL_WORDS))
    
    best_code = seed_words[:]
    best_size = len(best_code)
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Strategies for ordering
    def sort_weight(w): return (w.bit_count(), random.random())
    def sort_center(w): return (abs(w.bit_count() - 8), random.random())
    def sort_random(w): return random.random()
    
    strategies = [sort_weight, sort_center, sort_random]
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        sort_key = strategies[restarts % 3]
        
        # Create a shuffled copy of candidates to avoid bias
        # We exclude seed words from the candidate list for greedy addition
        candidates = [w for w in all_words if w not in seed_set]
        candidates.sort(key=sort_key)
        
        current_code = seed_words[:]
        current_set = seed_set.copy()
        
        # 1. Greedy Pass
        for w in candidates:
            valid = True
            for c in current_code:
                if BIT_COUNTS[w ^ c] < MIN_DIST:
                    valid = False
                    break
            if valid:
                current_code.append(w)
                current_set.add(w)
                
        # 2. Local Search (Swaps)
        # Try to improve the code by swapping words
        improved = True
        max_swaps = 5000
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() - start_time < time_limit:
            improved = False
            # Shuffle code to randomize order of checks
            code_list = current_code[:]
            random.shuffle(code_list)
            
            for idx, u in enumerate(code_list):
                if len(current_code) < 2: break
                # Try to swap u with a random v not in code
                # We need to find a v that is compatible with C \ {u}
                # Random sampling of v
                for _ in range(10): # Try 10 random candidates for swap
                    v = random.choice(all_words)
                    if v in current_set:
                        continue
                    
                    # Check if v is compatible with all c in C except u
                    compatible = True
                    for c in current_code:
                        if c == u:
                            continue
                        if BIT_COUNTS[v ^ c] < MIN_DIST:
                            compatible = False
                            break
                    
                    if compatible:
                        # Perform swap
                        current_code.remove(u)
                        current_set.remove(u)
                        current_code.append(v)
                        current_set.add(v)
                        improved = True
                        swaps_done += 1
                        break # Restart loop with new code
                
        # 3. Try to add more words after swaps
        # Re-run a quick greedy pass on remaining candidates
        # This might be slow, so we do it sparsely
        if restarts % 5 == 0:
            # Filter candidates that are not in current_set
            remaining = [w for w in all_words if w not in current_set]
            random.shuffle(remaining)
            for w in remaining[:1000]: # Check first 1000 shuffled
                valid = True
                for c in current_code:
                    if BIT_COUNTS[w ^ c] < MIN_DIST:
                        valid = False
                        break
                if valid:
                    current_code.append(w)
                    current_set.add(w)
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
    return best_code
