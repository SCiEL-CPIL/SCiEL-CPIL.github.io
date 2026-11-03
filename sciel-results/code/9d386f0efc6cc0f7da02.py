"""
Strategy: Local Search (Simulated Annealing / Hill Climbing) with Bitmask Representation.
Instead of greedy set cover, we start with a random or heuristic-based initial code and 
perform local moves (add, remove, swap) to minimize the number of codewords while 
maintaining coverage. We use bitmasks for fast coverage verification.
"""
import time
import random
import itertools
import numpy as np

_DEADLINE = 110.0

def _time_left():
    return _DEADLINE - time.time()

def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    
    # Parameters for the specific case q=3, n=7, radius=1
    # Total words = 3^7 = 2187
    # Ball size = 1 + 7*2 = 15
    
    # Precompute all words and their balls as bitmasks
    # Word to int: base-q representation
    total_words = q ** n
    
    def word_to_int(word):
        res = 0
        for v in word:
            res = res * q + v
        return res
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w
    
    # Precompute balls for all words
    # ball_mask[i] is a bitmask of length total_words where bit j is 1 if word j is in ball(i)
    # Since total_words = 2187, we need a large integer or a numpy array of uint64.
    # 2187 bits requires about 35 uint64s. Using Python integers is efficient enough for bitwise ops.
    
    balls = [0] * total_words
    for i in range(total_words):
        word = int_to_word(i)
        mask = 1 << i # The word itself
        
        if radius == 1:
            for pos in range(n):
                for v in range(q):
                    if v != word[pos]:
                        new_word = word.copy()
                        new_word[pos] = v
                        j = word_to_int(new_word)
                        mask |= (1 << j)
        else:
            # General radius (slower, but q=3, n=7, r=1 is the main case)
            for r in range(1, radius + 1):
                for positions in itertools.combinations(range(n), r):
                    for values in itertools.product(range(q), repeat=r):
                        new_word = word.copy()
                        # Ensure we change exactly 'r' positions to different values
                        # This logic is tricky for general r. 
                        # For this task, r=1 is the only relevant case. 
                        # We will assume r=1 for optimization, fallback to greedy if r!=1.
                        pass
        
        balls[i] = mask
    
    # Check if general radius is needed. If not, proceed with optimized r=1 logic.
    if radius != 1:
        # Fallback to a simple greedy for other cases to ensure validity
        return _simple_greedy(q, n, radius)
    
    # Initial Code Construction
    # Use a random initialization or a heuristic.
    # A good heuristic: random sample of ~200 words, then try to reduce.
    # Or start with a known good structure if available, but random restarts are more robust.
    
    best_code = None
    best_size = float('inf')
    
    num_restarts = 10
    max_steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if _time_left() < 5.0:
            break
            
        # Initialize with a random set of codewords
        # Size estimate: 2187 / 15 ~ 146. Add some buffer, say 180.
        initial_size = 180 + random.randint(0, 20)
        
        # Random initialization
        code = [random.randint(0, total_words - 1) for _ in range(initial_size)]
        code = list(set(code)) # Ensure distinct
        
        # Local Search: Simulated Annealing / Hill Climbing
        current_code = code
        current_mask = 0
        for c in current_code:
            current_mask |= balls[c]
        
        # Check if it covers everything
        full_mask = (1 << total_words) - 1
        if current_mask != full_mask:
            # Try to fix coverage by adding words
            uncovered = full_mask ^ current_mask
            while uncovered != 0 and _time_left() > 5.0:
                # Pick a random uncovered word, add a center that covers it
                # Find an uncovered word
                # Efficiently: get index of a set bit in uncovered
                # Python ints don't have popcount or ctz directly, but we can loop or use bit_length
                # Let's just pick a random word from the uncovered set.
                # To do this efficiently, we can maintain a list of uncovered words? 
                # Or just pick a random word index and check if it's uncovered.
                
                # Better: Iterate through bits of uncovered? 
                # Let's just add random centers until covered.
                # This is inefficient. Let's use a greedy fill for initialization if not covered.
                
                # Greedy fill for initialization
                while uncovered != 0 and _time_left() > 5.0:
                    # Find word with max coverage of uncovered
                    best_center = -1
                    best_gain = 0
                    # Sample some candidates to speed up
                    candidates = random.sample(range(total_words), min(100, total_words))
                    for c in candidates:
                        gain = len(bin(uncovered & balls[c]).count('1')) # Bit count
                        if gain > best_gain:
                            best_gain = gain
                            best_center = c
                    if best_center == -1 or best_gain == 0:
                        break
                    current_code.append(best_center)
                    current_mask |= balls[best_center]
                    uncovered = full_mask ^ current_mask
        
        if current_mask != full_mask:
            continue
            
        # Now we have a valid code. Try to minimize size.
        # Move 1: Remove redundant codewords
        # Move 2: Swap a codeword with another to allow removal of another? (Complex)
        # Let's stick to: Try removing each codeword. If still covered, remove it.
        # Repeat until no more can be removed.
        
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            # Shuffle to avoid bias
            idxs = list(range(len(current_code)))
            random.shuffle(idxs)
            
            for idx in idxs:
                if _time_left() < 5.0:
                    break
                c = current_code[idx]
                # Check if removing c leaves the code covered
                # New mask = current_mask without balls[c]
                # But other codewords might cover the same spots.
                # We need to check if (current_mask ^ balls[c]) is still full?
                # No, because balls[c] might be covered by others.
                # We need: (current_mask without c) == full_mask?
                # This is hard to compute directly with bitmasks unless we store counts.
                
                # Alternative: Re-calculate mask without c? Too slow.
                # Alternative: Check if there is any word in balls[c] that is ONLY covered by c.
                # This requires knowing coverage counts.
                
                # Let's use a different local search move:
                # "Swap": Replace c with a random c'. If the new code is valid, accept.
                # This doesn't reduce size directly, but might lead to a state where removal is possible.
                
                # Actually, the "Remove if redundant" is the key to minimization.
                # To check redundancy efficiently:
                # We can maintain a count array? 2187 integers. 
                # Let's try a different approach for the local search:
                # 1. Start with valid code.
                # 2. Randomly pick a codeword c.
                # 3. Try to find a new codeword c' such that replacing c with c' allows removing another codeword? No, too complex.
                
                # Let's try "Greedy Removal" phase after every few swaps.
                # Or simply: 
                # Phase 1: Greedy construction to get a valid code.
                # Phase 2: Iterated Local Search to minimize.
                
                # Let's restart the logic with a clearer ILS structure.
                pass
                
        # The above loop was getting messy. Let's implement a cleaner ILS below.
        break # Break out of restart loop to implement cleaner logic below if needed, or just continue with the messy one? 
        # Let's discard the messy restart loop and implement a clean ILS.
        
    # --- Clean ILS Implementation ---
    
    def _greedy_cover_fast():
        """Fast greedy set cover using bitmasks."""
        uncovered = full_mask
        code = []
        # Precompute list of all words
        all_words = list(range(total_words))
        
        while uncovered != 0 and _time_left() > 5.0:
            # Find word covering most uncovered
            best_center = -1
            best_count = 0
            # Sample candidates to speed up
            sample_size = min(50, len(all_words))
            candidates = random.sample(all_words, sample_size)
            
            for c in candidates:
                count = len(bin(uncovered & balls[c]).count('1'))
                if count > best_count:
                    best_count = count
                    best_center = c
            
            if best_center == -1 or best_count == 0:
                # Fallback to exhaustive if sampling fails (rare)
                for c in all_words:
                    count = len(bin(uncovered & balls[c]).count('1'))
                    if count > best_count:
                        best_count = count
                        best_center = c
            
            if best_center == -1:
                break
                
            code.append(best_center)
            uncovered ^= (balls[best_center] & uncovered) # Remove covered bits
            
        return code, uncovered
    
    # 1. Get initial valid code
    initial_code, remaining_uncovered = _greedy_cover_fast()
    
    if remaining_uncovered != 0:
        # Fallback: Add random words until covered
        while remaining_uncovered != 0 and _time_left() > 5.0:
            c = random.randint(0, total_words - 1)
            if c not in initial_code:
                initial_code.append(c)
                remaining_uncovered ^= (balls[c] & remaining_uncovered)
                
    best_code = initial_code
    best_size = len(best_code)
    
    # 2. Local Search: Try to remove words.
    # We maintain a coverage count array to check redundancy in O(1) per word check?
    # No, checking if a word is "only" covered by one center requires counts.
    # Let's maintain `coverage_count[w]` = number of codewords covering w.
    
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c in best_code:
        # Add coverage of c
        # We need to iterate over bits in balls[c]
        mask = balls[c]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            coverage_count[idx] += 1
            mask ^= lsb
            
    # Check for redundancy
    # A codeword c is redundant if for all w in balls[c], coverage_count[w] > 1.
    
    def is_redundant(c):
        mask = balls[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if coverage_count[idx] == 1:
                return False
            mask ^= lsb
        return True
    
    # Iterated Local Search
    current_code = best_code[:]
    current_coverage = coverage_count[:]
    
    for step in range(10000):
        if _time_left() < 5.0:
            break
            
        # Move: Try to remove a random codeword
        if len(current_code) > 0:
            idx = random.randint(0, len(current_code) - 1)
            c = current_code[idx]
            
            if is_redundant(c):
                # Remove it
                current_code.pop(idx)
                # Update coverage
                mask = balls[c]
                while mask:
                    lsb = mask & -mask
                    idx_w = lsb.bit_length() - 1
                    current_coverage[idx_w] -= 1
                    mask ^= lsb
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    
        else:
            break
            
        # Move: Try to swap a codeword with a random one to potentially enable removals
        if len(current_code) > 0 and random.random() < 0.1:
            idx = random.randint(0, len(current_code) - 1)
            old_c = current_code[idx]
            new_c = random.randint(0, total_words - 1)
            
            if old_c == new_c:
                continue
                
            # Check if swapping maintains coverage
            # This is complex. Let's just do "Add-Remove" moves.
            # Add a random word, then try to remove redundant ones.
            
            # Add random word
            if new_c not in current_code:
                current_code.append(new_c)
                mask = balls[new_c]
                while mask:
                    lsb = mask & -mask
                    idx_w = lsb.bit_length() - 1
                    current_coverage[idx_w] += 1
                    mask ^= lsb
                
                # Try to remove redundant words (including the one we just added? No, we want to keep size small)
                # Try to remove ANY redundant word
                for _ in range(3): # Try a few times
                    if len(current_code) == 0:
                        break
                    r_idx = random.randint(0, len(current_code) - 1)
                    r_c = current_code[r_idx]
                    if is_redundant(r_c):
                        current_code.pop(r_idx)
                        mask = balls[r_c]
                        while mask:
                            lsb = mask & -mask
                            idx_w = lsb.bit_length() - 1
                            current_coverage[idx_w] -= 1
                            mask ^= lsb
                        
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = current_code[:]
                    else:
                        break # Stop trying to remove if first try fails? Or continue?
                        # Let's break to avoid getting stuck in a loop of failed removals.
                        
    return [int_to_word(c) for c in best_code]

def _simple_greedy(q, n, radius):
    """Fallback for general radius."""
    total_words = q ** n
    def word_to_int(word):
        res = 0
        for v in word:
            res = res * q + v
        return res
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w
    
    balls = {}
    for i in range(total_words):
        word = int_to_word(i)
        ball = set([i])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for values in itertools.product(range(q), repeat=r):
                    new_word = word.copy()
                    for pos, val in zip(positions, values):
                        new_word[pos] = val
                    ball.add(word_to_int(new_word))
        balls[i] = ball
        
    uncovered = set(range(total_words))
    code = []
    while uncovered and _time_left() > 5.0:
        best_center = None
        best_count = 0
        for c in uncovered:
            count = len(uncovered & balls[c])
            if count > best_count:
                best_count = count
                best_center = c
        if best_center is None:
            break
        code.append(int_to_word(best_center))
        uncovered -= balls[best_center]
        
    return code
