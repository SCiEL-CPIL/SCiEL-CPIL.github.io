import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    ALL_WORDS = 1 << N
    DEADLINE = time.time() + 85.0  # 5s margin
    
    # Generate Reed-Muller code (32 words)
    def generate_rm_code():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    rm_words = generate_rm_code()
    best_code = rm_words[:]
    best_size = len(best_code)
    
    # Pre-compute bit counts for all words to speed up sorting?
    # Not strictly necessary for the greedy check, but useful for sorting.
    
    # Candidate pool is all words
    # We will use different ordering strategies
    
    # Strategy 1: Sorted by weight (center-out)
    # Strategy 2: Random shuffle
    # Strategy 3: Sorted by weight, but starting from extremes?
    
    # To avoid re-creating the list of 262k words every time, we can just iterate range(ALL_WORDS)
    # But we need different orders.
    
    # Let's create a list of indices 0..2^18-1
    # Shuffling 262k ints is fast.
    
    indices = list(range(ALL_WORDS))
    
    for restart in range(100):
        if time.time() > DEADLINE:
            break
        
        # Choose strategy
        if restart % 3 == 0:
            # Center-out weight sort
            # We need to sort by weight. 
            # To avoid sorting 262k items every time (which takes ~0.1s), 
            # we can pre-sort once and reuse? 
            # But we need to skip words already in code.
            # Actually, sorting 262k items is fast enough (<0.1s).
            # Let's just do it.
            
            # Optimization: Pre-compute weights?
            # No, just sort.
            current_order = sorted(indices, key=lambda x: (abs(x.bit_count() - 9), x))
        elif restart % 3 == 1:
            current_order = indices[:]
            random.shuffle(current_order)
        else:
            # Random but biased towards weight 9?
            # Just shuffle for now.
            current_order = indices[:]
            random.shuffle(current_order)
            
        current_code = rm_words[:]
        code_set = set(rm_words)
        
        # Greedy
        for w in current_order:
            if time.time() > DEADLINE:
                break
                
            if w in code_set:
                continue
                
            # Check compatibility
            # Early termination
            compatible = True
            for c in current_code:
                if (w ^ c).bit_count() < MIN_DIST:
                    compatible = False
                    break
                    
            if compatible:
                current_code.append(w)
                code_set.add(w)
                
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
    return best_code
