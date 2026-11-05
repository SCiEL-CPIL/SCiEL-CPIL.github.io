import time
import random

def construct():
    N = 17
    MIN_DISTANCE = 6
    
    # Generate the 32-word Reed-Muller seed
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
            
    best_codes = list(rm_words)
    best_size = len(best_codes)
    
    start_time = time.time()
    # Use 50s limit to be safe within 60s wall clock
    deadline = start_time + 50.0
    
    # Pre-generate a shuffled list of all words to avoid repeated list creation overhead
    all_words = list(range(1 << N))
    
    while time.time() < deadline:
        random.shuffle(all_words)
        current_codes = list(best_codes)
        
        added_any = False
        for w in all_words:
            if time.time() > deadline:
                break
            
            # Check if w is valid
            is_valid = True
            for c in current_codes:
                if (w ^ c).bit_count() < MIN_DISTANCE:
                    is_valid = False
                    break
            
            if is_valid:
                current_codes.append(w)
                added_any = True
                
        if len(current_codes) > best_size:
            best_size = len(current_codes)
            best_codes = current_codes
            
        # If we couldn't add any words in a full pass, we might be stuck.
        # Break to avoid infinite loop of no progress.
        if not added_any:
            break

    return best_codes
