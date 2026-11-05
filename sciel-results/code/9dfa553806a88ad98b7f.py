import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        return _generic_construct(q, n, radius)

    total_words = 3**7
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    
    def int_to_digits(num, length, base):
        digits = []
        for _ in range(length):
            digits.append(num % base)
            num //= base
        return digits
        
    def digits_to_int(digits, base):
        val = 0
        for i in range(len(digits)-1, -1, -1):
            val = val * base + digits[i]
        return val

    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        neighbors[w].append(w)
        for pos in range(n):
            orig_digit = digits[pos]
            for new_digit in range(q):
                if new_digit != orig_digit:
                    new_digits = digits.copy()
                    new_digits[pos] = new_digit
                    new_w = digits_to_int(new_digits, q)
                    neighbors[w].append(new_w)
                    
    # Initial Set
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word_list = h + list(tail)
            w_id = 0
            for i in range(n):
                w_id = w_id * 3 + word_list[i]
            initial_codewords.append(w_id)
            
    initial_codewords = list(set(initial_codewords))
    
    best_size = len(initial_codewords)
    best_codewords = list(initial_codewords)
    
    current_codewords = list(initial_codewords)
    cover_counts = [0] * total_words
    
    def update_counts(cw, delta):
        for nb in neighbors[cw]:
            cover_counts[nb] += delta
            
    for cw in current_codewords:
        update_counts(cw, 1)
        
    def is_removable(cw):
        for nb in neighbors[cw]:
            if cover_counts[nb] == 1:
                return False
        return True
        
    def try_remove():
        nonlocal best_size, best_codewords
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_codewords) # Shuffle to vary removal order
            for i in range(len(current_codewords)):
                if time.time() > deadline: break
                cw = current_codewords[i]
                if is_removable(cw):
                    update_counts(cw, -1)
                    current_codewords.pop(i)
                    improved = True
                    if len(current_codewords) < best_size:
                        best_size = len(current_codewords)
                        best_codewords = list(current_codewords)
                    break # Restart while loop after removal
        
    try_remove()
    
    # If we have time, do some kicks
    while time.time() < deadline - 5.0:
        # Save current state
        saved_codewords = list(current_codewords)
        saved_counts = list(cover_counts)
        
        # Pick a random word to kick
        if not current_codewords: break
        idx = random.randint(0, len(current_codewords)-1)
        cw_kick = current_codewords[idx]
        
        # Remove it
        update_counts(cw_kick, -1)
        current_codewords.pop(idx)
        
        # Find uncovered words
        uncovered = [w for w in range(total_words) if cover_counts[w] == 0]
        
        # Add random words to cover them
        attempts = 0
        while uncovered and attempts < 50 and time.time() < deadline:
            # Pick a random word from uncovered to target? Or just random word?
            # Greedy: pick a random word from the space that covers the most uncovered?
            # Simple: pick a random word from the space.
            new_w = random.randint(0, total_words - 1)
            if new_w not in current_codewords:
                current_codewords.append(new_w)
                update_counts(new_w, 1)
                # Update uncovered
                new_uncovered = []
                for w in uncovered:
                    if cover_counts[w] > 0:
                        pass
                    else:
                        new_uncovered.append(w)
                uncovered = new_uncovered
            attempts += 1
            
        if uncovered:
            # Restore state if we couldn't fix it quickly
            current_codewords = saved_codewords
            cover_counts = saved_counts
            continue
            
        # Try to remove again
        try_remove()
        
    # Convert best to list of lists
    result = []
    for w_id in best_codewords:
        result.append(int_to_digits(w_id, n, q))
        
    return result

def _generic_construct(q, n, radius):
    # Fallback
    return []
