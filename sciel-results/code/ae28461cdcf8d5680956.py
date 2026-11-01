import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Fallback to seed for non-standard parameters or if something goes wrong
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    total_words = q ** n
    # For q=3, n=8, radius=1
    # Ball size = 1 + n * (q - 1) = 1 + 8 * 2 = 17
    ball_size = 1 + n * (q - 1)
    
    # Pre-compute balls: array of shape (total_words, ball_size)
    # balls[i] contains the indices of the words covered by codeword i
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    
    # Helper to convert integer to base-q digits
    def int_to_digits(num, base, length):
        digits = [0] * length
        for i in range(length - 1, -1, -1):
            digits[i] = num % base
            num //= base
        return digits

    # Helper to convert digits to integer
    def digits_to_int(digits, base):
        val = 0
        for d in digits:
            val = val * base + d
        return val

    # Generate balls
    for i in range(total_words):
        digits = int_to_digits(i, q, n)
        balls[i, 0] = i  # The word itself
        idx = 1
        for pos in range(n):
            orig_digit = digits[pos]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = new_digit
                balls[i, idx] = digits_to_int(new_digits, q)
                idx += 1
                if idx == ball_size:
                    break
            if idx == ball_size:
                break

    best_code = None
    best_size = float('inf')
    
    # Multiple restarts to find a better local minimum
    rng = np.random.default_rng(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # 1. Greedy Set Cover
        covered = np.zeros(total_words, dtype=bool)
        selected = []
        
        for step in range(total_words):
            # Calculate gains: number of uncovered words in each ball
            # ~covered[balls] creates a boolean matrix (total_words, ball_size)
            # Sum along axis 1 gives the count of uncovered words per candidate
            gains = (~covered[balls]).sum(axis=1)
            
            # Find candidate with max gain
            # Tie-breaking: random among max gain candidates
            max_gain = np.max(gains)
            if max_gain == 0:
                break # All covered
            
            candidates = np.where(gains == max_gain)[0]
            # Pick randomly from candidates to diversify restarts
            c = candidates[rng.integers(len(candidates))]
            
            selected.append(c)
            covered[balls[c]] = True
            
            if len(selected) >= best_size:
                break # No point continuing if already worse than best found
        
        # 2. Local Search: Remove redundant codewords
        # Initialize counts
        counts = np.zeros(total_words, dtype=np.int32)
        # Vectorized addition
        # Flatten selected balls and use np.add.at
        if selected:
            flat_balls = balls[selected].flatten()
            np.add.at(counts, flat_balls, 1)
            
        current_selected = list(selected)
        
        # Removal loop
        # We repeat passes until no more removals
        # To avoid getting stuck in a bad order, we randomize the order of checks in each pass
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle order to break symmetry
            order = list(range(len(current_selected)))
            rng.shuffle(order)
            
            new_selected = []
            for idx in order:
                c = current_selected[idx]
                # Check if c is redundant: min count in its ball > 1
                # If min count > 1, then every word in ball(c) is covered by at least one other codeword
                if np.min(counts[balls[c]]) > 1:
                    # Remove c
                    counts[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            
            current_selected = new_selected
            
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_code = [list(map(int, digits_to_int.__self__ if False else [0]*n)) for _ in range(0)] # Placeholder
            # Actually, we need to return the list of codewords as lists of integers
            # current_selected contains indices. We need to convert them back to digit lists.
            best_code = []
            for c in current_selected:
                best_code.append(int_to_digits(c, q, n))
            
            # Early exit if we hit the known lower bound or a very good number
            if best_size <= 403:
                break

    if best_code is None:
        # Fallback to seed if something failed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
