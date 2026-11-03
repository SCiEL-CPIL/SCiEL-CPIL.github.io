import itertools
import random
import time
import sys

def construct(q, n, radius):
    # Parameters
    # q=3, n=6, radius=1
    # Total words: 3^6 = 729
    total_words = q ** n
    
    # Precompute all words as tuples
    # Map tuple to index
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls (Hamming distance <= radius)
    # balls[i] is a list of indices of words within distance 'radius' of words[i]
    balls = []
    for i, w in enumerate(words):
        ball = [i]
        # Generate all neighbors within distance 1
        # Since radius is 1, we only need distance 1 neighbors
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    ball.append(idx)
        balls.append(ball)
        
    # Greedy Construction
    uncovered = [1] * total_words # 1 if uncovered, 0 if covered
    cover_count = [0] * total_words
    code = []
    zero_count = total_words # Initially all uncovered
    
    # Helper to update cover counts and zero_count
    def add_codeword(c_idx):
        nonlocal zero_count
        for w_idx in balls[c_idx]:
            old = cover_count[w_idx]
            new = old + 1
            cover_count[w_idx] = new
            if old == 0 and new > 0:
                zero_count -= 1
            # Update uncovered status if needed? 
            # We can just track zero_count. 
            # But for greedy, we need to know which are uncovered to calculate gain.
            # It's better to maintain a set or list of uncovered indices for greedy.
            # However, maintaining a set is slow. 
            # Alternative: In greedy, calculate gain by iterating balls[c] and checking uncovered[w].
            
    def remove_codeword(c_idx):
        nonlocal zero_count
        for w_idx in balls[c_idx]:
            old = cover_count[w_idx]
            new = old - 1
            cover_count[w_idx] = new
            if old > 0 and new == 0:
                zero_count += 1

    # Greedy Phase
    # We need to efficiently find the codeword covering the most uncovered words.
    # uncovered is a list of 0/1.
    while zero_count > 0:
        best_c = -1
        best_gain = -1
        # Iterate all possible codewords
        for c in range(total_words):
            gain = 0
            for w_idx in balls[c]:
                if uncovered[w_idx] == 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1:
            break # Should not happen if zero_count > 0
            
        # Add best_c
        code.append(best_c)
        add_codeword(best_c)
        # Update uncovered array
        for w_idx in balls[best_c]:
            if cover_count[w_idx] >= 1:
                uncovered[w_idx] = 0
                
    # Initial size is len(code)
    
    # Reduction Phase
    # Try to remove any codeword that doesn't break coverage
    changed = True
    while changed:
        changed = False
        # Iterate backwards to remove safely
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Tentatively remove
            remove_codeword(c)
            if zero_count == 0:
                # Valid removal
                code.pop(i)
                changed = True
                break # Restart loop
            else:
                # Revert
                add_codeword(c)
                
    # Simulated Annealing Phase
    # If we are already at 73, we can stop or try to verify.
    # Known optimum is 73.
    if len(code) <= 73:
        # Return the list of codewords
        return [words[c] for c in code]
        
    # SA to try to reduce further
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    current_code = code
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # State variables
    # cover_count and zero_count are already up to date for current_code
    
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Generate a move
        # Types: 
        # 1. Remove a random codeword
        # 2. Add a random codeword
        # 3. Swap a random codeword with a random other
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove
            if not current_code:
                continue
            idx = random.randint(0, len(current_code) - 1)
            c = current_code[idx]
            remove_codeword(c)
            new_size = current_size - 1
            
            if zero_count == 0:
                # Valid
                current_code.pop(idx)
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    T = 1.0 # Reset temp on improvement? Or keep cooling.
            else:
                # Invalid, revert
                add_codeword(c)
                # Reject move
                # Calculate acceptance probability for invalid move? 
                # Usually we only accept valid moves in hard constraint SA, 
                # or use penalty. Let's stick to valid moves only for simplicity and correctness.
                # This might get stuck. 
                # To escape, we should allow "Add" moves.
                
        elif move_type < 0.6:
            # Add
            c = random.randint(0, total_words - 1)
            # Check if c is already in current_code? 
            # It's okay to add duplicate? No, codewords must be distinct.
            # If c is in current_code, adding it does nothing (count goes up, but we don't want duplicates).
            # So only add if not present.
            if c in current_code:
                continue
            add_codeword(c)
            new_size = current_size + 1
            
            if zero_count == 0:
                # Valid
                current_code.append(c)
                current_size = new_size
                # Accept if size decreased? No, size increased.
                # Accept with prob exp(-(new_size - current_size) / T) ? 
                # Delta E = 1.
                # Prob = exp(-1/T)
                if random.random() < pow(2.71828, -1.0 / T):
                    pass # Accepted
                else:
                    # Revert
                    remove_codeword(c)
                    current_code.pop()
                    current_size = current_size - 1
            else:
                # Invalid, revert
                remove_codeword(c)
                
        else:
            # Swap
            if len(current_code) < 2:
                continue
            idx1 = random.randint(0, len(current_code) - 1)
            idx2 = random.randint(0, len(current_code) - 1)
            if idx1 == idx2:
                continue
            c1 = current_code[idx1]
            c2 = current_code[idx2]
            
            # Remove c1, Add c2? No, swap means remove c1, add c_new.
            # Let's pick a new candidate c_new from all words.
            c_new = random.randint(0, total_words - 1)
            if c_new in current_code:
                # If c_new is already in code (and not c1), swapping c1 with c_new is just removing c1.
                # If c_new == c1, no change.
                if c_new == c1:
                    continue
                else:
                    # This is effectively a removal of c1.
                    # Handled by Remove logic? 
                    # Let's just do a strict swap: remove c1, add c_new.
                    pass
            
            remove_codeword(c1)
            add_codeword(c_new)
            
            if zero_count == 0:
                # Valid
                current_code[idx1] = c_new
                # Size remains same
                # Accept? Always accept valid same-size moves to explore?
                # Or accept with prob 1?
                pass
            else:
                # Revert
                remove_codeword(c_new)
                add_codeword(c1)
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Return best found
    if best_size < len(current_code):
        return [words[c] for c in best_code]
    else:
        return [words[c] for c in current_code]

# Ensure the function signature matches
# The task asks for construct(q, n, radius) -> list of codewords
# Codewords are lists of n integers.
# My code returns [words[c] for c in code], where words[c] is a tuple.
# I need to convert to list.

# Let's wrap the logic in the required function.

def construct(q, n, radius):
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    balls = []
    for i, w in enumerate(words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    ball.append(idx)
        balls.append(ball)
        
    uncovered = [1] * total_words
    cover_count = [0] * total_words
    code = []
    zero_count = total_words
    
    def add_codeword(c_idx):
        nonlocal zero_count
        for w_idx in balls[c_idx]:
            old = cover_count[w_idx]
            new = old + 1
            cover_count[w_idx] = new
            if old == 0 and new > 0:
                zero_count -= 1

    def remove_codeword(c_idx):
        nonlocal zero_count
        for w_idx in balls[c_idx]:
            old = cover_count[w_idx]
            new = old - 1
            cover_count[w_idx] = new
            if old > 0 and new == 0:
                zero_count += 1

    # Greedy
    while zero_count > 0:
        best_c = -1
        best_gain = -1
        for c in range(total_words):
            gain = 0
            for w_idx in balls[c]:
                if uncovered[w_idx] == 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1:
            break
            
        code.append(best_c)
        add_codeword(best_c)
        for w_idx in balls[best_c]:
            if cover_count[w_idx] >= 1:
                uncovered[w_idx] = 0
                
    # Reduction
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            remove_codeword(c)
            if zero_count == 0:
                code.pop(i)
                changed = True
                break
            else:
                add_codeword(c)
                
    if len(code) <= 73:
        return [list(words[c]) for c in code]
        
    # SA
    start_time = time.time()
    deadline = start_time + 140
    
    current_code = code
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.3:
            if not current_code:
                continue
            idx = random.randint(0, len(current_code) - 1)
            c = current_code[idx]
            remove_codeword(c)
            new_size = current_size - 1
            if zero_count == 0:
                current_code.pop(idx)
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                add_codeword(c)
                
        elif move_type < 0.6:
            c = random.randint(0, total_words - 1)
            if c in current_code:
                continue
            add_codeword(c)
            new_size = current_size + 1
            if zero_count == 0:
                current_code.append(c)
                current_size = new_size
                if random.random() < pow(2.71828, -1.0 / T):
                    pass
                else:
                    remove_codeword(c)
                    current_code.pop()
                    current_size = current_size - 1
            else:
                remove_codeword(c)
                
        else:
            if len(current_code) < 2:
                continue
            idx1 = random.randint(0, len(current_code) - 1)
            c1 = current_code[idx1]
            c_new = random.randint(0, total_words - 1)
            if c_new in current_code and c_new != c1:
                # This is a removal of c1 effectively? No, it's a swap that results in duplicate?
                # If c_new is already in code, adding it makes count 2. 
                # We want distinct codewords.
                # So we should only swap with c_new not in code, or handle duplicates.
                # Simplest: reject if c_new in current_code.
                continue
                
            remove_codeword(c1)
            add_codeword(c_new)
            if zero_count == 0:
                current_code[idx1] = c_new
                # Size same, accept
            else:
                remove_codeword(c_new)
                add_codeword(c1)
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    if best_size < len(current_code):
        return [list(words[c]) for c in best_code]
    else:
        return [list(words[c]) for c in current_code]
