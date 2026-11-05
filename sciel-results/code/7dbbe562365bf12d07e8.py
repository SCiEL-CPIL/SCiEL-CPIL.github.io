import itertools
import time
import random

def construct(q, n, radius):
    # Precompute word indices and balls
    # q=3, n=6
    # Total words = 729
    
    # Map tuple to index
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * (q ** i)
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(q ** n)]
    
    # Precompute balls (list of indices covered by each word)
    balls = []
    for w in all_words:
        covered = set()
        # The word itself
        covered.add(word_to_idx(w))
        # Words at distance 1
        for i in range(n):
            orig_val = w[i]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[i] = new_val
                    covered.add(word_to_idx(new_w))
        balls.append(list(covered))
        
    # For fast subset check: balls_set[i] is a set of indices covered by word i
    balls_set = [set(b) for b in balls]
    
    # Precompute: for each word index, which codeword indices cover it
    # cover_by[w] = list of codeword indices c such that w in balls[c]
    cover_by = [[] for _ in range(q ** n)]
    for c in range(q ** n):
        for w in balls[c]:
            cover_by[w].append(c)
    
    # Initial code: Direct sum of ternary tetracode and full space of length 2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_code.append(word_to_idx(w))
            
    # Ensure distinct
    initial_code = list(set(initial_code))
    
    # Current code
    code = set(initial_code)
    
    # Compute coverage counts
    counts = [0] * (q ** n)
    for c in code:
        for w in balls[c]:
            counts[w] += 1
            
    # Verify initial code covers everything
    assert all(counts[w] >= 1 for w in range(q ** n)), "Initial code doesn't cover all words"
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Temperature schedule
    T_init = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    best_code = set(code)
    best_size = len(code)
    
    T = T_init
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Choose a random move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Move: remove a codeword (if valid, i.e., no uncovered words)
            c = rng.choice(list(code))
            unique = [w for w in balls[c] if counts[w] == 1]
            if unique:
                # Can't remove without uncovering
                # Skip this move
                continue
            # Remove c
            code.remove(c)
            for w in balls[c]:
                counts[w] -= 1
            delta_size = -1
            # Always accept (size decreased)
            # Check if we improved
            if len(code) < best_size:
                best_code = set(code)
                best_size = len(code)
                
        elif move_type < 0.7:
            # Move: add a codeword
            c_new = rng.randrange(q ** n)
            if c_new in code:
                continue
            code.add(c_new)
            for w in balls[c_new]:
                counts[w] += 1
            delta_size = 1
            # Accept with probability exp(-delta_size / T)
            if rng.random() > pow(2.718281828, -delta_size / T):
                # Reject: undo
                code.remove(c_new)
                for w in balls[c_new]:
                    counts[w] -= 1
            else:
                # Accept
                pass
                
        else:
            # Move: swap - remove c, add c_prime
            c = rng.choice(list(code))
            unique = [w for w in balls[c] if counts[w] == 1]
            if not unique:
                # Can remove without issue, but swap means we add something
                # Actually for swap, we want to replace c with c_prime
                # The net effect is size stays same
                pass
            
            # Find c_prime that covers all unique words of c
            if unique:
                candidates = []
                for c_prime in range(q ** n):
                    if c_prime in code:
                        continue
                    # Check if balls[c_prime] covers all words in unique
                    is_valid = True
                    for w in unique:
                        if w not in balls_set[c_prime]:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.append(c_prime)
                
                if not candidates:
                    continue
                    
                c_prime = rng.choice(candidates)
                
                # Perform swap
                code.remove(c)
                code.add(c_prime)
                for w in balls[c]:
                    counts[w] -= 1
                for w in balls[c_prime]:
                    counts[w] += 1
                delta_size = 0
                # Always accept equal-size moves (or with SA probability)
                # For equal size, always accept to explore
                if len(code) < best_size:
                    best_code = set(code)
                    best_size = len(code)
            else:
                # No unique words, so removing c is safe
                # Just add a random codeword and remove c (net +1 then -1 = 0)
                # Actually this is just adding a codeword
                c_prime = rng.randrange(q ** n)
                if c_prime in code:
                    continue
                code.remove(c)
                code.add(c_prime)
                for w in balls[c]:
                    counts[w] -= 1
                for w in balls[c_prime]:
                    counts[w] += 1
                delta_size = 0
                if len(code) < best_size:
                    best_code = set(code)
                    best_size = len(code)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final greedy cleanup: try to remove any redundant codewords
    code = set(best_code)
    counts = [0] * (q ** n)
    for c in code:
        for w in balls[c]:
            counts[w] += 1
    
    # Greedy removal pass
    for _ in range(100):
        removed = False
        for c in list(code):
            unique = [w for w in balls[c] if counts[w] == 1]
            if not unique:
                code.remove(c)
                for w in balls[c]:
                    counts[w] -= 1
                removed = True
        if not removed:
            break
    
    # Return the list of codewords (as lists of integers)
    return [idx_to_word(c) for c in code]
