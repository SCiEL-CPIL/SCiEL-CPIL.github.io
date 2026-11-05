import numpy as np
import time
import random

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1
    # Total words = 729
    # Each ball has size 1 + n*(q-1) = 1 + 6*2 = 13
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute all words as numpy array of shape (total_words, n)
    all_words = np.zeros((total_words, n), dtype=np.int32)
    for idx in range(total_words):
        val = idx
        for i in range(n):
            all_words[idx, i] = val % q
            val //= q
    
    # Precompute balls: for each word, which words are within radius 1
    # balls_mask[c] is a boolean array of length total_words
    # But 729*729 booleans = ~530K, manageable
    
    # Actually, let's use a different approach: 
    # For each position i and each value change, compute the index offset
    # word index: sum(w[j] * q^j)
    # Changing position i from v to v' changes index by (v'-v) * q^i
    
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    # For each word c, its ball consists of:
    # c itself, and for each position i, c + (v'-v)*q^i for v' != v
    # Let's precompute the ball offsets for each position
    # offsets[i][v][v'] = (v'-v) * q^i
    
    # Actually, simpler: for each word c, the ball is:
    # {c + d : d in deltas} where deltas includes 0 and all single-position changes
    # But the deltas depend on the current values at each position
    
    # Let's just precompute balls as lists of indices
    balls = []
    for c in range(total_words):
        covered = set()
        covered.add(c)
        for i in range(n):
            orig_val = all_words[c, i]
            for new_val in range(q):
                if new_val != orig_val:
                    new_idx = c + (new_val - orig_val) * q_powers[i]
                    covered.add(new_idx)
        balls.append(list(covered))
    
    # Convert to numpy arrays for speed
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    # Initial code: tetracode * {0,1,2}^2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in range(q ** (n - 4)):
            w = h + [tail % q, tail // q % q]
            idx = 0
            for j, val in enumerate(w):
                idx += val * q_powers[j]
            initial_code.append(idx)
    
    initial_code = list(set(initial_code))
    code_set = set(initial_code)
    
    # Compute coverage counts
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        counts[balls_np[c]] += 1
    
    assert np.all(counts >= 1), "Initial code doesn't cover all words"
    
    # Simulated Annealing with numpy
    start_time = time.time()
    deadline = start_time + 110
    
    T_init = 3.0
    T_min = 0.001
    cooling = 0.9998
    
    best_code = list(code_set)
    best_size = len(code_set)
    
    T = T_init
    rng = random.Random(42)
    
    # Precompute: for each word, the list of codewords that cover it (for swap moves)
    # This is expensive, so let's skip it and use a simpler approach
    
    while time.time() < deadline:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a codeword if no unique words
            if not code_set:
                continue
            c = rng.choice(list(code_set))
            # Check if any word is uniquely covered by c
            ball_c = balls_np[c]
            unique_count = np.sum(counts[ball_c] == 1)
            if unique_count == 0:
                # Safe to remove
                code_set.remove(c)
                counts[ball_c] -= 1
                if len(code_set) < best_size:
                    best_code = list(code_set)
                    best_size = len(code_set)
            # If unique_count > 0, skip (can't remove)
            
        elif move_type < 0.7:
            # Add a codeword
            c_new = rng.randrange(total_words)
            if c_new in code_set:
                continue
            code_set.add(c_new)
            counts[balls_np[c_new]] += 1
            # Accept with probability exp(-1/T)
            if rng.random() < np.exp(-1.0 / T):
                pass  # Accept
            else:
                # Reject: undo
                code_set.remove(c_new)
                counts[balls_np[c_new]] -= 1
                
        else:
            # Swap: remove c, add c_prime
            if not code_set:
                continue
            c = rng.choice(list(code_set))
            ball_c = balls_np[c]
            unique_words = ball_c[counts[ball_c] == 1]
            
            if len(unique_words) == 0:
                # Can remove c safely, so just add a random codeword and remove c
                c_prime = rng.randrange(total_words)
                if c_prime in code_set:
                    continue
                code_set.remove(c)
                code_set.add(c_prime)
                counts[ball_c] -= 1
                counts[balls_np[c_prime]] += 1
            else:
                # Need c_prime to cover all unique words
                # Find candidates that cover all unique words
                # This is expensive, so sample a few candidates
                found = False
                for _ in range(20):
                    c_prime = rng.randrange(total_words)
                    if c_prime in code_set:
                        continue
                    # Check if balls_np[c_prime] covers all unique_words
                    if np.all(np.isin(unique_words, balls_np[c_prime])):
                        # Perform swap
                        code_set.remove(c)
                        code_set.add(c_prime)
                        counts[ball_c] -= 1
                        counts[balls_np[c_prime]] += 1
                        found = True
                        break
                if not found:
                    continue
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final greedy cleanup
    code_set = set(best_code)
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        counts[balls_np[c]] += 1
    
    for _ in range(200):
        removed = False
        for c in list(code_set):
            ball_c = balls_np[c]
            if np.sum(counts[ball_c] == 1) == 0:
                code_set.remove(c)
                counts[ball_c] -= 1
                removed = True
        if not removed:
            break
    
    # Convert back to list of lists
    result = []
    for c in code_set:
        w = [0] * n
        val = c
        for i in range(n):
            w[i] = val % q
            val //= q
        result.append(w)
    
    return result
