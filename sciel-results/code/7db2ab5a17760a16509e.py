import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by enhanced Local Search
    with bitmask-based coverage counts and targeted swaps.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls: list of lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)
        
    # Precompute ball sets for fast membership tests
    ball_sets = [set(b) for b in balls]
    
    # 1. Greedy Set Cover with multiple restarts, keep best
    best_code = None
    best_size = float('inf')
    
    for restart in range(3):
        if time.time() - start_time > timeout:
            break
        
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            best_idx = -1
            max_cov = 0
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
                    if max_cov == 15:
                        break
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    code = best_code[:]
    
    # 2. Local Search with bitmask-based coverage
    # Maintain coverage counts as numpy array
    count = np.zeros(n_words, dtype=np.int32)
    for c in code:
        for j in balls[c]:
            count[j] += 1
    
    # Maintain a set for O(1) lookup of current codewords
    code_set = set(code)
    
    rng = random.Random(42)
    
    # Track best code found during local search
    best_code_ls = code[:]
    best_size_ls = len(code)
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout:
        if no_improve_count > max_no_improve:
            # Restart local search with current best
            code = best_code_ls[:]
            code_set = set(code)
            count = np.zeros(n_words, dtype=np.int32)
            for c in code:
                for j in balls[c]:
                    count[j] += 1
            no_improve_count = 0
            continue
        
        # Step A: Prune redundant codewords
        pruned = False
        new_code = []
        for c in code:
            redundant = True
            for j in balls[c]:
                if count[j] < 2:
                    redundant = False
                    break
            if redundant:
                for j in balls[c]:
                    count[j] -= 1
                pruned = True
            else:
                new_code.append(c)
        
        if pruned:
            code = new_code
            code_set = set(code)
            no_improve_count = 0
            if len(code) < best_size_ls:
                best_size_ls = len(code)
                best_code_ls = code[:]
            continue
        
        # Step B: Try targeted swaps
        # Strategy 1: Try to swap a codeword with a word that covers more uncovered/low-coverage points
        # Strategy 2: Random swap
        
        swap_found = False
        
        # Try several swap candidates
        for attempt in range(10):
            if time.time() - start_time > timeout:
                break
            
            # Pick a random codeword to potentially replace
            c_idx = rng.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Pick a candidate word w
            # Bias towards words that are "useful" - i.e., cover points with low coverage
            if attempt < 5:
                # Targeted: pick w that maximizes coverage of low-count points
                best_w = -1
                best_score = -1
                # Sample a subset of words for speed
                sample_size = min(200, n_words)
                for _ in range(sample_size):
                    w = rng.randint(0, n_words - 1)
                    if w == c or w in code_set:
                        continue
                    # Score: sum of (2 - count[j]) for j in balls[w] where count[j] < 2
                    score = 0
                    ball_w = balls[w]
                    for j in ball_w:
                        if count[j] == 0:
                            score += 10
                        elif count[j] == 1:
                            score += 5
                    if score > best_score:
                        best_score = score
                        best_w = w
                w = best_w
                if w == -1:
                    continue
            else:
                # Random
                w = rng.randint(0, n_words - 1)
                if w == c or w in code_set:
                    continue
            
            # Check validity of swap c -> w
            ball_w_set = ball_sets[w]
            valid = True
            for j in balls[c]:
                if j not in ball_w_set:
                    if count[j] < 2:
                        valid = False
                        break
                        
            if valid:
                # Accept swap
                for j in balls[c]:
                    count[j] -= 1
                for j in balls[w]:
                    count[j] += 1
                code[c_idx] = w
                code_set.discard(c)
                code_set.add(w)
                swap_found = True
                no_improve_count = 0
                if len(code) < best_size_ls:
                    best_size_ls = len(code)
                    best_code_ls = code[:]
                break
        
        if not swap_found:
            no_improve_count += 1
    
    # Use best code found
    code = best_code_ls[:]
    
    # Convert to output format
    result = [list(digits[c]) for c in code]
    return result
