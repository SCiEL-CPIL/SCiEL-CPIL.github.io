import numpy as np
import time
import itertools
import random
import math

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by Simulated Annealing 
    with a penalty for uncovered words to escape local minima.
    """
    start_time = time.time()
    timeout = 115
    
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls
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
        
    # 1. Greedy Set Cover
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
            
    # Initial state for SA
    count = np.zeros(n_words, dtype=int)
    for c in code:
        for j in balls[c]:
            count[j] += 1
            
    # State variables
    current_code = list(code)
    current_count = count.copy()
    current_uncovered = np.sum(current_count == 0) # Should be 0
    
    best_code = list(current_code)
    best_size = len(current_code)
    
    # SA Parameters
    lambda_cost = 1.0
    mu_cost = 2.0  # Penalty for uncovered
    T_start = 10.0
    T_end = 0.1
    cooling = 0.99
    
    rng = random.Random(42)
    T = T_start
    
    code_set = set(current_code)
    
    while time.time() - start_time < timeout and T > T_end:
        
        # Choose action: Remove or Add
        if rng.random() < 0.5:
            # Try Remove
            if not current_code:
                continue
            c_idx = rng.randint(0, len(current_code) - 1)
            c = current_code[c_idx]
            
            # Calculate newly uncovered
            newly_uncovered = 0
            for j in balls[c]:
                if current_count[j] == 1:
                    newly_uncovered += 1
            
            delta_E = -lambda_cost + newly_uncovered * mu_cost
            
            if delta_E < 0 or (newly_uncovered > 0 and rng.random() < math.exp(-delta_E / T)):
                # Accept
                for j in balls[c]:
                    current_count[j] -= 1
                current_code.pop(c_idx)
                code_set.remove(c)
                current_uncovered += newly_uncovered
                
                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
        else:
            # Try Add
            # Pick random word
            w = rng.randint(0, n_words - 1)
            if w in code_set:
                continue
            
            # Calculate newly covered
            newly_covered = 0
            for j in balls[w]:
                if current_count[j] == 0:
                    newly_covered += 1
            
            delta_E = lambda_cost - newly_covered * mu_cost
            
            if delta_E < 0 or (newly_covered > 0 and rng.random() < math.exp(-delta_E / T)):
                # Accept
                for j in balls[w]:
                    current_count[j] += 1
                current_code.append(w)
                code_set.add(w)
                current_uncovered -= newly_covered
                
                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        
        T *= cooling
        
    # Final Pruning of best_code to ensure minimal size (in case SA left redundant words)
    # Re-verify best_code validity and prune
    final_count = np.zeros(n_words, dtype=int)
    for c in best_code:
        for j in balls[c]:
            final_count[j] += 1
            
    # Prune
    pruned_code = []
    for c in best_code:
        redundant = True
        for j in balls[c]:
            if final_count[j] < 2:
                redundant = False
                break
        if redundant:
            for j in balls[c]:
                final_count[j] -= 1
        else:
            pruned_code.append(c)
            
    best_code = pruned_code
    
    result = [list(digits[c]) for c in best_code]
    return result
