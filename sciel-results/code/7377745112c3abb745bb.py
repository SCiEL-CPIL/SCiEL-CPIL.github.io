import time
import random
import numpy as np

def construct():
    # Deadline: 85 seconds to be safe (evaluator kills at 90s)
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    # Helper to calculate multiplicities using convolution for speed
    # Returns (max_multiplicity, is_valid)
    def calc_mult(A):
        B = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for a in A:
            B[a] = 1
        # Convolution gives ordered pair counts
        conv = np.convolve(B, B, mode='full')
        # conv[t] = number of ordered pairs (a,b) in A x A with a+b=t
        
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        for idx, t in enumerate(range(MIN_SUM, MAX_SUM + 1)):
            ordered = conv[t]
            if ordered == 0:
                return np.inf, False
            
            # Convert ordered to unordered
            if t % 2 == 0:
                half = t // 2
                if B[half] > 0:
                    # unordered = (ordered + 1) / 2
                    # Since ordered must be odd for this to be integer? 
                    # Actually ordered counts (a,b) and (b,a).
                    # If a!=b, they come in pairs. If a==b, only one.
                    # So ordered = 2 * (unordered - delta_{a==b}) + delta_{a==b}
                    # unordered = (ordered + delta_{a==b}) / 2
                    # delta is 1 if half in A, else 0.
                    M[idx] = (ordered + 1) // 2
                else:
                    M[idx] = ordered // 2
            else:
                M[idx] = ordered // 2
        
        if np.min(M) < 1:
            return np.inf, False
            
        return np.max(M), True

    # Initial set: The seed
    A = set(range(64, 129))
    current_score, valid = calc_mult(A)
    
    if not valid:
        # Fallback if seed somehow invalid (shouldn't happen)
        return list(A)
        
    best_score = current_score
    best_A = set(A)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    
    # Pre-compute potential elements for addition/swaps
    ALL_ELEMENTS = set(range(0, 129))
    
    # Track step for cooling if needed, but time-based is fine
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove
            if len(A) <= 10: # Don't make set too small
                continue
            x = random.choice(list(A))
            new_A = set(A)
            new_A.remove(x)
            
            new_score, new_valid = calc_mult(new_A)
            
            if new_valid:
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    current_score = new_score
                    if score_new < best_score:
                        best_score = new_score
                        best_A = set(A)
                else:
                    pass # Reject
            else:
                pass # Reject
                
        elif move_type < 0.8:
            # Swap
            if len(A) == 0:
                continue
            x = random.choice(list(A))
            # Choose y not in A
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_A = set(A)
            new_A.remove(x)
            new_A.add(y)
            
            new_score, new_valid = calc_mult(new_A)
            
            if new_valid:
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_A = set(A)
                else:
                    pass # Reject
            else:
                pass # Reject
                
        else:
            # Add
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_A = set(A)
            new_A.add(y)
            
            new_score, new_valid = calc_mult(new_A)
            
            if new_valid:
                delta = new_score - current_score
                # Adding usually increases max, so delta > 0.
                # Only accept if we are in a high T phase.
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    current_score = new_score
                    # Don't update best here as it's likely worse, but keep for SA
                else:
                    pass # Reject
            else:
                pass # Reject
                
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return list(best_A)
