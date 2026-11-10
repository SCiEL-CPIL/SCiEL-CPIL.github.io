import numpy as np
import time
import random

def get_max_ap_sum(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n // 2 + 1):
        for r in range(d):
            curr_max = 0
            curr_min = 0
            max_sub = 0
            min_sub = 0
            for idx in range(r, n, d):
                val = signs[idx]
                # Update max subarray ending here
                if curr_max < 0:
                    curr_max = val
                else:
                    curr_max += val
                if curr_max > max_sub:
                    max_sub = curr_max
                
                # Update min subarray ending here
                if curr_min > 0:
                    curr_min = val
                else:
                    curr_min += val
                if curr_min < min_sub:
                    min_sub = curr_min
            
            local_max_abs = max(abs(max_sub), abs(min_sub))
            if local_max_abs > max_abs:
                max_abs = local_max_abs
    return max_abs

def construct():
    n = 256
    # Thue-Morse initialization
    signs = [0] * n
    for i in range(n):
        # Count set bits
        signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        
    best_signs = signs[:]
    best_score = get_max_ap_sum(best_signs)
    
    start_time = time.time()
    limit = 55.0 # Leave 5s for safety
    
    # Local search
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < limit:
        # Pick a random index to flip
        idx = random.randint(0, n - 1)
        
        # Flip
        signs[idx] *= -1
        
        # Evaluate
        current_score = get_max_ap_sum(signs)
        
        if current_score < best_score:
            best_score = current_score
            best_signs = signs[:]
            no_improve_count = 0
        else:
            # Revert
            signs[idx] *= -1
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart from Thue-Morse or Random
                # Let's try a few random restarts
                if random.random() < 0.1:
                    signs = [random.choice([-1, 1]) for _ in range(n)]
                    current_score = get_max_ap_sum(signs)
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = signs[:]
                        no_improve_count = 0
                else:
                    # Reset to Thue-Morse
                    signs = [0] * n
                    for i in range(n):
                        signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
                    no_improve_count = 0
                    # Maybe try a few flips from Thue-Morse
                    for _ in range(10):
                        idx = random.randint(0, n - 1)
                        signs[idx] *= -1
                        sc = get_max_ap_sum(signs)
                        if sc < best_score:
                            best_score = sc
                            best_signs = signs[:]
                            no_improve_count = 0
                            break
                        else:
                            signs[idx] *= -1

    return best_signs
