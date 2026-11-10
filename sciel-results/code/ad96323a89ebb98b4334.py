import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Generate all possible words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    num_words = len(all_words)
    
    # Pre-compute is_in_ball
    a = all_words[:, None, :]
    b = all_words[None, :, :]
    dist = np.sum(a != b, axis=2)
    is_in_ball = dist <= radius
    
    # Convert to float for faster matrix multiplication
    is_in_ball_float = is_in_ball.astype(np.float32)
    
    best_code = None
    best_len = num_words + 1
    
    # Set a time limit
    start_time = time.time()
    time_limit = 110
    
    rng = np.random.default_rng(42)
    
    # We can run multiple restarts
    # Each restart:
    # 1. Reset covered
    # 2. Loop until covered
    #    a. Compute gains using matmul
    #    b. Pick best (random tie break)
    #    c. Update covered
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(num_words, dtype=bool)
        code = []
        
        while covered.sum() < num_words:
            uncovered = ~covered
            uncovered_float = uncovered.astype(np.float32)
            
            # Compute gains: dot product of is_in_ball_float (N x N) and uncovered_float (N x 1)
            # Result is N x 1 array of gains
            gains = is_in_ball_float @ uncovered_float
            gains = gains.flatten()
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            # Due to float precision, we might want to be careful, but gains are integers 0..13
            # So exact comparison should be fine
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 1:
                best_idx = candidates[rng.integers(len(candidates))]
            else:
                best_idx = candidates[0]
                
            code.append(all_words[best_idx].tolist())
            covered |= is_in_ball[best_idx]
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
    return best_code
