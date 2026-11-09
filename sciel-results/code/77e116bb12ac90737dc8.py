import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Vectorized count function
    def count_mono(c):
        # c is array of 0/1
        # Check if all 4 are equal
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
                      
    # Initial coloring: Random or Thue-Morse? 
    # Thue-Morse is a good start.
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    current_score = count_mono(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Local Search with Block Flips
    # Parameters
    max_iter = 1000000
    T = 5.0
    T_min = 0.05
    cooling_rate = 0.99995
    
    # We will try flipping blocks of size 1 to 8
    # To speed up, we can precompute the APs affected by a block flip?
    # No, just recompute the score for the whole thing. It's vectorized and fast (~1ms for 256 elements).
    # 115s / 1ms = 115,000 iterations. This is much better than SA with slow delta.
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # Pick a random block size and start
        # Bias towards smaller blocks for fine tuning, larger for exploration
        if np.random.random() < 0.5:
            block_size = 1
        else:
            block_size = np.random.randint(2, 9)
            
        max_start = n - block_size
        if max_start <= 0:
            block_size = 1
            max_start = n
            
        start = np.random.randint(max_start)
        
        # Create a copy and flip the block
        new_colors = colors.copy()
        new_colors[start:start+block_size] = 1 - new_colors[start:start+block_size]
        
        new_score = count_mono(new_colors)
        
        # Acceptance
        if new_score < current_score:
            colors = new_colors
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            # Worse or equal
            if new_score == current_score:
                # Accept neutral moves to explore
                if np.random.random() < 0.5:
                    colors = new_colors
            else:
                r = np.random.random()
                if r < np.exp(-(new_score - current_score) / T):
                    colors = new_colors
                    current_score = new_score
                    
        T *= cooling_rate
        
    return [int(c) for c in best_colors]
