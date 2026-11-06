import time
import random
import numpy as np

def construct():
    n = 256
    # Deadline: 150s limit, aim for 145s to be safe. 
    # However, the prompt says 120s for the function. 
    # "Each evaluation must finish within 120 s."
    # So deadline is 118s.
    deadline = time.time() + 118.0
    
    # Initialize with Thue-Morse sequence
    # Thue-Morse is a classic 2-coloring that minimizes short APs
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # d >= 1, a+3d <= n
    # Indices are 0-based in numpy, so positions are 0..255
    # APs are (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return color.tolist()
        
    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Helper to count monochromatic 4-APs
    def count_mono(color_vec):
        # color_vec is shape (n,)
        # For each AP, check if all 4 colors are equal
        # c0 == c1 == c2 == c3
        # This is equivalent to (c0==c1) & (c1==c2) & (c2==c3)
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        
        # Count where all are equal
        # Using logical AND
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mask)
    
    current_score = count_mono(color)
    best_color = color.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 50.0
    T_end = 0.1
    # We will run for as long as we have time
    # Estimate cost per step: 
    # count_mono is O(num_aps) which is ~10k. 
    # numpy ops are fast. 
    # Let's say 1000 steps per second is a conservative guess.
    # We can probably do much more.
    
    # Precompute random indices for multi-flips
    # We will use a mix of single flips and small multi-flips (size 2-3)
    
    while time.time() < deadline - 1.0:
        # Choose move size: 70% single, 20% double, 10% triple
        r = random.random()
        if r < 0.7:
            k = 1
        elif r < 0.9:
            k = 2
        else:
            k = 3
            
        # Select k distinct random indices
        # For efficiency, sample without replacement
        if k == 1:
            idxs = np.array([random.randint(0, n - 1)], dtype=np.int32)
        else:
            idxs = np.random.choice(n, k, replace=False)
            
        # Apply flip
        new_color = color.copy()
        new_color[idxs] = 1 - new_color[idxs]
        
        new_score = count_mono(new_color)
        
        delta = new_score - current_score
        
        # Cooling schedule: simple exponential decay based on time or steps?
        # Time-based is safer.
        elapsed = time.time() - (deadline - 118.0) # This is tricky, let's just use a global step count or fraction of time
        # Better: Calculate T based on remaining time
        remaining = deadline - time.time()
        total_time = 118.0
        fraction = max(0.0, remaining / total_time)
        T = T_end + (T_start - T_end) * (fraction ** 2) # Quadratic cooling
        
        if delta < 0:
            # Improvement
            color = new_color
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_color = color.copy()
        else:
            # Accept with probability exp(-delta / T)
            if delta == 0:
                accept = True
            else:
                # If T is very small, exp(-large/T) is 0
                if T < 1e-6:
                    accept = False
                else:
                    # Avoid overflow/underflow
                    if delta / T > 50: 
                        accept = False
                    else:
                        accept = random.random() < np.exp(-delta / T)
                        
            if accept:
                color = new_color
                current_score = new_score
                
    # Return the best found
    return best_color.tolist()

# EVOLVE-BLOCK-END
