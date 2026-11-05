import time
import random
import numpy as np

def count_mono_ap(colors):
    """
    Count monochromatic 4-term arithmetic progressions.
    colors: list or array of 0/1
    """
    n = len(colors)
    count = 0
    # d max is (n-1)//3
    for d in range(1, (n - 1) // 3 + 1):
        # a max is n - 3*d
        # Vectorized check
        # indices: a, a+d, a+2d, a+3d
        # We can slice
        # c[0:n-3d] == c[d:n-2d] == c[2d:n-d] == c[3d:n]
        # This creates 4 arrays of size n-3d.
        # Sum of (A==B) & (B==C) & (C==D)
        A = colors[:n-3*d]
        B = colors[d:n-2*d]
        C = colors[2*d:n-d]
        D = colors[3*d:n]
        count += np.sum((A == B) & (B == C) & (C == D))
    return count

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = float('inf')
    best_colors = None
    
    # Try multiple initializations
    # 1. Thue-Morse
    # 2. Random
    # 3. Periodic patterns
    
    initializers = []
    
    # Thue-Morse
    tm = [bin(i).count('1') % 2 for i in range(n)]
    initializers.append(tm)
    
    # Random
    for _ in range(5):
        initializers.append([random.randint(0, 1) for _ in range(n)])
        
    # Periodic 0011
    initializers.append([(i // 2) % 2 for i in range(n)])
    
    for init in initializers:
        if time.time() > deadline:
            break
            
        colors = np.array(init, dtype=np.int8)
        current_score = count_mono_ap(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize order of indices to escape symmetry
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Calculate delta for flipping i
                # Old color c, New color 1-c
                c_old = colors[i]
                c_new = 1 - c_old
                
                delta = 0
                max_d = (n - 1) // 3
                
                # Case 1: i is a (start)
                # AP: i, i+d, i+2d, i+3d
                # Valid if i + 3d < n
                for d in range(1, max_d + 1):
                    if i + 3 * d >= n:
                        break
                    idx = [i, i + d, i + 2 * d, i + 3 * d]
                    # Check old
                    old_vals = colors[idx]
                    old_mono = 1 if len(set(old_vals)) == 1 else 0
                    # Check new (replace c_old with c_new at pos i)
                    # Since only i changes, we can check if others are c_new and c_old was c_new?
                    # Simpler: 
                    # New mono is 1 if colors[idx[1:]] are all c_new AND c_new == c_new (trivial)
                    # i.e., colors[idx[1]], colors[idx[2]], colors[idx[3]] must all equal c_new
                    new_mono = 1 if (colors[idx[1]] == c_new and colors[idx[2]] == c_new and colors[idx[3]] == c_new) else 0
                    delta += new_mono - old_mono
                
                # Case 2: i is a+d
                # AP: i-d, i, i+d, i+2d
                for d in range(1, max_d + 1):
                    if i - d < 0 or i + 2 * d >= n:
                        continue
                    idx = [i - d, i, i + d, i + 2 * d]
                    old_vals = colors[idx]
                    old_mono = 1 if len(set(old_vals)) == 1 else 0
                    # New: others must be c_new
                    new_mono = 1 if (colors[idx[0]] == c_new and colors[idx[2]] == c_new and colors[idx[3]] == c_new) else 0
                    delta += new_mono - old_mono
                
                # Case 3: i is a+2d
                # AP: i-2d, i-d, i, i+d
                for d in range(1, max_d + 1):
                    if i - 2 * d < 0 or i + d >= n:
                        continue
                    idx = [i - 2 * d, i - d, i, i + d]
                    old_vals = colors[idx]
                    old_mono = 1 if len(set(old_vals)) == 1 else 0
                    new_mono = 1 if (colors[idx[0]] == c_new and colors[idx[1]] == c_new and colors[idx[3]] == c_new) else 0
                    delta += new_mono - old_mono
                
                # Case 4: i is a+3d
                # AP: i-3d, i-2d, i-d, i
                for d in range(1, max_d + 1):
                    if i - 3 * d < 0:
                        continue
                    idx = [i - 3 * d, i - 2 * d, i - d, i]
                    old_vals = colors[idx]
                    old_mono = 1 if len(set(old_vals)) == 1 else 0
                    new_mono = 1 if (colors[idx[0]] == c_new and colors[idx[1]] == c_new and colors[idx[2]] == c_new) else 0
                    delta += new_mono - old_mono
                
                if delta < 0:
                    colors[i] = c_new
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()

    # Return list of ints
    return best_colors.tolist()
