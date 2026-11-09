import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps_list.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps_list, dtype=np.int32)
    M = len(aps_arr)
    
    # Helper to count monochromatic APs
    def count_mono_vec(c):
        # c is shape (n,)
        # c[aps_arr] is shape (M, 4)
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
    
    best_score = float('inf')
    best_colors = None
    
    # Multiple restarts to escape local minima
    # We use random initializations and one Thue-Morse
    initializations = []
    
    # Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
    initializations.append(np.array([thue_morse(i) for i in range(n)], dtype=np.int8))
    
    # Random initializations
    np.random.seed(42)
    for _ in range(20):
        initializations.append(np.random.randint(0, 2, n, dtype=np.int8))
        
    for init_colors in initializations:
        if time.time() > deadline:
            break
            
        colors = init_colors.copy()
        current_score = count_mono_vec(colors)
        
        # Greedy Coordinate Descent
        # We iterate until no improvement is found
        # To speed up, we compute delta for all positions at once.
        # Delta for flipping position i:
        # For each AP containing i, check if it becomes mono or stops being mono.
        
        # Efficient vectorized delta calculation:
        # Let's precompute the structure of APs to allow fast delta calculation.
        # Actually, simply recomputing the whole score is O(M) which is ~10k ops.
        # M = sum_{d=1}^{85} (256 - 3d) = 85*256 - 3*(85*86/2) = 21760 - 8715 = 13045.
        # So count_mono_vec is very fast.
        # We can just do:
        # 1. Compute score for current colors.
        # 2. For each i, flip i, compute new score, if better, keep flip.
        # This is O(n * M) per full sweep. 256 * 13000 = 3.3M ops.
        # NumPy can handle this easily in <1ms per sweep.
        # So we can do many sweeps.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate through positions. 
            # To be smarter, we can shuffle the order to avoid bias.
            indices = np.random.permutation(n)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                old_val = colors[i]
                new_val = 1 - old_val
                colors[i] = new_val
                
                new_score = count_mono_vec(colors)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    colors[i] = old_val
            
            # If we finished a full sweep with no improvement, we are at a local minimum.
            # We can break and try next restart.
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
            # If we found a very good score, we might want to continue with SA from here?
            # For now, let's just rely on multiple GCD restarts.
            # If time permits, we could run a short SA phase from the best GCD result.
            
    # If we have time left, run a short SA from the best GCD result to try to escape the local minimum
    if best_colors is not None and time.time() < deadline - 5:
        colors = best_colors.copy()
        current_score = best_score
        
        # Precompute pos_others for SA delta
        pos_others = [[] for _ in range(n)]
        for idx, ap in enumerate(aps_arr):
            for pos_idx in range(4):
                i = ap[pos_idx]
                others = [ap[p] for p in range(4) if p != pos_idx]
                pos_others[i].append(others)
                
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline:
            if T < T_min:
                break
                
            i = np.random.randint(n)
            
            delta = 0
            old_val = colors[i]
            new_val = 1 - old_val
            
            for others in pos_others[i]:
                j, k, l = others
                c_j, c_k, c_l = colors[j], colors[k], colors[l]
                
                old_mono = 1 if (old_val == c_j and c_j == c_k and c_k == c_l) else 0
                new_mono = 1 if (new_val == c_j and c_j == c_k and c_k == c_l) else 0
                
                delta += (new_mono - old_mono)
                
            if delta < 0:
                colors[i] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
            else:
                if delta == 0:
                    if np.random.random() < 0.5:
                        colors[i] = new_val
                else:
                    r = np.random.random()
                    if r < np.exp(-delta / T):
                        colors[i] = new_val
                        current_score += delta
                        
            T *= cooling_rate
            
    return [int(c) for c in best_colors]
