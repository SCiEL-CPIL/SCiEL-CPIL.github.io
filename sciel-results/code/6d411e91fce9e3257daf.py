import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Thue-Morse initialization
    # colors[i] corresponds to integer i+1
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    # Precompute 4-APs
    # AP: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= n
    # 0-indexed: (i, i+d, i+2d, i+3d) with 0 <= i and i+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Precompute neighbors for each position
    # neighbors[i] = list of (ap_idx, o1, o2, o3)
    neighbors = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for k in range(4):
            i = ap[k]
            others = [ap[j] for j in range(4) if j != k]
            neighbors[i].append((idx, others[0], others[1], others[2]))
    
    def count_mono(col):
        cnt = 0
        for ap in aps:
            c0 = col[ap[0]]
            if c0 == col[ap[1]] == col[ap[2]] == col[ap[3]]:
                cnt += 1
        return cnt
    
    best_score = count_mono(colors)
    best_colors = colors.copy()
    
    # Local Search
    def local_search(col, time_limit):
        col = col.copy()
        is_mono = np.zeros(num_aps, dtype=bool)
        for idx, ap in enumerate(aps):
            c0 = col[ap[0]]
            is_mono[idx] = (c0 == col[ap[1]] == col[ap[2]] == col[ap[3]])
        
        current_score = np.sum(is_mono)
        
        # Steepest Descent
        while time.time() < time_limit:
            best_delta = 0
            best_pos = -1
            
            # Evaluate all positions
            for i in range(n):
                c_new = 1 - col[i]
                delta = 0
                for ap_idx, o1, o2, o3 in neighbors[i]:
                    if is_mono[ap_idx]:
                        delta -= 1
                    else:
                        if col[o1] == c_new and col[o2] == c_new and col[o3] == c_new:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = i
            
            if best_pos == -1:
                break
            
            # Apply flip
            col[best_pos] = 1 - col[best_pos]
            c_new = col[best_pos]
            
            # Update is_mono
            for ap_idx, o1, o2, o3 in neighbors[best_pos]:
                if col[o1] == c_new and col[o2] == c_new and col[o3] == c_new:
                    is_mono[ap_idx] = True
                else:
                    is_mono[ap_idx] = False
            
            current_score += best_delta
            
            if current_score < best_score:
                best_score = current_score
                best_colors = col.copy()
                
        return current_score, col

    # Main loop with restarts
    while time.time() < deadline:
        # Run local search
        _, col = local_search(colors, deadline - 5.0)
        
        if np.sum((col == best_colors)) == n:
             # No improvement or stuck, try perturbation
             pass
        
        # Perturb the best known solution
        if time.time() < deadline - 10.0:
            colors = best_colors.copy()
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                i = random.randint(0, n - 1)
                colors[i] = 1 - colors[i]
        
    return [int(c) for c in best_colors]
