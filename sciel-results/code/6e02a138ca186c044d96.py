import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, which APs contain it and at what position
    # index_aps[idx] = list of (ap_index, position)
    index_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for j in range(4):
            idx = aps[i, j]
            index_aps[idx].append((i, j))
    
    # Convert to numpy arrays
    index_aps_arr = []
    for i in range(N):
        if index_aps[i]:
            arr = np.array(index_aps[i], dtype=np.int32)
            index_aps_arr.append(arr)
        else:
            index_aps_arr.append(np.array([], dtype=np.int32).reshape(0, 2))
    
    # Function to calculate full score
    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))
    
    # Fast delta computation: only re-evaluate affected APs
    def compute_delta(colors, idx):
        pairs = index_aps_arr[idx]
        if len(pairs) == 0:
            return 0
        
        ap_indices = pairs[:, 0]
        positions = pairs[:, 1]
        
        # Get current colors for all 4 positions in affected APs
        old_c0 = colors[aps[ap_indices, 0]]
        old_c1 = colors[aps[ap_indices, 1]]
        old_c2 = colors[aps[ap_indices, 2]]
        old_c3 = colors[aps[ap_indices, 3]]
        
        # Which were monochromatic before?
        old_mono = (old_c0 == old_c1) & (old_c1 == old_c2) & (old_c2 == old_c3)
        old_count = int(np.sum(old_mono))
        
        # Create new color arrays with the flip applied
        new_c0 = old_c0.copy()
        new_c1 = old_c1.copy()
        new_c2 = old_c2.copy()
        new_c3 = old_c3.copy()
        
        # Apply flip at the correct position for each affected AP
        for pos in range(4):
            mask = (positions == pos)
            if np.any(mask):
                if pos == 0:
                    new_c0[mask] = 1 - new_c0[mask]
                elif pos == 1:
                    new_c1[mask] = 1 - new_c1[mask]
                elif pos == 2:
                    new_c2[mask] = 1 - new_c2[mask]
                elif pos == 3:
                    new_c3[mask] = 1 - new_c3[mask]
        
        new_mono = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count
    
    best_score = float('inf')
    best_colors = None
    
    # Multiple random restarts with SA
    num_restarts = 6
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 10.0)
        
        # Random initial coloring
        colors = np.random.randint(0, 2, N).astype(np.int32)
        current_score = get_score(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
        
        # Simulated Annealing
        T = 8.0
        cooling_factor = 0.996
        min_T = 0.05
        
        while time.time() < restart_deadline and T > min_T:
            idx = np.random.randint(0, N)
            
            delta = compute_delta(colors, idx)
            
            if delta <= 0:
                old_val = colors[idx]
                colors[idx] = 1 - old_val
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
            else:
                if random.random() < np.exp(-delta / T):
                    old_val = colors[idx]
                    colors[idx] = 1 - old_val
                    current_score += delta
            
            T *= cooling_factor
    
    # Final greedy local search on best solution
    if best_colors is not None:
        colors = best_colors.copy()
        current_score = get_score(colors)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for idx in range(N):
                if time.time() >= deadline - 3.0:
                    break
                delta = compute_delta(colors, idx)
                if delta < 0:
                    colors[idx] = 1 - colors[idx]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
