import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (total_aps, 4)

    # Map each position to the indices of APs that contain it
    pos_to_ap_indices = [[] for _ in range(N)]
    for i in range(total_aps):
        for j in range(4):
            pos_to_ap_indices[aps[i, j]].append(i)
    
    # Convert to numpy arrays for faster access if needed, but lists are fine for small N
    pos_to_ap_indices = [np.array(lst, dtype=np.int32) for lst in pos_to_ap_indices]

    def count_mono(colors):
        c = colors[aps]
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def get_delta(colors, flip_positions):
        """
        Calculate the change in monochromatic AP count if bits in flip_positions are flipped.
        flip_positions: list or array of indices to flip.
        """
        if not flip_positions:
            return 0
        
        # Find all APs that contain any of the flip positions
        affected_ap_set = set()
        for p in flip_positions:
            affected_ap_set.update(pos_to_ap_indices[p].tolist())
        
        if not affected_ap_set:
            return 0
            
        affected_aps = np.array(list(affected_ap_set), dtype=np.int32)
        
        # Get current colors of these APs
        old_aps_colors = colors[aps[affected_aps]]
        
        # Create a copy to apply flips
        new_aps_colors = old_aps_colors.copy()
        
        # Apply flips
        # We need to know which column in new_aps_colors corresponds to which flip position
        # This is a bit tricky because different APs might have the flip position at different indices (0,1,2,3)
        # So we iterate over the flip positions and update the specific cells in new_aps_colors
        
        # Optimization: Precompute for each AP and each position in the AP, the index in the AP row?
        # Actually, for small N and few flips, iterating is fine.
        
        # Let's map flip position to its index in each affected AP
        # This is complex. Simpler: 
        # For each affected AP row, check which of its 4 elements are in flip_positions.
        
        # Vectorized approach for flipping:
        # Create a mask of shape (len(affected_aps), 4) where True means "flip this bit"
        flip_mask = np.zeros((len(affected_aps), 4), dtype=bool)
        for p in flip_positions:
            # Find where p appears in the affected_aps rows
            # We can do this by broadcasting or simple loops since len(flip_positions) is small (1-3)
            # Let's use a simple loop for clarity and speed given small size
            for k in range(len(affected_aps)):
                ap_row = aps[affected_aps[k]]
                for j in range(4):
                    if ap_row[j] == p:
                        flip_mask[k, j] = True
                        break
        
        # Apply flip: XOR with 1 where mask is True
        new_aps_colors[flip_mask] ^= 1
        
        # Count monochromatic APs before and after
        old_mono = (old_aps_colors.max(axis=1) == old_aps_colors.min(axis=1)).sum()
        new_mono = (new_aps_colors.max(axis=1) == new_aps_colors.min(axis=1)).sum()
        
        return int(new_mono - old_mono)

    def hill_climb(colors):
        current_count = count_mono(colors)
        improved = True
        while improved:
            improved = False
            for p in range(N):
                delta = get_delta(colors, [p])
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
        return current_count

    def simulated_annealing(colors, max_steps=50000, T_start=2.0, T_end=0.05, cooling=0.999):
        current_count = count_mono(colors)
        T = T_start
        best_count = current_count
        best_colors = colors.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Choose number of bits to flip
            r = random.random()
            if r < 0.6:
                k = 1
            elif r < 0.85:
                k = 2
            else:
                k = 3
            
            flip_positions = random.sample(range(N), k)
            delta = get_delta(colors, flip_positions)
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                for p in flip_positions:
                    colors[p] ^= 1
                current_count += delta
                
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            
            T = max(T_end, T * cooling)
            
        return best_count, best_colors

    best_colors = None
    best_count = float('inf')

    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Initialization strategies
    inits = []
    
    # 1. Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Count set bits in i, mod 2
        tm[i] = bin(i).count('1') % 2
    inits.append(tm)
    
    # 2. Random
    inits.append(np.random.randint(0, 2, size=N, dtype=np.int8))
    
    # 3. Block patterns
    for block_size in [2, 4, 8, 16, 32]:
        block = np.zeros(N, dtype=np.int8)
        for i in range(N):
            block[i] = (i // block_size) % 2
        inits.append(block)
        
    # 4. Checkerboard (alternating)
    checker = np.array([i % 2 for i in range(N)], dtype=np.int8)
    inits.append(checker)
    
    # 5. Random with some structure (e.g., random permutation of 0s and 1s)
    perm = np.random.permutation(N)
    rand_struct = np.zeros(N, dtype=np.int8)
    rand_struct[perm[:N//2]] = 0
    rand_struct[perm[N//2:]] = 1
    inits.append(rand_struct)

    for init in inits:
        if time.time() > deadline - 10:
            break
            
        colors = init.copy()
        
        # Phase 1: Hill Climb
        count = hill_climb(colors)
        update_best(colors, count)
        
        # Phase 2: Simulated Annealing
        if time.time() > deadline - 20:
            continue
            
        sa_count, sa_colors = simulated_annealing(colors, max_steps=20000, T_start=2.0, T_end=0.05, cooling=0.9995)
        update_best(sa_colors, sa_count)
        
        # Phase 3: Hill Climb again
        if time.time() > deadline - 10:
            continue
            
        final_count = hill_climb(sa_colors)
        update_best(sa_colors, final_count)

    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
