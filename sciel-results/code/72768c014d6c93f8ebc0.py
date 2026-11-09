import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 100  # Safety margin before 120s/150s kill
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N
    # Using 0-indexed: 0 <= a, a+3d < N
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps_arr)
    
    # Precompute which APs involve which index for efficient delta calculation
    # We'll store the row indices of the APs for each position
    ap_indices_by_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps_arr):
        for pos in ap:
            ap_indices_by_pos[pos].append(idx)
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))
    
    def thue_morse(n):
        return bin(n).count('1') % 2
        
    best_score = float('inf')
    best_colors = None
    
    # Try a few initializations
    initializations = []
    
    # 1. Thue-Morse
    tm_colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    initializations.append(tm_colors.copy())
    
    # 2. Random
    rng = np.random.default_rng(42)
    initializations.append(rng.integers(0, 2, size=N, dtype=np.int8))
    
    # 3. Thue-Morse with random flips
    for _ in range(5):
        flipped_tm = tm_colors.copy()
        mask = rng.random(N) < 0.1
        flipped_tm[mask] = 1 - flipped_tm[mask]
        initializations.append(flipped_tm)
        
    # 4. Block patterns
    for block_size in [2, 3, 4, 5, 8]:
        block_colors = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)
        initializations.append(block_colors)

    for init_colors in initializations:
        if time.time() - start_time > time_limit:
            break
            
        colors = init_colors.copy()
        current_count = count_mono(colors)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Shuffle order of indices to avoid bias? Or just sequential.
            # Sequential is fine for hill climbing.
            for i in range(N):
                if time.time() - start_time > time_limit:
                    break
                    
                # Calculate delta if we flip i
                # Current contribution of APs involving i
                # We need to check each AP in ap_indices_by_pos[i]
                
                # Optimization: Just try the flip and recount?
                # Recounting is O(M) ~ 14000 ops. N=256. 256*14000 = 3.6M ops/pass.
                # This is very fast in NumPy.
                
                colors[i] = 1 - colors[i]
                new_count = count_mono(colors)
                
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                else:
                    colors[i] = 1 - colors[i] # revert
                    
        if current_count < best_score:
            best_score = current_count
            best_colors = colors.copy()
            
    # Final check
    if best_colors is None:
        best_colors = initializations[0]
        
    return [int(c) for c in best_colors]
