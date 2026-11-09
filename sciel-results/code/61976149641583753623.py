import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256

    # Precompute all 4-term APs: (i0, i1, i2, i3) 0-indexed
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
    
    # Build per-position list of AP indices
    # pos_aps[i] = list of AP indices where position i appears
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    # Precompute "others" for each position to speed up flip_delta
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p] # shape (K, 4)
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)

    # Count monochromatic APs for a coloring (0/1 array)
    def count_mono(color):
        # For each AP, check if all 4 positions have same color
        # color[aps_arr[k]] gives colors at the 4 positions
        cols = color[aps_arr]  # shape (M, 4)
        # Monochromatic if all 4 are equal: min==max
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    # Compute delta in mono count if we flip position p
    # For each AP containing p, check if flipping p makes it mono or breaks mono
    def flip_delta(color, p):
        """Return change in mono count if we flip color[p]."""
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = color[p]
        c_new = 1 - c_p
        # Get colors of the other 3 positions for all APs containing p
        others_colors = color[others] # shape (K, 3)
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))
    
    # Thue-Morse initial coloring
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits in i
            color[i] = bin(i).count('1') % 2
        return color
    
    # Generate several diverse starting points
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color
    
    # Local search: hill climbing with first-improvement (randomized order)
    def local_search(color, max_iter=5000):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            # Randomize the order of positions to try
            perm = np.random.permutation(n)
            for p in perm:
                d = flip_delta(color, p)
                if d < 0:
                    color[p] = 1 - color[p]
                    current_score += d
                    improved = True
                    break
            if not improved:
                break
        return color, current_score
    
    # Iterated local search
    def iterated_local_search(color, max_iterations=200):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        
        for it in range(max_iterations):
            # Check time
            if time.time() - start_time > deadline - 5:
                break
            
            # Local search
            color, score = local_search(color, max_iter=3000)
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
            
            # Perturbation: flip random subset of positions
            # Adaptive perturbation: if we are stuck, flip more
            num_flips = np.random.randint(5, 20)
            flip_positions = np.random.choice(n, size=num_flips, replace=False)
            color[flip_positions] = 1 - color[flip_positions]
        
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(8):
        startings.append(random_coloring())
    
    # Also try Thue-Morse with random perturbations
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    for start_color in startings:
        if time.time() - start_time > deadline - 10:
            break
        
        # Run iterated local search
        # Increased iterations due to speedup
        color, score = iterated_local_search(start_color, max_iterations=100)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # If we have time, do more iterations on the best
    if time.time() - start_time < deadline - 20 and best_overall_color is not None:
        color, score = iterated_local_search(best_overall_color, max_iterations=300)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
