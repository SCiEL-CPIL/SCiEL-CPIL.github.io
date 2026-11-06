import time
import random
import numpy as np

def construct():
    n = 256
    # Deadline: 120s limit, aim for 115s to be safe.
    deadline = time.time() + 115.0
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Indices are 0-based in numpy, so positions are 0..255
    # APs are (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return [0] * n
        
    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Helper to count monochromatic 4-APs (Full count, O(N_aps))
    def count_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mask))
    
    # Initialize with Thue-Morse sequence
    # Thue-Morse is a classic 2-coloring that minimizes short APs
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    current_score = count_mono(color)
    best_color = color.copy()
    best_score = current_score
    
    # Precompute reverse mapping: for each index i, which APs contain i?
    # This allows O(1) delta calculation for a single flip.
    # ap_indices[i] is a list of AP indices where i appears.
    # Also store the position of i within that AP (0, 1, 2, or 3) to update logic.
    # Actually, for a flip at index i, the delta depends on the colors of the other 3 elements in each AP containing i.
    
    # Let's precompute for each index i:
    # 1. List of AP indices involving i
    # 2. For each such AP, the indices of the other 3 elements
    
    # This might be memory heavy if done naively, but n=256 is small.
    # num_aps is approx sum_{d=1}^{85} (256-3d) ~ 85*256 - 3*85*86/2 ~ 21760 - 8655 ~ 13000.
    # Each AP has 4 indices. Total occurrences = 4 * 13000 = 52000.
    # So on average, each index is in ~200 APs.
    
    # Structure: 
    # affected_aps[i] = list of ap_idx
    # For each ap_idx, we need to know the other 3 positions.
    # We can just look up ap_array[ap_idx] and mask out i.
    
    affected_aps = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        a, b, c, d = ap_array[ap_idx]
        affected_aps[a].append(ap_idx)
        affected_aps[b].append(ap_idx)
        affected_aps[c].append(ap_idx)
        affected_aps[d].append(ap_idx)
    
    # Function to calculate delta for flipping index i
    def calc_delta(i, color_vec, current_score):
        # Flip color at i
        old_c = color_vec[i]
        new_c = 1 - old_c
        
        delta = 0
        for ap_idx in affected_aps[i]:
            # Get the 4 indices of this AP
            idxs = ap_array[ap_idx]
            
            # Get current colors
            c0 = color_vec[idxs[0]]
            c1 = color_vec[idxs[1]]
            c2 = color_vec[idxs[2]]
            c3 = color_vec[idxs[3]]
            
            # Check if this AP is currently monochromatic
            is_mono_before = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            # Simulate flip: replace color at i with new_c
            # We need to check if the new set of colors is monochromatic
            # The other 3 colors are fixed. Let's call them o1, o2, o3.
            # The new colors are new_c and o1, o2, o3 (in some order).
            # For the AP to be monochromatic AFTER the flip, all 4 must be equal.
            # This implies:
            # 1. new_c must equal o1, o2, and o3.
            # 2. o1, o2, o3 must all be equal to each other.
            
            # Let's extract the other 3 indices
            other_idxs = [idxs[j] for j in range(4) if idxs[j] != i]
            o1 = color_vec[other_idxs[0]]
            o2 = color_vec[other_idxs[1]]
            o3 = color_vec[other_idxs[2]]
            
            # Check if the other 3 are equal
            others_equal = (o1 == o2) and (o2 == o3)
            
            # Check if new_c equals them
            new_c_matches = (new_c == o1)
            
            is_mono_after = others_equal and new_c_matches
            
            if is_mono_before and not is_mono_after:
                delta -= 1 # We removed a mono AP (good)
            elif not is_mono_before and is_mono_after:
                delta += 1 # We added a mono AP (bad)
            
        return delta

    # Iterated Local Search
    # 1. Start with Thue-Morse
    # 2. Perform a local search phase (Hill Climbing) until no improvement is found
    # 3. Perturb the best solution
    # 4. Repeat
    
    # To make it faster, we can use a "best-improvement" or "first-improvement" strategy.
    # With O(1) delta, we can evaluate many moves.
    # However, scanning all 256 indices to find the best move is O(256 * avg_degree) ~ O(256 * 200) = 50k ops per step.
    # This is still fast in Python if optimized, but let's try random sampling of moves in the local search phase to keep it simple and fast enough.
    # Or, since we have 115s, we can afford a more thorough local search.
    
    # Let's do a local search that tries all 256 moves, calculates delta, and picks the best one if it's an improvement.
    # If no improvement, stop local search.
    
    def local_search(color_vec, current_score):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_idx = -1
            
            # Try all indices
            for i in range(n):
                d = calc_delta(i, color_vec, current_score)
                if d < best_delta:
                    best_delta = d
                    best_idx = i
            
            if best_idx != -1 and best_delta < 0:
                # Apply flip
                color_vec[best_idx] = 1 - color_vec[best_idx]
                current_score += best_delta
                improved = True
                # Check time
                if time.time() > deadline - 5.0:
                    break
                    
        return color_vec, current_score

    # Main Loop
    while time.time() < deadline - 5.0:
        # Phase 1: Local Search
        color, current_score = local_search(color, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()
        
        # Phase 2: Perturbation
        # Randomly flip a few bits to escape local minimum
        # Number of flips: 2-5
        num_flips = random.randint(2, 5)
        for _ in range(num_flips):
            idx = random.randint(0, n - 1)
            color[idx] = 1 - color[idx]
        
        # Re-calculate score after perturbation (O(N_aps))
        current_score = count_mono(color)
        
    return best_color.tolist()
