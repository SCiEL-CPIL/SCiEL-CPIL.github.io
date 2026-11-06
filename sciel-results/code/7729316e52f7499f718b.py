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
        # Fallback, though n=256 guarantees APs
        return [0] * n
        
    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build reverse index: for each position i, list of AP indices containing i
    # This allows O(1) delta calculation for a single flip.
    ap_indices_at_pos = [[] for _ in range(n)]
    for idx in range(num_aps):
        a, b, c, d = ap_array[idx]
        ap_indices_at_pos[a].append(idx)
        ap_indices_at_pos[b].append(idx)
        ap_indices_at_pos[c].append(idx)
        ap_indices_at_pos[d].append(idx)
    
    # Helper to count monochromatic 4-APs from scratch (for initialization and verification)
    def count_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mask))
    
    # Helper to check if a specific AP is monochromatic
    def is_mono(ap_idx, color_vec):
        a, b, c, d = ap_array[ap_idx]
        c0 = color_vec[a]
        c1 = color_vec[b]
        c2 = color_vec[c]
        c3 = color_vec[d]
        return (c0 == c1) and (c1 == c2) and (c2 == c3)
    
    # Local Search using Incremental Delta Updates
    def local_search(color_vec, max_steps=10000):
        """
        Perform a local search using greedy single-flip improvements.
        Returns the improved color vector and the number of mono APs.
        """
        current_score = count_mono(color_vec)
        
        # Precompute which APs are currently mono for quick check?
        # Or just check on the fly.
        # To speed up, we can maintain a boolean array 'is_mono_arr'
        # But updating it incrementally is complex.
        # Instead, for each candidate flip, we check the delta.
        
        # For a single flip at index i:
        # Delta = (new_mono_count - old_mono_count)
        # We only need to check APs containing i.
        
        # To make this fast, we iterate through possible flips.
        # A standard ILS approach:
        # 1. Try all single flips, pick the best improving one.
        # 2. If none, stop.
        # This is O(n * avg_aps_per_pos) per step.
        # avg_aps_per_pos is roughly num_aps * 4 / n ~ 10000 * 4 / 256 ~ 150.
        # So 256 * 150 = ~38,000 operations per step.
        # 10000 steps = 380M operations. Might be too slow in Python.
        
        # Optimization: Randomized Greedy / First-Improve
        # Instead of checking all, check a random subset or use a heuristic.
        # However, deterministic "best-improve" is more robust.
        # Let's try to vectorize the delta calculation for all possible single flips at once?
        # That would be expensive to set up.
        
        # Alternative: Use the "First-Improve" strategy with random order.
        # This is faster per step but may require more steps.
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline - 1.0:
                break
                
            best_delta = 0
            best_idx = -1
            best_new_score = current_score
            
            # Iterate through all positions
            # To speed up, we can use numpy to calculate the "potential" change?
            # No, let's stick to Python loops but optimized.
            
            # Actually, let's try a different approach:
            # Calculate the "cost" of each position being 0 vs 1?
            # The objective is sum over APs of [all equal].
            # This is a high-order interaction, so simple additive costs don't work directly.
            
            # Let's use a randomized local search:
            # Pick a random index, flip it if it improves.
            # Repeat until no improvement in a certain number of tries.
            
            tries = 0
            max_tries = 1000
            improved = False
            
            while tries < max_tries:
                i = random.randint(0, n - 1)
                old_color = color_vec[i]
                new_color = 1 - old_color
                
                # Calculate delta
                delta = 0
                for ap_idx in ap_indices_at_pos[i]:
                    # Check if AP was mono before
                    a, b, c, d = ap_array[ap_idx]
                    # Get colors of the other 3 positions
                    # We need to be careful: if i is one of a,b,c,d
                    # Let's just fetch all 4 and compare
                    c0 = color_vec[a]
                    c1 = color_vec[b]
                    c2 = color_vec[c]
                    c3 = color_vec[d]
                    
                    was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    # Apply flip to i
                    if a == i: c0 = new_color
                    elif b == i: c1 = new_color
                    elif c == i: c2 = new_color
                    elif d == i: c3 = new_color
                    
                    is_mono_now = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    if was_mono and not is_mono_now:
                        delta -= 1
                    elif not was_mono and is_mono_now:
                        delta += 1
                    
                    if delta > 0: # Early exit if we already know it's worse
                        break
                
                if delta < 0:
                    # Accept
                    color_vec[i] = new_color
                    current_score += delta
                    improved = True
                    tries = 0 # Reset tries if improved
                else:
                    tries += 1
            
            if not improved:
                break
            steps += 1
            
        return color_vec, current_score

    # Initial states
    # 1. Thue-Morse
    color_tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    # 2. Random
    color_rand = np.random.randint(0, 2, n).astype(np.int8)
    
    # 3. Alternating
    color_alt = np.array([i % 2 for i in range(n)], dtype=np.int8)
    
    # 4. Block-based (e.g., 4s)
    color_block = np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8)
    
    initial_states = [color_tm, color_rand, color_alt, color_block]
    
    best_color = None
    best_score = float('inf')
    
    # Multi-start ILS
    for init_idx, init_color in enumerate(initial_states):
        if time.time() > deadline - 5.0:
            break
            
        current_color = init_color.copy()
        
        # Phase 1: Local Search (Greedy)
        current_color, current_score = local_search(current_color, max_steps=5000)
        
        if current_score < best_score:
            best_score = current_score
            best_color = current_color.copy()
            
        # Phase 2: Perturbation (2-flip) and Local Search again
        # We do several rounds of perturbation
        for round_num in range(20):
            if time.time() > deadline - 2.0:
                break
                
            # Perturb: flip 2 random bits
            if random.random() < 0.8:
                # 2-flip
                i1, i2 = random.sample(range(n), 2)
                current_color[i1] = 1 - current_color[i1]
                current_color[i2] = 1 - current_color[i2]
            else:
                # 3-flip
                i1, i2, i3 = random.sample(range(n), 3)
                current_color[i1] = 1 - current_color[i1]
                current_color[i2] = 1 - current_color[i2]
                current_color[i3] = 1 - current_color[i3]
                
            # Local Search
            current_color, current_score = local_search(current_color, max_steps=2000)
            
            if current_score < best_score:
                best_score = current_score
                best_color = current_color.copy()
                
    # Final verification
    if best_color is None:
        return color_tm.tolist()
        
    return best_color.tolist()

# EVOLVE-BLOCK-END
