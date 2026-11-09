import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
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
    def count_mono(c):
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
    
    def thue_morse(i):
        return bin(i).count('1') % 2

    # Generate diverse initial colorings
    initial_colors_list = []
    
    # Thue-Morse
    initial_colors_list.append(np.array([thue_morse(i) for i in range(n)], dtype=np.int8))
    
    # Randoms
    for _ in range(5):
        initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))
        
    # Structured
    initial_colors_list.append(np.array([i % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8))
    
    best_score_global = float('inf')
    best_colors_global = None

    def run_gcd(initial_colors, seed_idx):
        nonlocal best_score_global, best_colors_global
        np.random.seed(100 + seed_idx)
        colors = initial_colors.copy()
        current_score = count_mono(colors)
        
        best_score_local = current_score
        best_colors_local = colors.copy()
        
        # GCD Loop
        # We will run multiple rounds of GCD.
        # In each round, we compute delta for all bits.
        # If any delta < 0, we flip all such bits.
        # We repeat until no more beneficial flips or time runs out.
        
        max_gcd_rounds = 5000 # Cap to prevent infinite loop if stuck in cycle (unlikely with strict improvement)
        
        while time.time() < deadline:
            # Compute delta for flipping each bit
            # Delta_i = (new_count - old_count)
            # For each AP, if it is mono, flipping a bit in it breaks it (-1).
            # If it is not mono, flipping a bit might make it mono (+1).
            
            # Vectorized approach:
            # 1. Identify all mono APs.
            # 2. For each position i, count how many mono APs contain i. (These will be broken)
            # 3. Identify all non-mono APs that are "one flip away" from being mono.
            #    A non-mono AP is one flip away if it has exactly 3 of one color and 1 of the other, 
            #    AND the single different color is at the position we are considering flipping?
            #    Actually, simpler: 
            #    For a given AP, let the colors be c0, c1, c2, c3.
            #    If we flip bit at index k in this AP, the new colors are c0', c1', c2', c3'.
            #    It becomes mono if c0'=c1'=c2'=c3'.
            
            # Efficient Delta Calculation:
            # Let c be the current coloring.
            # For each AP j, let m_j be 1 if mono, 0 otherwise.
            # If we flip bit i, the change in score is:
            # Sum over all APs j containing i of: (m_j_after - m_j_before)
            
            # We can precompute for each position i, the list of APs it belongs to.
            # But vectorized over all i at once is tricky without sparse matrices or loops.
            # However, n=256 is small. We can do this:
            
            # 1. Compute current mono status for all APs.
            mono_status = (c := colors)[aps_arr[:,0]] == c[aps_arr[:,1]]
            mono_status &= c[aps_arr[:,1]] == c[aps_arr[:,2]]
            mono_status &= c[aps_arr[:,2]] == c[aps_arr[:,3]]
            
            # 2. For each position i, we want to know:
            #    - How many mono APs contain i? (These contribute -1 if flipped)
            #    - How many non-mono APs contain i such that flipping i makes them mono? (These contribute +1)
            
            # Let's build a matrix or use broadcasting.
            # aps_arr is (M, 4).
            # We can create a boolean matrix "contains" of shape (n, M) where contains[i, j] is True if i is in AP j.
            # This is 256 * M. M is approx 256*256/4 / 2? No, sum of (256-3d) for d=1..85.
            # Approx 256 * 85 / 2 * 0.8 ~ 8000.
            # 256 * 8000 = 2M booleans. This is fine.
            
            # Precompute "contains" matrix once?
            # It's better to precompute it outside the loop.
            
            if not hasattr(run_gcd, 'contains_matrix'):
                contains = np.zeros((n, M), dtype=bool)
                for k in range(4):
                    # For each position index k in the AP tuple, set contains[position, ap_idx] = True
                    # We can use np.add.at or just loop over the 4 columns
                    positions = aps_arr[:, k]
                    ap_indices = np.arange(M)
                    contains[positions, ap_indices] = True
                run_gcd.contains_matrix = contains
                
            contains = run_gcd.contains_matrix
            
            # Count mono APs containing each position
            # mono_status is (M,)
            # contains is (n, M)
            # mono_count_per_pos[i] = sum(mono_status[contains[i, :]])
            # Using matrix multiplication: contains @ mono_status
            mono_count_per_pos = contains @ mono_status
            
            # Now, for the +1 part:
            # An AP j is non-mono (mono_status[j] == False).
            # Flipping position i makes AP j mono IF:
            # 1. AP j contains i.
            # 2. After flipping i, all 4 elements in AP j are equal.
            
            # Let's analyze the condition for AP j becoming mono after flipping i.
            # Let the values in AP j be v0, v1, v2, v3.
            # Suppose i is at index k in AP j.
            # New values: v0', v1', v2', v3' where vk' = 1-vk, others unchanged.
            # They are all equal if and only if:
            # - The 3 unchanged values are all equal to each other.
            # - The new value vk' is equal to that common value.
            #   => vk' == v_uncommon
            #   => 1 - vk == v_uncommon
            #   => vk != v_uncommon
            
            # So, for AP j to become mono by flipping i (at index k):
            # - The other 3 values must be equal.
            # - The value at i must be different from them.
            
            # This means AP j must have exactly 3 of one color and 1 of the other, 
            # AND the single different one is at position i.
            
            # We can precompute for each AP:
            # - Does it have 3 of one color and 1 of the other?
            # - If so, which position is the "odd one out"?
            
            # Let's compute this vectorized.
            # For each AP j, get the 4 values.
            vals = colors[aps_arr] # Shape (M, 4)
            
            # Count 0s and 1s in each AP
            num_zeros = np.sum(vals == 0, axis=1)
            num_ones = np.sum(vals == 1, axis=1)
            
            # Is it 3-1 split?
            is_3_1 = (num_zeros == 3) | (num_ones == 3)
            
            # If is_3_1, which position is the odd one?
            # If num_zeros == 3, the odd one is the 1.
            # If num_ones == 3, the odd one is the 0.
            
            # We want a matrix "would_become_mono_if_flipped" of shape (n, M)
            # would_become_mono_if_flipped[i, j] = True if flipping i makes AP j mono.
            
            # This is equivalent to:
            # For AP j, if it is 3-1, let k be the index of the odd one.
            # Then flipping the position corresponding to index k in AP j makes it mono.
            # The position in the global array is aps_arr[j, k].
            
            # So, for each AP j that is 3-1, we find the global position p = aps_arr[j, k].
            # Then we set would_become_mono_if_flipped[p, j] = True.
            
            # Let's find k for each AP.
            # If num_zeros == 3, the odd one is where vals == 1.
            # If num_ones == 3, the odd one is where vals == 0.
            
            # We can create a mask for the odd position.
            # odd_mask[j, k] is True if k is the odd position for AP j.
            
            # Initialize odd_pos_indices to -1
            odd_pos_indices = np.full(M, -1, dtype=np.int32)
            
            # Case 1: 3 zeros, 1 one. Odd one is the 1.
            mask_3z = (num_zeros == 3)
            if np.any(mask_3z):
                # Find index of 1 in vals[mask_3z]
                # vals is (M, 4). We want the column index where val is 1.
                # Since there is exactly one 1, we can use argmax on (vals == 1)
                odd_1_indices = np.argmax(vals[mask_3z] == 1, axis=1)
                odd_pos_indices[mask_3z] = odd_1_indices
                
            # Case 2: 3 ones, 1 zero. Odd one is the 0.
            mask_3o = (num_ones == 3)
            if np.any(mask_3o):
                odd_0_indices = np.argmax(vals[mask_3o] == 0, axis=1)
                odd_pos_indices[mask_3o] = odd_0_indices
            
            # Now, for each AP j where odd_pos_indices[j] != -1:
            # The global position to flip is p = aps_arr[j, odd_pos_indices[j]]
            # We mark would_become_mono_if_flipped[p, j] = True.
            
            # We can do this with advanced indexing.
            valid_aps = odd_pos_indices != -1
            if np.any(valid_aps):
                ap_indices_valid = np.where(valid_aps)[0]
                pos_indices_valid = odd_pos_indices[valid_aps]
                global_positions = aps_arr[ap_indices_valid, pos_indices_valid]
                
                # Create a sparse-like update
                # We need to set would_become_mono_if_flipped[global_positions, ap_indices_valid] = True
                # Since this is a boolean matrix, we can just assign.
                # But wait, would_become_mono_if_flipped is (n, M).
                # We are setting specific (row, col) pairs.
                
                # We can create a temporary zero matrix and use np.add.at or just direct indexing if we handle duplicates.
                # Direct indexing: matrix[rows, cols] = True works if rows and cols are arrays of same length.
                # It sets all specified pairs to True.
                
                # However, we are re-computing this matrix every iteration.
                # It's better to just compute the count directly.
                # Count of +1s for position i = number of APs j such that:
                # 1. j contains i
                # 2. j is 3-1 split
                # 3. i is the odd one out in j.
                
                # This is exactly the number of True values in would_become_mono_if_flipped[i, :].
                # So we can build this matrix or compute the sum directly.
                
                # Let's build the "would_become_mono" indicator for each AP j.
                # It's a single global position p_j.
                # So for each AP j, it contributes +1 to delta[p_j] and -1 to no other position?
                # No, it contributes +1 to delta[p_j] ONLY.
                # So we can just scatter add 1 to delta[global_positions] for each valid AP.
                
                # Delta_i = (mono_count_per_pos[i]) - (number of 3-1 APs where i is the odd one)
                # Wait, delta = new - old.
                # Breaking a mono AP: -1.
                # Creating a mono AP: +1.
                # So Delta_i = (-1) * (mono_count_per_pos[i]) + (+1) * (creation_count_per_pos[i])
                
                creation_counts = np.zeros(n, dtype=np.int32)
                np.add.at(creation_counts, global_positions, 1)
                
            else:
                creation_counts = np.zeros(n, dtype=np.int32)
            
            deltas = -mono_count_per_pos + creation_counts
            
            # Find positions where delta < 0 (improvement)
            improvement_mask = deltas < 0
            if np.any(improvement_mask):
                # Flip all improving bits
                colors[improvement_mask] = 1 - colors[improvement_mask]
                current_score = int(np.sum(deltas[improvement_mask])) + current_score
                # Note: This sum is correct only if the deltas are independent?
                # Flipping multiple bits at once:
                # If two flipped bits are in the same AP, the delta calculation might be off.
                # However, in GCD, we usually flip sequentially or assume small interactions.
                # With n=256 and random-ish colorings, the probability of two flipped bits being in the same AP is low but non-zero.
                # To be safe, we should recompute the score.
                current_score = count_mono(colors)
                
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_colors_local = colors.copy()
            else:
                # Stagnation: No single bit flip improves.
                # Try a random perturbation (kick)
                # Flip a random subset of bits (e.g., 10-20 bits)
                num_kicks = np.random.randint(5, 20)
                kick_positions = np.random.choice(n, num_kicks, replace=False)
                colors[kick_positions] = 1 - colors[kick_positions]
                current_score = count_mono(colors)
                
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_colors_local = colors.copy()
                
                # Optional: Run a few more GCD steps after kick
                # The loop will continue and try to improve again.
        
        # Update global best
        if best_score_local < best_score_global:
            best_score_global = best_score_local
            best_colors_global = best_colors_local.copy()

    # Run GCD in parallel? 
    # GCD is vectorized and fast. Running multiple instances in parallel might be overkill 
    # or might help escape local minima. 
    # Given the 150s limit, we can run 4-8 instances in threads.
    
    import threading
    lock = threading.Lock()
    
    def run_gcd_threaded(initial_colors, seed_idx):
        run_gcd(initial_colors, seed_idx)
        with lock:
             if best_score_local < best_score_global: # This logic is inside run_gcd already
                 pass # Handled inside run_gcd

    # Actually, let's just run them sequentially if GCD is fast enough, 
    # or use threads if we want to exploit parallelism.
    # Vectorized ops release GIL, so threads are useful.
    
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i, init_c in enumerate(initial_colors_list):
            if time.time() > deadline - 5:
                break
            futures.append(executor.submit(run_gcd, init_c, i))
        for future in futures:
            try:
                future.result()
            except Exception as e:
                pass

    if best_colors_global is None:
        return [int(c) for c in initial_colors_list[0]]
        
    return [int(c) for c in best_colors_global]
