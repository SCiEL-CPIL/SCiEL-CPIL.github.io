import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 140
    
    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    def count_mono(colors):
        """Count monochromatic 4-term APs. colors is a numpy array of 0s and 1s."""
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        mono0 = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        mono1 = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        return int(np.sum(mono0) + np.sum(mono1))
    
    def compute_deltas(colors):
        """
        For each position p, compute the change in mono count if we flip colors[p].
        Returns array of length N where deltas[p] = new_count - current_count.
        
        When we flip position p from c to 1-c:
        - APs where p is one of the 4 elements may change their mono status
        - For each such AP, check if it was mono before and after the flip
        """
        current = count_mono(colors)
        
        # For each AP, determine if it's currently monochromatic
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        
        # An AP is mono if all 4 are 0 or all 4 are 1
        all_zero = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        all_one = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        is_mono = all_zero | all_one
        
        # For each position p, flip it and see which APs change
        # Instead of iterating, use vectorized approach:
        # For position p, the APs it belongs to are those where p appears in any of the 4 slots
        
        # Build a mapping: for each AP, which positions are in it
        # aps_array has shape (num_aps, 4)
        
        # For each position p, find all APs containing p
        # This is expensive to do naively, so let's precompute
        
        # Precompute: for each position, list of (ap_idx, role)
        pos_to_aps = [[] for _ in range(N)]
        for ap_idx in range(num_aps):
            for role in range(4):
                pos = aps_array[ap_idx, role]
                pos_to_aps[pos].append((ap_idx, role))
        
        # Convert to arrays for vectorized access
        pos_aps_indices = []
        pos_aps_roles = []
        pos_aps_starts = [0]
        for p in range(N):
            idxs = [x[0] for x in pos_to_aps[p]]
            roles = [x[1] for x in pos_to_aps[p]]
            pos_aps_indices.extend(idxs)
            pos_aps_roles.extend(roles)
            pos_aps_starts.append(len(pos_aps_indices))
        
        pos_aps_indices = np.array(pos_aps_indices, dtype=np.int32)
        pos_aps_roles = np.array(pos_aps_roles, dtype=np.int32)
        
        # For each position, compute delta by checking each AP it belongs to
        deltas = np.zeros(N, dtype=np.int32)
        
        for p in range(N):
            start = pos_aps_starts[p]
            end = pos_aps_starts[p + 1]
            if start == end:
                continue
            
            ap_idxs = pos_aps_indices[start:end]
            roles = pos_aps_roles[start:end]
            
            # For each AP in this list, check if flipping p changes its mono status
            # Get the 4 colors of each AP
            ac0 = colors[aps_array[ap_idxs, 0]]
            ac1 = colors[aps_array[ap_idxs, 1]]
            ac2 = colors[aps_array[ap_idxs, 2]]
            ac3 = colors[aps_array[ap_idxs, 3]]
            
            # Current mono status
            cur_all_zero = (ac0 == 0) & (ac1 == 0) & (ac2 == 0) & (ac3 == 0)
            cur_all_one = (ac0 == 1) & (ac1 == 1) & (ac2 == 1) & (ac3 == 1)
            cur_mono = cur_all_zero | cur_all_one
            
            # After flipping position p (which is in role r for each AP)
            # The color at position p changes from colors[p] to 1-colors[p]
            old_c = colors[p]
            new_c = 1 - old_c
            
            # For each AP, the color at the role position changes
            # We need to check if the AP becomes mono after the flip
            # The other 3 positions keep their colors
            
            # Build the new color array for each AP with the flip applied
            new_ac0 = ac0.copy()
            new_ac1 = ac1.copy()
            new_ac2 = ac2.copy()
            new_ac3 = ac3.copy()
            
            # Apply flip to the correct role
            mask0 = (roles == 0)
            mask1 = (roles == 1)
            mask2 = (roles == 2)
            mask3 = (roles == 3)
            
            new_ac0[mask0] = new_c
            new_ac1[mask1] = new_c
            new_ac2[mask2] = new_c
            new_ac3[mask3] = new_c
            
            new_all_zero = (new_ac0 == 0) & (new_ac1 == 0) & (new_ac2 == 0) & (new_ac3 == 0)
            new_all_one = (new_ac0 == 1) & (new_ac1 == 1) & (new_ac2 == 1) & (new_ac3 == 1)
            new_mono = new_all_zero | new_all_one
            
            # Delta for this position: (number of APs that become non-mono) - (number that become mono)
            # = -sum(cur_mono) + sum(new_mono) for the APs that contain p
            # But we only count APs where the flip actually changes something
            # Actually, delta = new_mono_count - cur_mono_count for APs containing p
            delta_p = int(np.sum(new_mono) - np.sum(cur_mono))
            deltas[p] = delta_p
        
        return deltas
    
    def greedy_polish_vectorized(colors, max_iters=200):
        """Local search using vectorized delta computation."""
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(colors)
            
            # Find the best position to flip (most negative delta)
            best_pos = int(np.argmin(deltas))
            best_delta = deltas[best_pos]
            
            if best_delta >= 0:
                break
            
            colors[best_pos] = 1 - colors[best_pos]
            current += best_delta
        
        return colors, current
    
    def simulated_annealing_vectorized(initial_colors, start_temp=0.5, end_temp=0.01,
                                       cooling_rate=0.9995, max_steps=30000):
        """Simulated annealing with vectorized delta computation."""
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position
            pos = random.randint(0, N - 1)
            old_c = colors[pos]
            new_c = 1 - old_c
            colors[pos] = new_c
            new_count = count_mono(colors)
            
            delta = new_count - current
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current = new_count
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            else:
                colors[pos] = old_c
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count
    
    def pair_flip_polish(colors, max_iters=50):
        """Try flipping pairs of positions to escape local optima."""
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random pairs
            for _ in range(100):
                if time.time() > deadline:
                    break
                p1 = random.randint(0, N - 1)
                p2 = random.randint(0, N - 1)
                if p1 == p2:
                    continue
                
                old_c1 = colors[p1]
                old_c2 = colors[p2]
                colors[p1] = 1 - old_c1
                colors[p2] = 1 - old_c2
                new_count = count_mono(colors)
                
                if new_count < current:
                    current = new_count
                    improved = True
                else:
                    colors[p1] = old_c1
                    colors[p2] = old_c2
            
            if not improved:
                break
        
        return colors, current
    
    best_colors = None
    best_score = float('inf')
    
    # Multiple random restarts with greedy polish + pair flip + SA
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        random_colors = np.random.randint(0, 2, N, dtype=np.int8)
        
        # Greedy polish
        polished, score = greedy_polish_vectorized(random_colors, max_iters=100)
        
        if score < best_score:
            best_score = score
            best_colors = polished.copy()
        
        # Pair flip polish
        if time.time() < deadline - 10:
            pair_polished, pair_score = pair_flip_polish(polished, max_iters=30)
            if pair_score < best_score:
                best_score = pair_score
                best_colors = pair_polished.copy()
            
            # Greedy again after pair flips
            polished2, score2 = greedy_polish_vectorized(pair_polished, max_iters=100)
            if score2 < best_score:
                best_score = score2
                best_colors = polished2.copy()
        
        # Simulated annealing if time permits
        if time.time() < deadline - 15:
            sa_colors, sa_score = simulated_annealing_vectorized(
                best_colors, start_temp=0.3, end_temp=0.01,
                cooling_rate=0.9997, max_steps=15000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
            
            # Final greedy polish after SA
            final_colors, final_score = greedy_polish_vectorized(best_colors, max_iters=50)
            if final_score < best_score:
                best_score = final_score
                best_colors = final_colors.copy()
    
    # Final polish
    if best_colors is not None and time.time() < deadline - 2:
        final_colors, final_score = greedy_polish_vectorized(best_colors, max_iters=200)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    # Convert to list of ints
    return [int(c) for c in best_colors]
