"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    def get_distinct_count(colors):
        return len(set(colors))
    
    def compute_violation_count(colors):
        """Count number of 4-term APs that violate the constraint (use < 3 distinct colors)."""
        violations = 0
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations
    
    def compute_energy(colors, weight_color_count=100):
        """
        Energy function for simulated annealing.
        Energy = violation_count + weight_color_count * num_distinct_colors
        This penalizes both invalid colorings and high color counts.
        """
        violations = compute_violation_count(colors)
        n_colors = len(set(colors))
        return violations + weight_color_count * n_colors
    
    def hamming_weight_coloring():
        """
        Deterministic construction using Hamming weight of binary representation.
        Position i gets color = popcount(i), which ranges from 0 to 7.
        This should give valid colorings due to binary carry properties.
        """
        colors = [bin(i).count('1') for i in range(N)]
        if is_valid_coloring(colors):
            return colors
        return None
    
    def greedy_construct(max_colors):
        """
        Greedy left-to-right construction using bitmask for fast color selection.
        Tries to assign the smallest valid color in 0..max_colors-1 to each position.
        Returns None if stuck.
        """
        colors = [-1] * N
        
        for pos in range(N):
            forbidden_mask = 0
            stuck = False
            
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                pts = [a, b, c, d]
                others = [p for p in pts if p != pos]
                
                # Check if all others are assigned
                assigned_colors = []
                all_assigned = True
                for p in others:
                    if colors[p] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[p])
                
                if not all_assigned:
                    continue
                
                s = set(assigned_colors)
                if len(s) == 1:
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
                # len(s) == 3: no restriction
            
            if stuck:
                return None
            
            # Find smallest valid color
            found = False
            for c in range(max_colors):
                if not (forbidden_mask & (1 << c)):
                    colors[pos] = c
                    found = True
                    break
            
            if not found:
                return None
        
        # Final validation
        if not is_valid_coloring(colors):
            return None
        
        return colors
    
    def is_local_valid(pos, c, new_colors):
        """Check if assigning c to pos is valid with respect to all taps where pos is involved."""
        for ti in pos_taps[pos]:
            a, b, c_tap, d_tap = all_taps[ti]
            pts = [a, b, c_tap, d_tap]
            others = [p for p in pts if p != pos]
            
            vals = [c]
            for p in others:
                vals.append(new_colors[p])
            
            if len(set(vals)) < 3:
                return False
        return True
    
    def try_remove_color(colors, target_color, time_limit=3.0):
        """
        Attempts to remove target_color by recoloring its occurrences to other existing colors.
        Uses backtracking with forward checking.
        """
        start_t = time.time()
        positions = [i for i, c in enumerate(colors) if c == target_color]
        if not positions:
            return None
        
        available_colors = [c for c in set(colors) if c != target_color]
        if not available_colors:
            return None
        
        new_colors = colors[:]
        
        # Heuristic: order positions by number of constraints (most constrained first)
        positions.sort(key=lambda i: len(pos_taps[i]), reverse=True)
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            
            if idx == len(positions):
                return True
            
            pos = positions[idx]
            
            # Try available colors
            for c in available_colors:
                if c == new_colors[pos]:
                    continue
                
                # Tentative assignment
                new_colors[pos] = c
                
                if is_local_valid(pos, c, new_colors):
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                new_colors[pos] = target_color
            
            return False
        
        if backtrack(0):
            return new_colors
        return None
    
    def try_merge_pair(colors, color1, color2, time_limit=5.0):
        """
        Attempts to merge color2 into color1 by recoloring all positions with color2 to color1
        or other available colors.
        """
        start_t = time.time()
        positions = [i for i, c in enumerate(colors) if c == color2]
        if not positions:
            return None
        
        # Available colors: all colors except color2 (we want to eliminate color2)
        all_colors = set(colors)
        available_colors = [c for c in all_colors if c != color2]
        if not available_colors:
            return None
        
        new_colors = colors[:]
        
        # Order positions by constraint degree
        positions.sort(key=lambda i: len(pos_taps[i]), reverse=True)
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            
            if idx == len(positions):
                # Final check: color2 should be gone
                if color2 not in new_colors:
                    return True
                return False
            
            pos = positions[idx]
            
            # Try available colors, prefer color1 first
            for c in available_colors:
                if c == new_colors[pos]:
                    continue
                
                # Tentative assignment
                new_colors[pos] = c
                
                if is_local_valid(pos, c, new_colors):
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                new_colors[pos] = color2
            
            return False
        
        if backtrack(0):
            return new_colors
        return None
    
    def local_search_polish(colors, time_limit=10.0):
        """
        Try to reduce color count by merging pairs of colors.
        """
        start_t = time.time()
        current = colors[:]
        
        while time.time() - start_t < time_limit:
            colors_in_use = list(set(current))
            if len(colors_in_use) <= 2:
                break
            
            # Try all pairs, sorted by frequency (try merging rare colors first)
            colors_in_use.sort(key=lambda c: current.count(c))
            
            improved = False
            for i in range(len(colors_in_use)):
                for j in range(i + 1, len(colors_in_use)):
                    if time.time() - start_t >= time_limit:
                        break
                    
                    c1, c2 = colors_in_use[i], colors_in_use[j]
                    
                    # Try to merge c2 into c1
                    new_colors = try_merge_pair(current, c1, c2, time_limit=2.0)
                    if new_colors is not None and color_count(new_colors) < color_count(current):
                        current = new_colors
                        improved = True
                        break
                    
                    # Try to merge c1 into c2
                    new_colors = try_merge_pair(current, c2, c1, time_limit=2.0)
                    if new_colors is not None and color_count(new_colors) < color_count(current):
                        current = new_colors
                        improved = True
                        break
                
                if improved or time.time() - start_t >= time_limit:
                    break
            
            if not improved:
                break
        
        return current
    
    def color_count(colors):
        return len(set(colors))
    
    def simulated_annealing(initial_colors, time_limit=25.0):
        """
        Simulated annealing with penalty-based objective.
        Allows temporarily invalid colorings to escape local minima.
        Energy = violation_count + weight * num_distinct_colors
        """
        start_t = time.time()
        T_start = 100.0
        T_end = 0.01
        weight_color_count = 50
        
        current = initial_colors[:]
        current_energy = compute_energy(current, weight_color_count)
        best = current[:]
        best_energy = current_energy
        best_valid = current[:]
        best_valid_energy = float('inf')
        
        if is_valid_coloring(current):
            best_valid = current[:]
            best_valid_energy = compute_energy(current, weight_color_count)
        
        # Precompute color range for SA
        max_color_val = max(max(initial_colors), 7)
        
        step = 0
        total_steps = 50000  # Will be adjusted based on time
        
        while time.time() - start_t < time_limit:
            # Adaptive temperature
            progress = (time.time() - start_t) / time_limit
            T = T_start * math.exp(-progress * math.log(T_start / T_end))
            
            # Pick a random position to recolor
            pos = random.randint(0, N - 1)
            old_color = current[pos]
            
            # Pick a new color (biased toward existing colors to encourage merging)
            existing_colors = list(set(current))
            if random.random() < 0.7:
                # Choose from existing colors
                new_color = random.choice(existing_colors)
            else:
                # Choose from a small range around existing colors
                min_c = min(existing_colors) if existing_colors else 0
                max_c = max(existing_colors) if existing_colors else 0
                new_color = random.randint(max(0, min_c - 1), min(max_color_val, max_c + 1))
            
            if new_color == old_color:
                continue
            
            # Compute delta energy
            current[pos] = new_color
            new_energy = compute_energy(current, weight_color_count)
            delta = new_energy - current_energy
            
            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_energy = new_energy
                step += 1
                
                # Track best valid solution
                if is_valid_coloring(current):
                    valid_energy = compute_energy(current, weight_color_count)
                    if valid_energy < best_valid_energy:
                        best_valid_energy = valid_energy
                        best_valid = current[:]
            else:
                # Reject: revert
                current[pos] = old_color
            
            # Periodically check if we can improve with local search
            if step % 3000 == 0 and time.time() - start_t < time_limit - 5:
                if is_valid_coloring(current):
                    # Try to polish
                    polished = local_search_polish(current, time_limit=3.0)
                    if is_valid_coloring(polished):
                        pol_energy = compute_energy(polished, weight_color_count)
                        if pol_energy < best_valid_energy:
                            best_valid_energy = pol_energy
                            best_valid = polished[:]
                            current = polished[:]
                            current_energy = pol_energy
        
        # Return the best valid solution found
        if is_valid_coloring(best_valid):
            return best_valid
        return None
    
    def final_polish_pass(colors, deadline):
        """
        Final high-precision polish pass that systematically tries to reduce color count.
        Uses aggressive backtracking with explicit deadline guard.
        """
        start_t = time.time()
        current = colors[:]
        current_count = get_distinct_count(current)
        
        # Try to remove colors one at a time, starting with the least frequent
        while time.time() < deadline - 2:
            colors_in_use = list(set(current))
            if len(colors_in_use) <= 2:
                break
            
            # Sort by frequency (least frequent first)
            colors_in_use.sort(key=lambda c: current.count(c))
            
            improved = False
            for target_color in colors_in_use:
                if time.time() > deadline - 2:
                    break
                
                # Try to remove this color
                new_colors = try_remove_color(current, target_color, time_limit=5.0)
                if new_colors is not None:
                    new_count = get_distinct_count(new_colors)
                    if new_count < current_count:
                        current = new_colors
                        current_count = new_count
                        improved = True
                        break
            
            if not improved:
                # Try merging pairs
                colors_in_use = list(set(current))
                if len(colors_in_use) <= 2:
                    break
                
                colors_in_use.sort(key=lambda c: current.count(c))
                
                pair_improved = False
                for i in range(len(colors_in_use)):
                    for j in range(i + 1, len(colors_in_use)):
                        if time.time() > deadline - 2:
                            break
                        
                        c1, c2 = colors_in_use[i], colors_in_use[j]
                        
                        # Try to merge c2 into c1
                        new_colors = try_merge_pair(current, c1, c2, time_limit=3.0)
                        if new_colors is not None and get_distinct_count(new_colors) < current_count:
                            current = new_colors
                            current_count = get_distinct_count(current)
                            pair_improved = True
                            break
                        
                        # Try to merge c1 into c2
                        new_colors = try_merge_pair(current, c2, c1, time_limit=3.0)
                        if new_colors is not None and get_distinct_count(new_colors) < current_count:
                            current = new_colors
                            current_count = get_distinct_count(current)
                            pair_improved = True
                            break
                    
                    if pair_improved or time.time() > deadline - 2:
                        break
                
                if not pair_improved:
                    break
        
        return current
    
    # 1. Generate initial coloring - try multiple strategies and pick the best
    initial_colors = None
    best_initial_count = float('inf')
    
    # Try Hamming weight first
    hw_colors = hamming_weight_coloring()
    if hw_colors is not None:
        initial_colors = hw_colors
        best_initial_count = get_distinct_count(hw_colors)
    
    # Try greedy constructions with different color limits
    if initial_colors is None or best_initial_count > 5:
        for k in range(3, 12):
            g_colors = greedy_construct(k)
            if g_colors is not None:
                g_count = get_distinct_count(g_colors)
                if g_count < best_initial_count:
                    initial_colors = g_colors
                    best_initial_count = g_count
    
    # If still no good initial coloring, use fallback
    if initial_colors is None:
        initial_colors = [i % 4 for i in range(N)]
        if not is_valid_coloring(initial_colors):
            initial_colors = list(range(N))
    
    current_colors = initial_colors
    current_count = get_distinct_count(current_colors)
    
    # 2. Local search to reduce color count
    if time.time() < deadline - 15:
        # First pass: try removing individual colors
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            colors_in_use = list(set(current_colors))
            colors_in_use.sort(key=lambda c: current_colors.count(c))
            
            for c in colors_in_use:
                if time.time() > deadline - 10:
                    break
                
                new_colors = try_remove_color(current_colors, c, time_limit=3.0)
                if new_colors is not None:
                    new_count = get_distinct_count(new_colors)
                    if new_count < current_count:
                        current_colors = new_colors
                        current_count = new_count
                        improved = True
                        break
        
        # Second pass: try merging pairs
        if time.time() < deadline - 5:
            polished = local_search_polish(current_colors, time_limit=10.0)
            if get_distinct_count(polished) < current_count:
                current_colors = polished
                current_count = get_distinct_count(current_colors)
    
    # 3. Simulated annealing to further reduce color count
    if time.time() < deadline - 5:
        sa_result = simulated_annealing(current_colors, time_limit=20.0)
        if sa_result is not None:
            sa_count = get_distinct_count(sa_result)
            if sa_count < current_count:
                current_colors = sa_result
                current_count = sa_count
            elif sa_count == current_count and is_valid_coloring(sa_result):
                # Even if same count, accept if valid (might help with subsequent polishing)
                current_colors = sa_result
    
    # 4. Final high-precision polish pass
    if time.time() < deadline - 2:
        final_polished = final_polish_pass(current_colors, deadline)
        if is_valid_coloring(final_polished):
            final_count = get_distinct_count(final_polished)
            if final_count <= current_count:
                current_colors = final_polished
                current_count = final_count
    
    # Final validation
    if not is_valid_coloring(current_colors):
        # Should not happen, but fallback
        current_colors = [i % 4 for i in range(N)]
    
    return current_colors
# EVOLVE-BLOCK-END
