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
    
    def count_violations(colors):
        """Count number of 4-term APs that violate the constraint (use < 3 distinct colors)."""
        violations = 0
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations

    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True

    def local_search(k, time_limit):
        """
        Attempt to find a valid coloring using exactly k colors (labels 0..k-1).
        Uses simulated annealing with incremental violation updates.
        """
        start_t = time.time()
        
        # Initialize with random coloring using k colors
        colors = [random.randint(0, k - 1) for _ in range(N)]
        
        # Initial violation count
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors[:]
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() - start_t < time_limit:
            if current_violations == 0:
                return colors
            
            # Pick a random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Pick a new color from 0..k-1
            if random.random() < 0.5:
                new_color = random.randint(0, k - 1)
            else:
                # Try a color different from old if possible
                new_color = (old_color + 1) % k
                if new_color == old_color and k > 1:
                    new_color = (old_color + 2) % k
            
            if new_color == old_color:
                continue
            
            # Calculate delta violations
            delta = 0
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                # Check before
                before_colors = {colors[a], colors[b], colors[c], colors[d]}
                was_violated = len(before_colors) < 3
                
                # Temporarily assign
                colors[pos] = new_color
                
                # Check after
                after_colors = {colors[a], colors[b], colors[c], colors[d]}
                is_violated = len(after_colors) < 3
                
                # Revert for now to calculate delta cleanly
                colors[pos] = old_color
                
                if was_violated and not is_violated:
                    delta -= 1
                elif not was_violated and is_violated:
                    delta += 1
            
            new_violations = current_violations + delta
            
            if new_violations < current_violations or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept move
                colors[pos] = new_color
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if current_violations == 0:
                        return colors
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
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
                    res = try_merge_pair(current, c1, c2, 2.0)
                    if res is not None:
                        current = res
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    # Strategy:
    # 1. Try to find a valid coloring with 3 colors using local search.
    # 2. If successful, try to polish it (though it's already 3).
    # 3. If 3 fails, try 4, then 5, etc.
    # 4. Use greedy construction as a fallback to get a valid coloring with more colors, then polish.
    
    # Try k=3
    res = local_search(3, 10.0)
    if res is not None:
        return res
        
    # Try k=4
    if time.time() < deadline - 5:
        res = local_search(4, 10.0)
        if res is not None:
            return res
            
    # Try k=5
    if time.time() < deadline - 5:
        res = local_search(5, 10.0)
        if res is not None:
            return res
            
    # Fallback: Greedy construction with many colors, then polish
    def greedy_valid_fallback():
        # Try greedy with increasing number of colors
        for max_c in [10, 15, 20, 30, 50, 128]:
            if time.time() > deadline - 5:
                break
            res = greedy_construct(max_c)
            if res is not None:
                # Polish to reduce colors
                polished = local_search_polish(res, 15.0)
                return polished
        
        # Absolute last resort: distinct colors
        return list(range(N))

    return greedy_valid_fallback()
# EVOLVE-BLOCK-END
