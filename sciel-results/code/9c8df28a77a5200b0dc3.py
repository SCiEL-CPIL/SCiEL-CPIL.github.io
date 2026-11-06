"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
    
    # 1. Generate initial coloring - try Hamming weight first, then greedy
    initial_colors = hamming_weight_coloring()
    
    if initial_colors is None:
        for k in range(3, 15):
            initial_colors = greedy_construct(k)
            if initial_colors is not None:
                break
    
    if initial_colors is None:
        # Fallback: use a simple pattern that's known to work
        initial_colors = [i % 4 for i in range(N)]
        if not is_valid_coloring(initial_colors):
            # Last resort: use many colors
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
    
    # Final validation
    if not is_valid_coloring(current_colors):
        # Should not happen, but fallback
        current_colors = [i % 4 for i in range(N)]
    
    return current_colors
# EVOLVE-BLOCK-END
