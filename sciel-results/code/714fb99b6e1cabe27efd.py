"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v
    
    def is_ap_valid(colors, ap):
        c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
        return len(c) >= 3
    
    def greedy_construct():
        """Fast deterministic greedy: assign smallest valid color at each position."""
        colors = [None] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] is None:
                colors[x] = x
        return colors
    
    def try_merge_color(colors, color_to_remove, target_color):
        """Try to merge color_to_remove into target_color. Returns new colors if valid."""
        new_colors = colors[:]
        for i in range(n):
            if new_colors[i] == color_to_remove:
                new_colors[i] = target_color
        if count_violations(new_colors) == 0:
            return new_colors
        return None
    
    def systematic_color_reduction(colors, max_time):
        """Systematically try to reduce colors by merging."""
        colors = colors[:]
        used = sorted(set(colors))
        current_count = len(used)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            used = sorted(set(colors))
            current_count = len(used)
            
            for c_to_remove in used:
                if time.time() > max_time:
                    break
                positions = [i for i in range(n) if colors[i] == c_to_remove]
                if not positions:
                    continue
                for target in used:
                    if target == c_to_remove:
                        continue
                    new_colors = try_merge_color(colors, c_to_remove, target)
                    if new_colors is not None:
                        colors = new_colors
                        improved = True
                        break
                if improved:
                    break
        
        return colors, len(set(colors))
    
    def local_search_minimize_colors(colors, time_limit):
        """Local search to minimize colors while maintaining validity."""
        colors = colors[:]
        best_colors = colors[:]
        best_count = len(set(colors))
        current_violations = count_violations(colors)
        
        if current_violations > 0:
            # First fix violations
            temp = 2.0
            cooling = 0.995
            min_temp = 0.05
            while time.time() < time_limit and temp > min_temp:
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                used = set(colors)
                candidates = list(used) + [random.randint(0, 127) for _ in range(3)]
                
                best_new_color = old_color
                best_delta_v = 0
                affected = pos_to_aps[pos]
                curr_invalid = sum(1 for ai in affected if not is_ap_valid(colors, aps[ai]))
                
                for nc in candidates:
                    if nc == old_color:
                        continue
                    colors[pos] = nc
                    new_invalid = sum(1 for ai in affected if not is_ap_valid(colors, aps[ai]))
                    delta_v = new_invalid - curr_invalid
                    if delta_v < best_delta_v:
                        best_delta_v = delta_v
                        best_new_color = nc
                colors[pos] = old_color
                
                if best_delta_v < 0:
                    colors[pos] = best_new_color
                    current_violations += best_delta_v
                    if current_violations < 0:
                        current_violations = 0
                elif best_delta_v == 0:
                    colors[pos] = best_new_color
                else:
                    energy = best_delta_v * 10
                    if energy > 0 and temp > 0:
                        prob = math.exp(-energy / temp)
                        if random.random() < prob:
                            colors[pos] = best_new_color
                            current_violations += best_delta_v
                
                temp *= cooling
            
            if current_violations > 0:
                return best_colors, best_count
        
        # Now minimize colors
        temp = 1.0
        cooling = 0.997
        min_temp = 0.01
        current_colors_used = len(set(colors))
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            used = set(colors)
            candidates = list(used)
            
            best_new_color = old_color
            best_delta_c = 0
            affected = pos_to_aps[pos]
            curr_invalid = sum(1 for ai in affected if not is_ap_valid(colors, aps[ai]))
            
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            for nc in candidates:
                if nc == old_color:
                    continue
                colors[pos] = nc
                new_invalid = sum(1 for ai in affected if not is_ap_valid(colors, aps[ai]))
                if new_invalid > curr_invalid:
                    colors[pos] = old_color
                    continue
                
                new_count = len(set(colors))
                delta_c = new_count - current_colors_used
                
                if delta_c < best_delta_c:
                    best_delta_c = delta_c
                    best_new_color = nc
                elif delta_c == best_delta_c and random.random() < 0.3:
                    best_new_color = nc
                colors[pos] = old_color
            
            if best_delta_c < 0:
                colors[pos] = best_new_color
                current_colors_used = len(set(colors))
                if current_colors_used < best_count:
                    best_count = current_colors_used
                    best_colors = colors[:]
            elif best_delta_c == 0:
                if random.random() < 0.5:
                    colors[pos] = best_new_color
            
            temp *= cooling
        
        return best_colors, best_count
    
    # Step 1: Fast greedy construction
    colors = greedy_construct()
    initial_violations = count_violations(colors)
    initial_colors = len(set(colors))
    
    # Step 2: Systematic color reduction
    if initial_violations == 0:
        colors, reduced_count = systematic_color_reduction(colors, time.time() + 20.0)
    
    # Step 3: Local search refinement
    remaining = deadline - time.time()
    if remaining > 5.0:
        colors, final_colors = local_search_minimize_colors(colors, deadline - 5.0)
    else:
        final_colors = initial_colors
    
    # Step 4: Final check and fallback
    if count_violations(colors) > 0:
        return list(range(n))
    
    return colors
# EVOLVE-BLOCK-END
