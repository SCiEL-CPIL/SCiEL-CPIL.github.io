"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
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
    
    def ap_has_at_least_3_colors(colors, ap):
        """Check if an AP uses at least 3 distinct colors."""
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        distinct = len({c0, c1, c2, c3})
        return distinct >= 3
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            if not ap_has_at_least_3_colors(colors, ap):
                v += 1
        return v
    
    def is_valid(colors):
        """Check if coloring is valid (every 4-AP has >= 3 distinct colors)."""
        for ap in aps:
            if not ap_has_at_least_3_colors(colors, ap):
                return False
        return True
    
    def randomized_greedy(palette_size, order=None):
        """Randomized greedy: assign random valid color from palette."""
        if order is None:
            order = list(range(n))
            random.shuffle(order)
        
        colors = [-1] * n
        for x in order:
            valid_colors = []
            for color in range(palette_size):
                colors[x] = color
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors[p] != -1 for p in others):
                        if not ap_has_at_least_3_colors(colors, ap):
                            valid = False
                            break
                if valid:
                    valid_colors.append(color)
            
            if valid_colors:
                colors[x] = random.choice(valid_colors)
            else:
                # Fallback: pick random color (will create violation)
                colors[x] = random.randint(0, palette_size - 1)
        
        return colors
    
    def local_search_reduce_colors(colors, time_limit, max_palette=12):
        """Local search focused on reducing number of distinct colors while maintaining validity."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 1.0
        cooling = 0.997
        min_temp = 0.005
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Candidate colors: existing colors + a few random ones
            candidates = set(colors)
            for _ in range(2):
                new_c = random.randint(0, max_palette - 1)
                candidates.add(new_c)
            candidates = list(candidates)
            
            best_new_color = old_color
            best_score = float('inf')
            
            # Count current violations involving this position
            curr_invalid = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                if not ap_has_at_least_3_colors(colors, ap):
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    if not ap_has_at_least_3_colors(colors, ap):
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                
                # Color reduction bonus
                old_count = sum(1 for c in colors if c == old_color)
                new_count = sum(1 for c in colors if c == new_color)
                delta_c = 0
                if old_count == 1:
                    delta_c -= 1
                if new_count == 0:
                    delta_c += 1
                
                # Heavily penalize violations
                score = delta_v * 1000 + delta_c
                if score < best_score:
                    best_score = score
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.2:
                    best_new_color = new_color
                
                colors[pos] = old_color
            
            colors[pos] = old_color
            
            accept = False
            if best_score < 0:
                accept = True
            elif best_score == 0:
                accept = True
            else:
                energy = best_score
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
            
            if accept and best_new_color != old_color:
                colors[pos] = best_new_color
                current_violations = count_violations(colors)
                current_colors_used = len(set(colors))
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                
                if current_violations == 0 and current_colors_used <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_colors_used
    
    # Strategy: Portfolio of randomized greedy with different palette sizes
    best_result = None
    best_num_colors = 999
    
    # Try randomized greedy with various palette sizes
    for palette_size in [3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts for each palette size
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            colors = randomized_greedy(palette_size)
            v = count_violations(colors)
            c_used = len(set(colors))
            
            if v == 0 and c_used < best_num_colors:
                best_result = colors
                best_num_colors = c_used
            
            # If we have a valid coloring, try to reduce colors via local search
            if v == 0 and c_used > 3 and time.time() < deadline - 15:
                refined, v_ref, c_ref = local_search_reduce_colors(colors, deadline - 15, palette_size)
                if v_ref == 0 and c_ref < best_num_colors:
                    best_result = refined
                    best_num_colors = c_ref
            
            # If we have violations but few colors, try local search to fix
            if v > 0 and c_used <= 5 and time.time() < deadline - 15:
                refined, v_ref, c_ref = local_search_reduce_colors(colors, deadline - 15, max(6, palette_size))
                if v_ref == 0 and c_ref < best_num_colors:
                    best_result = refined
                    best_num_colors = c_ref
    
    # If we have a valid result, try further refinement
    if best_result is not None and time.time() < deadline - 5:
        refined, v_ref, c_ref = local_search_reduce_colors(best_result, deadline - 5, max(4, best_num_colors))
        if v_ref == 0 and c_ref <= best_num_colors:
            best_result = refined
            best_num_colors = c_ref
    
    # Fallback: identity mapping
    if best_result is None:
        return list(range(n))
    
    return best_result
# EVOLVE-BLOCK-END
