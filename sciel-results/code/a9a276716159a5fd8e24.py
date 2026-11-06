"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
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
    
    def is_ap_valid(c0, c1, c2, c3):
        """Check if a 4-AP has at least 3 distinct colors."""
        # Invalid if: all same, 3+1, or 2+2
        if c0 == c1 == c2 == c3:
            return False
        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
            return False
        if c0 == c1 and c2 == c3:
            return False
        if c0 == c2 and c1 == c3:
            return False
        if c0 == c3 and c1 == c2:
            return False
        return True
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if not is_ap_valid(c0, c1, c2, c3):
                v += 1
        return v
    
    def is_valid_coloring(colors):
        """Check if coloring is valid (every 4-AP has >= 3 distinct colors)."""
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if not is_ap_valid(c0, c1, c2, c3):
                return False
        return True
    
    # Strategy: Start with a valid 4-coloring, then try to reduce to 3 colors
    # Initial 4-coloring: use a modular pattern
    def make_initial_4coloring():
        # Try pattern based on position mod 4
        colors = [i % 4 for i in range(n)]
        v = count_violations(colors)
        if v == 0:
            return colors
        # Try different patterns
        patterns = [
            lambda i: i % 4,
            lambda i: (i // 2) % 4,
            lambda i: (i // 4) % 4,
            lambda i: (i * 3) % 4,
            lambda i: (i * 5) % 4,
            lambda i: (i * 7) % 4,
        ]
        for pat in patterns:
            colors = [pat(i) for i in range(n)]
            v = count_violations(colors)
            if v == 0:
                return colors
        # Fallback: greedy 4-coloring
        colors = [-1] * n
        for x in range(n):
            for color in range(4):
                colors[x] = color
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors[p] != -1 for p in others):
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        if not is_ap_valid(c0, c1, c2, c3):
                            valid = False
                            break
                if valid:
                    break
            else:
                colors[x] = 0
        return colors
    
    def try_reduce_to_3colors(colors, time_limit):
        """Try to reduce a valid 4-coloring to 3 colors using local search."""
        colors = colors[:]
        current_violations = count_violations(colors)
        
        best_colors = colors[:]
        best_violations = current_violations
        best_num_colors = len(set(colors))
        
        if best_violations == 0 and best_num_colors <= 3:
            return best_colors, best_violations, best_num_colors
        
        # Simulated annealing to minimize violations + color count
        temp = 5.0
        cooling = 0.998
        min_temp = 0.1
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all 4 colors (including possibly introducing new ones)
            candidates = [0, 1, 2, 3]
            random.shuffle(candidates)
            
            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            
            # Count current violations involving pos
            curr_viol = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if not is_ap_valid(c0, c1, c2, c3):
                    curr_viol += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_viol = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if not is_ap_valid(c0, c1, c2, c3):
                        new_viol += 1
                
                delta_v = new_viol - curr_viol
                
                # Color count change
                old_count = sum(1 for c in colors if c == old_color)
                new_count = sum(1 for c in colors if c == new_color)
                delta_c = 0
                if old_count == 1:
                    delta_c -= 1
                if new_count == 0:
                    delta_c += 1
                
                # Score: prioritize reducing violations, then reducing colors
                score = delta_v * 100 + delta_c
                if score < best_delta_v * 100 + best_delta_c:
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                
                colors[pos] = old_color
            
            colors[pos] = old_color
            
            # Acceptance criterion
            energy = best_delta_v * 100 + best_delta_c
            accept = False
            if energy < 0:
                accept = True
            elif energy == 0:
                accept = random.random() < 0.5
            else:
                if temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
            
            if accept and best_new_color != old_color:
                colors[pos] = best_new_color
                current_violations = count_violations(colors)
                num_colors = len(set(colors))
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and num_colors < best_num_colors):
                    best_violations = current_violations
                    best_num_colors = num_colors
                    best_colors = colors[:]
                
                if best_violations == 0 and best_num_colors <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_num_colors
    
    def greedy_3color():
        """Greedy 3-coloring: assign smallest valid color to each position in order."""
        colors = [-1] * n
        for x in range(n):
            for color in range(3):
                colors[x] = color
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors[p] != -1 for p in others):
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        if not is_ap_valid(c0, c1, c2, c3):
                            valid = False
                            break
                if valid:
                    break
            else:
                colors[x] = 0  # Fallback
        return colors
    
    def local_search_from_3color(colors, time_limit):
        """Local search starting from a 3-coloring to fix violations."""
        colors = colors[:]
        current_violations = count_violations(colors)
        
        best_colors = colors[:]
        best_violations = current_violations
        best_num_colors = len(set(colors))
        
        if best_violations == 0 and best_num_colors <= 3:
            return best_colors, best_violations, best_num_colors
        
        temp = 3.0
        cooling = 0.997
        min_temp = 0.05
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            candidates = [0, 1, 2]
            random.shuffle(candidates)
            
            best_new_color = old_color
            best_score = float('inf')
            
            curr_viol = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if not is_ap_valid(c0, c1, c2, c3):
                    curr_viol += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_viol = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if not is_ap_valid(c0, c1, c2, c3):
                        new_viol += 1
                
                delta_v = new_viol - curr_viol
                score = delta_v * 100
                
                if score < best_score:
                    best_score = score
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
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
                num_colors = len(set(colors))
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and num_colors < best_num_colors):
                    best_violations = current_violations
                    best_num_colors = num_colors
                    best_colors = colors[:]
                
                if best_violations == 0 and best_num_colors <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_num_colors
    
    # Main strategy
    best_result = None
    best_num_colors = 999
    
    # 1. Try greedy 3-coloring
    colors3 = greedy_3color()
    v3 = count_violations(colors3)
    c3 = len(set(colors3))
    if v3 == 0:
        best_result = colors3
        best_num_colors = c3
    
    # 2. If no valid 3-coloring, start from 4-coloring and reduce
    if best_num_colors > 3 and time.time() < deadline - 20:
        colors4 = make_initial_4coloring()
        v4 = count_violations(colors4)
        c4 = len(set(colors4))
        
        if v4 == 0:
            # Try to reduce to 3 colors
            refined, v_ref, c_ref = try_reduce_to_3colors(colors4, deadline - 10)
            if v_ref == 0 and c_ref <= 3:
                best_result = refined
                best_num_colors = c_ref
            elif v_ref == 0 and c_ref < best_num_colors:
                best_result = refined
                best_num_colors = c_ref
            elif best_result is None:
                best_result = refined
                best_num_colors = c_ref
    
    # 3. If still no good result, do local search from greedy 3-coloring
    if best_num_colors > 3 and time.time() < deadline - 5:
        refined, v_ref, c_ref = local_search_from_3color(colors3, deadline - 5)
        if v_ref == 0 and c_ref <= 3:
            best_result = refined
            best_num_colors = c_ref
        elif v_ref == 0 and c_ref < best_num_colors:
            best_result = refined
            best_num_colors = c_ref
        elif best_result is None:
            best_result = refined
            best_num_colors = c_ref
    
    # Fallback: return best result found or identity
    if best_result is not None:
        return best_result
    
    # Last resort: identity mapping (128 colors, definitely valid)
    return list(range(n))
# EVOLVE-BLOCK-END
