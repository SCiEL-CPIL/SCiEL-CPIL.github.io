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
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
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
    
    def is_valid_3coloring(colors):
        """Check if a 3-coloring is valid (every 4-AP has >= 3 distinct colors)."""
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if fewer than 3 distinct colors
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
    
    def try_3color_backtracking(order, max_nodes=500000):
        """Backtracking search for a valid 3-coloring with given position order."""
        colors = [-1] * n
        nodes = [0]
        
        def backtrack(pos_idx):
            if pos_idx == n:
                return True
            if nodes[0] >= max_nodes:
                return False
            
            x = order[pos_idx]
            nodes[0] += 1
            
            for color in range(3):
                colors[x] = color
                
                # Check all APs containing x where other positions are already colored
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors[p] != -1 for p in others):
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        if c0 == c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 and c2 == c3:
                            valid = False
                            break
                        if c0 == c2 and c1 == c3:
                            valid = False
                            break
                        if c0 == c3 and c1 == c2:
                            valid = False
                            break
                
                if valid:
                    if backtrack(pos_idx + 1):
                        return True
                else:
                    colors[x] = -1
            
            colors[x] = -1
            return False
        
        if backtrack(0):
            return colors
        return None
    
    def try_4color_backtracking(order, max_nodes=500000):
        """Backtracking search for a valid 4-coloring with given position order."""
        colors = [-1] * n
        nodes = [0]
        
        def backtrack(pos_idx):
            if pos_idx == n:
                return True
            if nodes[0] >= max_nodes:
                return False
            
            x = order[pos_idx]
            nodes[0] += 1
            
            for color in range(4):
                colors[x] = color
                
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors[p] != -1 for p in others):
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        if c0 == c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 and c2 == c3:
                            valid = False
                            break
                        if c0 == c2 and c1 == c3:
                            valid = False
                            break
                        if c0 == c3 and c1 == c2:
                            valid = False
                            break
                
                if valid:
                    if backtrack(pos_idx + 1):
                        return True
                else:
                    colors[x] = -1
            
            colors[x] = -1
            return False
        
        if backtrack(0):
            return colors
        return None
    
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
                        if c0 == c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 and c2 == c3:
                            valid = False
                            break
                        if c0 == c2 and c1 == c3:
                            valid = False
                            break
                        if c0 == c3 and c1 == c2:
                            valid = False
                            break
                if valid:
                    break
            else:
                colors[x] = 0  # Fallback
        return colors
    
    def local_search(colors, time_limit, num_colors):
        """Local search to fix violations and minimize colors."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 2.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            candidates = list(set(colors))
            for _ in range(3):
                new_c = random.randint(0, num_colors - 1)
                if new_c not in candidates:
                    candidates.append(new_c)
            
            best_new_color = old_color
            best_score = float('inf')
            
            curr_invalid = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if c0 == c1 == c2 == c3:
                    curr_invalid += 1
                elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                    curr_invalid += 1
                elif c0 == c1 and c2 == c3:
                    curr_invalid += 1
                elif c0 == c2 and c1 == c3:
                    curr_invalid += 1
                elif c0 == c3 and c1 == c2:
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if c0 == c1 == c2 == c3:
                        new_invalid += 1
                    elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                        new_invalid += 1
                    elif c0 == c1 and c2 == c3:
                        new_invalid += 1
                    elif c0 == c2 and c1 == c3:
                        new_invalid += 1
                    elif c0 == c3 and c1 == c2:
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                old_count = sum(1 for c in colors if c == old_color)
                new_count = sum(1 for c in colors if c == new_color)
                delta_c = 0
                if old_count == 1:
                    delta_c -= 1
                if new_count == 0:
                    delta_c += 1
                
                score = delta_v * 100 + delta_c
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
    
    # Strategy: Try to find a valid 3-coloring
    # First try greedy, then backtracking with different orders, then local search
    
    best_result = None
    best_num_colors = 999
    
    # Try greedy 3-coloring
    colors = greedy_3color()
    v = count_violations(colors)
    c_used = len(set(colors))
    if v == 0:
        best_result = colors
        best_num_colors = c_used
    
    # If greedy failed or used more than 3 colors, try backtracking
    if best_num_colors > 3:
        # Try backtracking with natural order
        order = list(range(n))
        result = try_3color_backtracking(order, max_nodes=200000)
        if result is not None:
            best_result = result
            best_num_colors = 3
    
    # Try with random orders if still no 3-coloring
    if best_num_colors > 3 and time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            order = list(range(n))
            random.shuffle(order)
            result = try_3color_backtracking(order, max_nodes=100000)
            if result is not None:
                best_result = result
                best_num_colors = 3
                break
    
    # If no 3-coloring found, try 4-coloring
    if best_num_colors > 3 and time.time() < deadline - 20:
        # Greedy 4-coloring
        colors4 = [-1] * n
        for x in range(n):
            for color in range(4):
                colors4[x] = color
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    others = [p for p in ap if p != x]
                    if all(colors4[p] != -1 for p in others):
                        c0, c1, c2, c3 = colors4[ap[0]], colors4[ap[1]], colors4[ap[2]], colors4[ap[3]]
                        if c0 == c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                            valid = False
                            break
                        if c0 == c1 and c2 == c3:
                            valid = False
                            break
                        if c0 == c2 and c1 == c3:
                            valid = False
                            break
                        if c0 == c3 and c1 == c2:
                            valid = False
                            break
                if valid:
                    break
            else:
                colors4[x] = 0
        
        v4 = count_violations(colors4)
        c4_used = len(set(colors4))
        if v4 == 0 and c4_used < best_num_colors:
            best_result = colors4
            best_num_colors = c4_used
    
    # Local search refinement
    if best_result is not None and time.time() < deadline - 5:
        refined, v_ref, c_ref = local_search(best_result, deadline - 5, max(3, best_num_colors))
        if v_ref == 0 and c_ref <= best_num_colors:
            best_result = refined
            best_num_colors = c_ref
    
    # Fallback: identity mapping
    if best_result is None:
        return list(range(n))
    
    return best_result
# EVOLVE-BLOCK-END
