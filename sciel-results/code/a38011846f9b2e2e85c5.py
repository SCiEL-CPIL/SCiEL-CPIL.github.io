import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 128
    
    # Precompute all 4-term APs
    ap_list = []
    pos_to_aps = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            idx = len(ap_list)
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_list.append(ap)
            for pos in ap:
                pos_to_aps[pos].append(idx)
                
    num_aps = len(ap_list)
    
    def is_ap_valid(ap_idx, colors):
        a, b, c, d = ap_list[ap_idx]
        colors_in_ap = {colors[a], colors[b], colors[c], colors[d]}
        return len(colors_in_ap) >= 3

    def count_violations(colors):
        violations = 0
        for i in range(num_aps):
            if not is_ap_valid(i, colors):
                violations += 1
        return violations

    # Fallback Greedy
    def greedy_solution(k):
        greedy_colors = []
        for x in range(n):
            idx = x
            best_color = -1
            for color in range(k):
                valid = True
                for d in range(1, (idx // 3) + 1):
                    a = idx - 3 * d
                    if a < 0: continue
                    c1 = greedy_colors[a]
                    c2 = greedy_colors[a + d]
                    c3 = greedy_colors[a + 2 * d]
                    if len({color, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    best_color = color
                    break
            if best_color == -1:
                best_color = 0
            greedy_colors.append(best_color)
        return greedy_colors

    # Try to find 3-coloring with multiple restarts
    best_result = None
    best_violations = float('inf')
    
    num_restarts = 100 # Max restarts, limited by time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        colors = [random.randint(0, 2) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_result = colors.copy()
            
        # Local Search
        temperature = 1.0
        cooling_rate = 0.9999
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0:
            if current_violations == 0:
                return colors
                
            violating_aps = [i for i in range(num_aps) if not is_ap_valid(i, colors)]
            if not violating_aps:
                return colors
                
            ap_idx = random.choice(violating_aps)
            a, b, c, d = ap_list[ap_idx]
            positions_in_ap = [a, b, c, d]
            pos = random.choice(positions_in_ap)
            old_color = colors[pos]
            
            best_delta = float('inf')
            best_new_color = old_color
            
            affected_aps = pos_to_aps[pos]
            
            for new_c in range(3):
                if new_c == old_color:
                    continue
                    
                delta = 0
                for ap_i in affected_aps:
                    ap_tuple = ap_list[ap_i]
                    others = [p for p in ap_tuple if p != pos]
                    c_others = [colors[p] for p in others]
                    
                    set_before = set(c_others + [old_color])
                    valid_before = len(set_before) >= 3
                    
                    set_after = set(c_others + [new_c])
                    valid_after = len(set_after) >= 3
                    
                    if valid_before and not valid_after:
                        delta += 1
                    elif not valid_before and valid_after:
                        delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_c
            
            if best_delta < 0:
                colors[pos] = best_new_color
                current_violations += best_delta
                no_improve_count = 0
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_result = colors.copy()
            else:
                if random.random() < pow(2.718281828, -best_delta / temperature):
                    colors[pos] = best_new_color
                    current_violations += best_delta
                no_improve_count += 1
                
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
                
            if no_improve_count > max_no_improve:
                break

    if best_violations == 0 and best_result:
        return best_result
        
    # Fallback to greedy 4-coloring
    g = greedy_solution(4)
    if count_violations(g) == 0:
        return g
        
    # If all fails, return best result (might be invalid, but better than nothing? No, must be valid)
    # Try greedy 5, 6...
    for k in range(5, 10):
        g = greedy_solution(k)
        if count_violations(g) == 0:
            return g
            
    return best_result

