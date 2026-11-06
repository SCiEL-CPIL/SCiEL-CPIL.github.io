import random
import time
import math

def construct():
    N = 128
    aps = []
    ap_indices_for_pos = [[] for _ in range(N)]
    
    ap_id = 0
    for a in range(N):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                aps.append((a, b, c, e))
                for idx in (a, b, c, e):
                    ap_indices_for_pos[idx].append(ap_id)
                ap_id += 1
                
    num_aps = len(aps)

    def try_k_colors(k):
        ap_violation = [False] * num_aps
        violation_count = 0
        
        def update_ap_status(ap_idx, colors):
            nonlocal violation_count
            a, b, c, e = aps[ap_idx]
            c0, c1, c2, c3 = colors[a], colors[b], colors[c], colors[e]
            is_violation = len({c0, c1, c2, c3}) < 3
            if is_violation and not ap_violation[ap_idx]:
                violation_count += 1
                ap_violation[ap_idx] = True
            elif not is_violation and ap_violation[ap_idx]:
                violation_count -= 1
                ap_violation[ap_idx] = False

        colors = [random.randint(0, k - 1) for _ in range(N)]
        
        for i in range(num_aps):
            update_ap_status(i, colors)
            
        current_violations = violation_count
        best_colors = colors.copy()
        best_violations = current_violations
        
        T = 10.0
        cooling = 0.99
        start_time = time.time()
        time_limit = 20.0
        
        while T > 0.1 and (time.time() - start_time) < time_limit:
            i = random.randint(0, N - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
                
            colors[i] = new_color
            
            affected_aps = ap_indices_for_pos[i]
            for ap_idx in affected_aps:
                update_ap_status(ap_idx, colors)
                
            new_violations = violation_count
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
            else:
                colors[i] = old_color
                for ap_idx in affected_aps:
                    update_ap_status(ap_idx, colors)
                current_violations = violation_count
                
            T *= cooling
            
        if best_violations == 0:
            return best_colors
        return None

    result = try_k_colors(3)
    if result is not None:
        return result
        
    result = try_k_colors(4)
    if result is not None:
        return result
        
    result = try_k_colors(5)
    if result is not None:
        return result

    # Fallback
    colors = []
    for x in range(1, N + 1):
        found = False
        for color in range(N):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - j * d for j in (1, 2, 3)]
                if all(0 <= idx < len(colors) for idx in idxs):
                    c_set = {color} | {colors[idx] for idx in idxs}
                    if len(c_set) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    return colors
