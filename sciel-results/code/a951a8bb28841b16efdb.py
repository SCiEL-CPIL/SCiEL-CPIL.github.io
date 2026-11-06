import time
import random
import math

def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def local_search(k, time_limit):
        # Precompute validity of all patterns
        # Pattern index = c0 * k^3 + c1 * k^2 + c2 * k + c3
        valid_pattern = [False] * (k ** 4)
        for c0 in range(k):
            for c1 in range(k):
                for c2 in range(k):
                    for c3 in range(k):
                        idx = c0 * k ** 3 + c1 * k ** 2 + c2 * k + c3
                        if len({c0, c1, c2, c3}) >= 3:
                            valid_pattern[idx] = True
        
        def get_pattern_idx(colors, ap):
            return colors[ap[0]] * k ** 3 + colors[ap[1]] * k ** 2 + colors[ap[2]] * k + colors[ap[3]]
        
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        # Initialize violation count
        ap_violation = [False] * len(aps)
        current_violations = 0
        for idx, ap in enumerate(aps):
            if not valid_pattern[get_pattern_idx(colors, ap)]:
                ap_violation[idx] = True
                current_violations += 1
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.01
        
        while time.time() < time_limit:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            best_new_color = old_color
            best_delta = 0
            
            affected_aps = pos_to_aps[pos]
            
            # Calculate delta for each new color
            # We need to check which APs are currently invalid
            # and which will become invalid/valid
            
            # To speed up, we can just re-evaluate the affected APs for each candidate color
            # But we can be smarter.
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                delta = 0
                for ap_idx in affected_aps:
                    was_invalid = ap_violation[ap_idx]
                    is_invalid = not valid_pattern[get_pattern_idx(colors, aps[ap_idx])]
                    if was_invalid and not is_invalid:
                        delta -= 1
                    elif not was_invalid and is_invalid:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                elif delta == best_delta and random.random() < 0.5:
                    best_new_color = new_color
            
            colors[pos] = old_color
            
            # Acceptance criterion
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = True
            else:
                prob = math.exp(-best_delta / temp)
                accept = random.random() < prob
            
            if accept:
                colors[pos] = best_new_color
                
                # Update state
                for ap_idx in affected_aps:
                    was_invalid = ap_violation[ap_idx]
                    is_invalid = not valid_pattern[get_pattern_idx(colors, aps[ap_idx])]
                    if was_invalid and not is_invalid:
                        current_violations -= 1
                        ap_violation[ap_idx] = False
                    elif not was_invalid and is_invalid:
                        current_violations += 1
                        ap_violation[ap_idx] = True
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None

    # Try k = 3, 4, 5
    # 3 colors is very hard. 4 is better.
    # We'll try 3 for a short time, then 4.
    
    if time.time() < deadline - 10.0:
        result = local_search(3, time.time() + 10.0)
        if result is not None:
            return result
    
    if time.time() < deadline - 10.0:
        result = local_search(4, time.time() + 20.0)
        if result is not None:
            return result
    
    if time.time() < deadline - 5.0:
        result = local_search(5, time.time() + 10.0)
        if result is not None:
            return result
    
    # Fallback
    return list(range(n))
