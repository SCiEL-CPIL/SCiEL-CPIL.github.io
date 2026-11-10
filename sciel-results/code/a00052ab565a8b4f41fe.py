import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    N = 11  # 11x11 grid
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = N * N
    
    # 1. Precompute lines with >= 4 points
    unique_line_keys = set()
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B < 0:
                A, B, C = -A, -B, -C
                
            unique_line_keys.add((A, B, C))
            
    relevant_lines = []
    for key in unique_line_keys:
        A, B, C = key
        mask = 0
        for idx, (x, y) in enumerate(points):
            if A * x + B * y + C == 0:
                mask |= (1 << idx)
        if mask.bit_count() >= 4:
            relevant_lines.append(mask)
            
    n_lines = len(relevant_lines)
    
    # Build point_to_lines mapping
    point_to_lines = [[] for _ in range(n_points)]
    for l_idx, mask in enumerate(relevant_lines):
        m = mask
        while m:
            lsb = m & -m
            p_idx = lsb.bit_length() - 1
            point_to_lines[p_idx].append(l_idx)
            m ^= lsb
            
    # State variables
    line_counts = [0] * n_lines
    current_score = 0
    current_invalid = 0
    
    def apply_delta(idx, sign):
        nonlocal current_score, current_invalid
        for l_idx in point_to_lines[idx]:
            old_count = line_counts[l_idx]
            new_count = old_count + sign
            
            if old_count == 4:
                current_score -= 1
            if old_count == 5:
                current_invalid -= 1
                
            if new_count == 4:
                current_score += 1
            if new_count == 5:
                current_invalid += 1
                
            line_counts[l_idx] = new_count

    # Initialize with a valid random configuration
    current_mask = 0
    valid_start = False
    attempts = 0
    while attempts < 2000:
        line_counts = [0] * n_lines
        current_score = 0
        current_invalid = 0
        current_mask = 0
        
        indices = list(range(n_points))
        random.shuffle(indices)
        init_indices = indices[:16]
        
        for idx in init_indices:
            current_mask |= (1 << idx)
            apply_delta(idx, 1)
            
        if current_invalid == 0:
            valid_start = True
            break
        attempts += 1
        
    if not valid_start:
        # Fallback: Return empty or raise, but this is highly unlikely
        return []

    best_score = current_score
    best_mask = current_mask
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n_points - 1)
        if not (current_mask & (1 << i)):
            continue
        j = random.randint(0, n_points - 1)
        if (current_mask & (1 << j)):
            continue
        
        old_score = current_score
        old_invalid = current_invalid
        
        # Apply move: Remove i, Add j
        apply_delta(i, -1)
        apply_delta(j, 1)
        
        if current_invalid == 0:
            delta = current_score - old_score
            # Accept if better or with probability exp(delta/T)
            if delta > 0 or (T > 0.001 and random.random() < math.exp(delta / T)):
                current_mask ^= (1 << i) ^ (1 << j)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                # Revert
                apply_delta(j, -1)
                apply_delta(i, 1)
        else:
            # Invalid, Revert
            apply_delta(j, -1)
            apply_delta(i, 1)
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Convert best_mask to list of points
    result = []
    for i in range(n_points):
        if best_mask & (1 << i):
            result.append([points[i][0], points[i][1]])
            
    return result
