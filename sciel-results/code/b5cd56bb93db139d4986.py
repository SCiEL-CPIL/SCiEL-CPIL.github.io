import time

def construct():
    # Hardcoded/Deterministic construction
    # 1. Generate Hexagon of radius 2 (19 points)
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                hex2.add((a, b))
                
    # 2. Generate Shell of radius 3 (candidate pool)
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                shell3.append((a, b))
                
    # 3. Greedy selection of 13 points from shell3
    # This is deterministic and fast.
    current_set = set(hex2)
    selected_shell = set()
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    for _ in range(13):
        best_point = None
        best_gain = -1
        
        remaining = [p for p in shell3 if p not in selected_shell]
        
        for p in remaining:
            gain = 0
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_point = p
            elif gain == best_gain and best_point is not None:
                # Tie-breaker: prefer points with lower a, then b for determinism
                if p < best_point:
                    best_point = p
                
        if best_point:
            selected_shell.add(best_point)
            current_set.add(best_point)
            
    points_list = list(current_set)
    
    return [[a, b] for a, b in points_list]
