import time

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # 1. Generate Hexagon of radius 2
    # Condition: max(|a|, |b|, |a+b|) <= 2
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                hex2.add((a, b))
                
    # 2. Generate Shell of radius 3
    # Condition: max(|a|, |b|, |a+b|) == 3
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                shell3.append((a, b))
                
    # 3. Greedy selection of 13 points from shell3
    current_set = set(hex2)
    selected_shell = set()
    
    # We need to pick 13 points.
    for _ in range(13):
        best_point = None
        best_gain = -1
        
        remaining = [p for p in shell3 if p not in selected_shell]
        
        for p in remaining:
            # Gain = edges to current_set + edges to selected_shell
            gain = 0
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            for q in selected_shell:
                if is_adjacent(p, q):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_point = p
                
        if best_point:
            selected_shell.add(best_point)
            current_set.add(best_point)
            
    # Verify we have 32 points
    if len(current_set) != 32:
        pass

    # Calculate score to ensure it's 76
    points_list = list(current_set)
    
    def count_edges(pts):
        cnt = 0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if is_adjacent(pts[i], pts[j]):
                    cnt += 1
        return cnt
    
    current_score = count_edges(points_list)
    
    # If we didn't hit 76, run a quick hill climb
    if current_score < 76:
        # Generate a larger pool for swaps if needed
        R = 5
        pool = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pool.append((a, b))
                    
        current_set = set(points_list)
        
        # Precompute adjacency for pool points to current_set?
        # Since current_set changes, we can't precompute fully.
        
        improved = True
        while improved and current_score < 76 and time.time() < deadline - 5:
            improved = False
            best_swap_score = current_score
            best_swap_in = None
            best_swap_out = None
            
            # Optimization: Calculate degree of each point in current_set
            degrees = {}
            for p in current_set:
                deg = 0
                for q in current_set:
                    if p != q and is_adjacent(p, q):
                        deg += 1
                degrees[p] = deg

            for p_in in current_set:
                deg_in = degrees[p_in]
                
                for p_out in pool:
                    if p_out in current_set:
                        continue
                    
                    # Calculate degree of p_out with respect to current_set
                    # Note: if p_out is adjacent to p_in, we subtract 1 because p_in is removed
                    deg_out = 0
                    for p_other in current_set:
                        if is_adjacent(p_out, p_other):
                            deg_out += 1
                            
                    if is_adjacent(p_in, p_out):
                        deg_out -= 1
                        
                    new_score = current_score - deg_in + deg_out
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_in = p_in
                        best_swap_out = p_out
                        
            if best_swap_in is not None:
                current_set.remove(best_swap_in)
                current_set.add(best_swap_out)
                current_score = best_swap_score
                improved = True
                if current_score > 76:
                    break

    return [[a, b] for a, b in current_set]
