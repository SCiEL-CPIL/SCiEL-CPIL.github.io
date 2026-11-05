"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db

    def count_edges(points):
        cnt = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    cnt += 1
        return cnt

    # Hardcoded optimal 32-point subset of the R=3 hexagon
    # Achieves 76 edges (Harborth's optimum for n=32)
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                points.append([a, b])
    
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    result = [p for p in points if (p[0], p[1]) not in remove_set]
    
    # Verify we have exactly 32 distinct points
    if len(result) != 32:
        # Fallback: use hexagon R=2 (19 pts) + greedy selection from R=3 shell
        core = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                if max(abs(a), abs(b), abs(a + b)) <= 2:
                    core.append([a, b])
        
        shell = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                if max(abs(a), abs(b), abs(a + b)) == 3:
                    shell.append([a, b])
        
        current_set = set(tuple(p) for p in core)
        selected_shell = set()
        
        for _ in range(13):
            best_point = None
            best_gain = -1
            remaining = [p for p in shell if tuple(p) not in selected_shell]
            for p in remaining:
                gain = 0
                for q in current_set:
                    if dist_sq(p, q) == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point:
                selected_shell.add(tuple(best_point))
                current_set.add(tuple(best_point))
        
        result = [[a, b] for a, b in current_set]
    
    # Verify edge count; if below 76, attempt a quick local search
    edges = count_edges(result)
    if edges < 76 and time.time() < deadline - 10:
        # Quick hill-climbing polish
        current_set = set(tuple(p) for p in result)
        current_score = edges
        R = 5
        pool = []
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                if max(abs(a), abs(b), abs(a + b)) <= R:
                    pool.append((a, b))
        
        improved = True
        while improved and current_score < 76 and time.time() < deadline - 5:
            improved = False
            best_swap_score = current_score
            best_swap_in = None
            best_swap_out = None
            
            for p_in in current_set:
                deg_in = 0
                for p_other in current_set:
                    if p_in != p_other and dist_sq(p_in, p_other) == 1:
                        deg_in += 1
                for p_out in pool:
                    if p_out in current_set:
                        continue
                    deg_out = 0
                    for p_other in current_set:
                        if p_out != p_other and dist_sq(p_out, p_other) == 1:
                            deg_out += 1
                    if dist_sq(p_in, p_out) == 1:
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
        
        if current_score > edges:
            result = [[a, b] for a, b in current_set]
    
    return result
# EVOLVE-BLOCK-END
