import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Generate all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    def is_valid_set(points):
        """Check if a set of points has no three forming an isosceles triangle."""
        n = len(points)
        if n < 3:
            return True
        # Precompute all pairwise squared distances
        dist_sq = {}
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq[(i, j)] = dx*dx + dy*dy
        
        # Check each triplet
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d_ij = dist_sq[(i, j)]
                    d_ik = dist_sq[(i, k)]
                    d_jk = dist_sq[(j, k)]
                    # Isosceles if any two distances are equal
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        return False
        return True

    def greedy_select(points_order):
        selected = []
        
        for p in points_order:
            px, py = p
            valid = True
            dists_from_p = set()
            
            for i in range(len(selected)):
                ax, ay = selected[i]
                d_pa_sq = (px - ax)**2 + (py - ay)**2
                
                if d_pa_sq in dists_from_p:
                    valid = False
                    break
                dists_from_p.add(d_pa_sq)
                
                for j in range(len(selected)):
                    if i == j:
                        continue
                    bx, by = selected[j]
                    d_ab_sq = (ax - bx)**2 + (ay - by)**2
                    d_pb_sq = (px - bx)**2 + (py - by)**2
                    
                    if d_pa_sq == d_ab_sq:
                        valid = False
                        break
                    if d_pb_sq == d_ab_sq:
                        valid = False
                        break
                    if d_pa_sq == d_pb_sq:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected.append(p)
                
        return selected

    def local_improve(points_set, max_iters=50):
        """Try to swap points to improve the set size."""
        if not points_set:
            return points_set
        
        selected = list(points_set)
        selected_set = set(selected)
        non_selected = [p for p in all_points if p not in selected_set]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try swapping each selected point with each non-selected point
            random.shuffle(non_selected)
            for i in range(len(selected)):
                for j in range(len(non_selected)):
                    # Try swapping selected[i] with non_selected[j]
                    new_set = selected[:i] + [non_selected[j]] + selected[i+1:]
                    if is_valid_set(new_set):
                        # Swap succeeded
                        removed_point = selected[i]
                        added_point = non_selected[j]
                        selected[i] = added_point
                        selected_set.remove(removed_point)
                        selected_set.add(added_point)
                        non_selected[j] = removed_point
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    # Strategy: Random restarts with greedy + local improvement
    while time.time() < deadline:
        random.shuffle(all_points)
        current_set = greedy_select(all_points)
        
        # Apply local improvement if we have time
        if time.time() < deadline - 5:
            current_set = local_improve(current_set, max_iters=30)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
