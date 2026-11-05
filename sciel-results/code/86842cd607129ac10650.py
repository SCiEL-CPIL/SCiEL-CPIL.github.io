import math
import time
import itertools

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance pairs (edges).
    The optimal number of edges is 76.
    """
    # Deadline guard
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # 1. Generate Hexagon of radius 2 (Core)
    core_points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                core_points.append([a, b])
    
    # 2. Generate Shell of radius 3 (Boundary candidates)
    boundary_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                boundary_points.append([a, b])
                
    n_core = len(core_points) # 19
    n_boundary = len(boundary_points) # 18
    target_count = 32
    num_to_pick = target_count - n_core # 13

    if len(core_points) != 19 or len(boundary_points) != 18:
        return [[a, b] for a in range(16) for b in range(2)]

    # 3. Pre-compute edges
    core_edges = 0
    for i in range(n_core):
        for j in range(i+1, n_core):
            if is_adjacent(core_points[i], core_points[j]):
                core_edges += 1

    b_core_edges = [0] * n_boundary
    for i, bp in enumerate(boundary_points):
        for cp in core_points:
            if is_adjacent(bp, cp):
                b_core_edges[i] += 1

    b_b_adj = [[0]*n_boundary for _ in range(n_boundary)]
    for i in range(n_boundary):
        for j in range(i+1, n_boundary):
            if is_adjacent(boundary_points[i], boundary_points[j]):
                b_b_adj[i][j] = 1
                b_b_adj[j][i] = 1

    best_edges = -1
    best_combo = None
    
    # 4. Brute force the best subset of 13 boundary points
    for combo in itertools.combinations(range(n_boundary), num_to_pick):
        if time.time() > deadline - 5:
            break
            
        edges = core_edges
        for i in combo:
            edges += b_core_edges[i]
            
        for idx in range(len(combo)):
            i = combo[idx]
            row = b_b_adj[i]
            for jdx in range(idx+1, len(combo)):
                j = combo[jdx]
                edges += row[j]
                
        if edges > best_edges:
            best_edges = edges
            best_combo = combo
            
            if best_edges == 76:
                break

    if best_combo is None:
        best_combo = tuple(range(num_to_pick))
        best_edges = core_edges
        for i in best_combo:
            best_edges += b_core_edges[i]
        for idx in range(len(best_combo)):
            i = combo[idx]
            for jdx in range(idx+1, len(combo)):
                j = combo[jdx]
                best_edges += b_b_adj[i][j]

    final_points = core_points + [boundary_points[i] for i in best_combo]
    
    # 5. Verification
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if is_adjacent(points[i], points[j]):
                    count += 1
        return count

    current_score = count_edges(final_points)
    
    # Fallback Local Search if not 76
    if current_score < 76 and time.time() < deadline - 10:
        # Use a simple swap-based local search
        # Generate pool of radius 4
        R = 4
        pool = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pool.append((a, b))
        
        current_set = set(map(tuple, final_points))
        points_list = [list(p) for p in current_set]
        
        improved = True
        while improved and current_score < 76 and time.time() < deadline - 5:
            improved = False
            best_swap_score = current_score
            best_swap_in_idx = -1
            best_swap_out_pt = None
            
            for i in range(len(points_list)):
                p_in = points_list[i]
                deg_in = 0
                for j in range(len(points_list)):
                    if i != j and is_adjacent(p_in, points_list[j]):
                        deg_in += 1
                        
                for p_out in pool:
                    if tuple(p_out) in current_set:
                        continue
                    
                    deg_out = 0
                    for j in range(len(points_list)):
                        if i != j and is_adjacent(p_out, points_list[j]):
                            deg_out += 1
                            
                    new_score = current_score - deg_in + deg_out
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_in_idx = i
                        best_swap_out_pt = p_out
                        
            if best_swap_in_idx != -1:
                old_pt = points_list[best_swap_in_idx]
                current_set.remove(tuple(old_pt))
                current_set.add(tuple(best_swap_out_pt))
                points_list[best_swap_in_idx] = list(best_swap_out_pt)
                current_score = best_swap_score
                improved = True

        if current_score > best_edges:
            best_edges = current_score
            final_points = points_list

    return final_points
