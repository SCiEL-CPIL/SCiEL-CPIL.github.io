import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 550
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3
    
    def field_add(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)
    
    def field_mul(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        a_res = (a1 * a2 + b1 * b2 * k_const) % p
        b_res = (a1 * b2 + a2 * b1) % p
        return a_res * p + b_res

    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # Total q + 1 = 50 directions
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    num_dirs = len(directions)
    
    # Precompute line points for each point and direction
    # line_pts[x][v] = list of points in L(x, v) excluding x
    # x is index 0..2400, v is index 0..49
    line_pts = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    # Mapping from point index to coordinates
    # idx = x_idx * q + y_idx
    # x_idx = x_a * p + x_b
    # y_idx = y_a * p + y_b
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            point_idx = x_idx * q + y_idx
            
            for v_idx, (dx_idx, dy_idx) in enumerate(directions):
                dx_a, dx_b = divmod(dx_idx, p)
                dy_a, dy_b = divmod(dy_idx, p)
                
                count = 0
                for t in range(q):
                    if t == 0:
                        continue
                    t_a, t_b = divmod(t, p)
                    
                    tdx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                    tdx_b = (t_a * dx_b + t_b * dx_a) % p
                    tdx_idx = tdx_a * p + tdx_b
                    
                    tdy_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    tdy_b = (t_a * dy_b + t_b * dy_a) % p
                    tdy_idx = tdy_a * p + tdy_b
                    
                    nx_a = (x_a + tdx_a) % p
                    nx_b = (x_b + tdx_b) % p
                    ny_a = (y_a + tdy_a) % p
                    ny_b = (y_b + tdy_b) % p
                    
                    n_point_idx = nx_a * p * q + nx_b * q + ny_a * p + ny_b
                    
                    line_pts[point_idx, v_idx, count] = n_point_idx
                    count += 1

    # State: For each point x, we assign a direction dir[x]
    # The set N is the union of all points in L(x, dir[x]) \ {x} for all x.
    
    # Initialize with random directions
    dir_assign = np.random.randint(0, num_dirs, size=total_points)
    
    # Calculate initial N
    def get_set_size(dir_assign):
        in_set = np.zeros(total_points, dtype=bool)
        for x in range(total_points):
            v = dir_assign[x]
            pts = line_pts[x, v]
            in_set[pts] = True
        return np.sum(in_set), in_set
    
    current_size, current_set = get_set_size(dir_assign)
    best_size = current_size
    best_dir = dir_assign.copy()
    
    # Hill Climbing / Local Search
    # We want to minimize the size of the union.
    # Changing dir[x] affects the set.
    
    # To speed up, we can track counts of how many times each point is covered.
    # cover_count[y] = number of x such that y in L(x, dir[x]) \ {x}
    # If cover_count[y] > 1, removing y's coverage (by changing dir[x]) doesn't change the union size.
    # If cover_count[y] == 1, removing it increases the union size by 1 (if y was in N).
    # If cover_count[y] == 0, adding it decreases the union size by 1 (if y was not in N).
    
    cover_count = np.zeros(total_points, dtype=np.int32)
    in_set = np.zeros(total_points, dtype=bool)
    
    # Initialize cover_count
    for x in range(total_points):
        v = dir_assign[x]
        pts = line_pts[x, v]
        for p_idx in pts:
            cover_count[p_idx] += 1
            in_set[p_idx] = True
    
    current_size = np.sum(in_set)
    best_size = current_size
    best_dir = dir_assign.copy()
    
    no_improve_count = 0
    step = 0
    
    while time.time() < deadline:
        # Pick a random point x to change direction
        x = np.random.randint(0, total_points)
        old_v = dir_assign[x]
        
        # Try a new random direction
        new_v = np.random.randint(0, num_dirs)
        if new_v == old_v:
            continue
        
        # Calculate delta in set size
        # Points in L(x, old_v) \ {x} lose one cover
        # Points in L(x, new_v) \ {x} gain one cover
        
        old_pts = line_pts[x, old_v]
        new_pts = line_pts[x, new_v]
        
        delta_size = 0
        
        # Process removals from old line
        for p_idx in old_pts:
            if cover_count[p_idx] == 1:
                # This point will leave the set
                delta_size += 1
            # cover_count decreases
            cover_count[p_idx] -= 1
        
        # Process additions to new line
        for p_idx in new_pts:
            if cover_count[p_idx] == 0:
                # This point will enter the set
                delta_size -= 1
            # cover_count increases
            cover_count[p_idx] += 1
        
        if delta_size < 0:
            # Improvement
            dir_assign[x] = new_v
            current_size += delta_size
            no_improve_count = 0
            
            if current_size < best_size:
                best_size = current_size
                best_dir = dir_assign.copy()
        
        elif delta_size == 0 and random.random() < 0.1:
            # Neutral move, explore
            dir_assign[x] = new_v
            no_improve_count += 1
        else:
            # Revert (already reverted cover_count logic implicitly? No, we modified it)
            # We need to revert cover_count if we don't accept
            # Actually, the logic above modifies cover_count in place.
            # If we reject, we must revert.
            
            # Revert old line
            for p_idx in old_pts:
                cover_count[p_idx] += 1
            # Revert new line
            for p_idx in new_pts:
                cover_count[p_idx] -= 1
            
            no_improve_count += 1
            
        step += 1
        
        if no_improve_count > 10000:
            # Restart or randomize?
            # Just continue, maybe randomize a few directions
            for _ in range(10):
                rx = np.random.randint(0, total_points)
                rv = np.random.randint(0, num_dirs)
                old_v = dir_assign[rx]
                
                # Revert logic is complex if we just change. 
                # Let's just accept random moves if stuck? 
                # Or break and restart from best?
                pass
            
            no_improve_count = 0

    # Build result from best_dir
    # N is the set of points y such that cover_count_best[y] > 0?
    # No, we need to reconstruct N from best_dir.
    
    in_set_final = np.zeros(total_points, dtype=bool)
    for x in range(total_points):
        v = best_dir[x]
        pts = line_pts[x, v]
        in_set_final[pts] = True
    
    # The Nikodym set N is the set of points that are in the union.
    # Wait, the task asks for the set N itself.
    # The points in N are those in the union.
    
    points = []
    for point_idx in range(total_points):
        if in_set_final[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    if not points:
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)
