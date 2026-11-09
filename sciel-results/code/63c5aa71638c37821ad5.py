import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 580
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element: (a, b) -> a + b*alpha, represented as a + b*7 (0..48)
    
    # Precompute field operations
    def sub_f(a, b):
        return ((a % 7 - b % 7) % 7) + (((a // 7 - b // 7) % 7) * 7)
    
    def mul_f(a, b):
        a1, a2 = a % 7, a // 7
        b1, b2 = b % 7, b // 7
        r1 = (a1 * b1 + a2 * b2 * 3) % 7
        r2 = (a1 * b2 + a2 * b1) % 7
        return r1 + r2 * 7
    
    # Directions: (1, m) for m in 0..48, and (0, 1)
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    num_dirs = len(directions)
    
    # Precompute line membership: for each direction, map line_id -> list of point indices
    line_membership = []
    for (vx, vy) in directions:
        lines = {}
        for x in range(q):
            for y in range(q):
                point_idx = x + y * q
                if vx == 0:
                    lid = x
                else:
                    lid = sub_f(y, mul_f(vy, x))
                if lid not in lines:
                    lines[lid] = []
                lines[lid].append(point_idx)
        line_membership.append(lines)
    
    # For each point, precompute its line_id for each direction
    point_line_id = np.zeros((total_points, num_dirs), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            point_idx = x + y * q
            for dir_idx, (vx, vy) in enumerate(directions):
                if vx == 0:
                    lid = x
                else:
                    lid = sub_f(y, mul_f(vy, x))
                point_line_id[point_idx, dir_idx] = lid
    
    # in_N: boolean array
    in_N = np.ones(total_points, dtype=bool)
    
    # For each point and direction, precompute the mask of other points on the line
    # This is too much memory. Instead, let's use a different approach.
    
    # Greedy removal with multiple passes
    # Try to remove points in order of "removability"
    
    # Compute initial removability: for each point, how many directions cover it?
    # A point is covered by direction v if all other points on the line are in N.
    # Initially, all points are in N, so every point is covered by every direction.
    
    # Greedy: remove points one by one, checking coverage after each removal.
    
    # To speed up, let's use a priority queue or just iterate.
    
    # Let's try: iterate through points, try to remove each one.
    # For efficiency, precompute for each point the list of other points for each direction.
    
    # other_points[point_idx, dir_idx] = array of point indices on the line through point_idx with direction dir_idx, excluding point_idx
    # This is 2401 * 50 * 48 = 5.76M integers, about 23 MB. Acceptable.
    
    other_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for point_idx in range(total_points):
        x = point_idx % q
        y = point_idx // q
        for dir_idx, (vx, vy) in enumerate(directions):
            lid = point_line_id[point_idx, dir_idx]
            line_pts = line_membership[dir_idx][lid]
            # Fill other_points[point_idx, dir_idx] with line_pts minus point_idx
            count = 0
            for p_idx in line_pts:
                if p_idx != point_idx:
                    other_points[point_idx, dir_idx, count] = p_idx
                    count += 1
    
    # Now greedy removal
    removed = 0
    for iteration in range(5):
        if time.time() > deadline - 60:
            break
        progress = False
        for point_idx in range(total_points):
            if time.time() > deadline - 60:
                break
            if not in_N[point_idx]:
                continue
            # Check if point_idx can be removed: is there a direction where all other points on the line are in N?
            can_remove = False
            for dir_idx in range(num_dirs):
                # Check if all other points on this line are in N
                others = other_points[point_idx, dir_idx]
                if np.all(in_N[others]):
                    can_remove = True
                    break
            if can_remove:
                in_N[point_idx] = False
                removed += 1
                progress = True
        if not progress:
            break
    
    # Build result
    mask = in_N.astype(bool)
    points = np.zeros((mask.sum(), 2, 2), dtype=np.int64)
    idx = 0
    for point_idx in range(total_points):
        if in_N[point_idx]:
            x = point_idx % q
            y = point_idx // q
            points[idx, 0, 0] = x % 7
            points[idx, 0, 1] = x // 7
            points[idx, 1, 0] = y % 7
            points[idx, 1, 1] = y // 7
            idx += 1
    
    return points
