import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k_const = 3 # alpha^2 = 3
    
    # Field operations
    # Represent element as (a, b)
    
    # Directions in F_q^2
    # v = (1, m) for m in F_q, and (0, 1)
    # m is in F_q, so m = (ma, mb)
    
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb))) # (1, m)
    dirs.append(((0, 1), (0, 0))) # (0, 1)
    
    # There are q+1 = 50 directions.
    
    # Points in F_q^2
    # Point is ((xa, xb), (ya, yb))
    # Index = ((xa*p + xb)*p + ya)*p + yb ? No, just linear index 0..2400
    # Let's map (xa, xb, ya, yb) to index
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    def get_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    # Precompute lines
    # For each direction v, for each line L, list of indices
    # Also for each index, list of line_ids passing through it
    
    line_points = [] # list of lists of indices
    point_lines = [[] for _ in range(q*q)] # list of line_ids for each point
    
    dir_id = 0
    for v in dirs:
        v1, v2 = v
        # Lines are defined by their intersection with some reference axis
        # Or just generate all lines for this direction
        # A line with direction v through point P0 contains P0 + t*v
        # We can identify lines by their "offset"
        
        # Simpler: iterate over all points, group them into lines
        # But we need to be careful not to duplicate lines.
        # For a fixed direction v, the lines partition the space.
        # We can pick a representative point for each line.
        
        # Let's just generate lines by picking a base point (xa, xb, ya, yb)
        # such that it is the "first" point in its line in lexicographic order?
        # Or simpler:
        # For direction (1, m), lines are y = x*m + c.
        # For direction (0, 1), lines are x = c.
        
        # Let's use the property that lines are affine subspaces.
        # We can just iterate all points and use a set to track visited lines.
        
        visited_lines = set()
        
        for idx in range(q*q):
            if idx in visited_lines:
                continue
                
            xa, xb, ya, yb = get_coords(idx)
            # Generate the line containing this point with direction v
            line_pts = []
            for t in range(q):
                # t = (ta, tb)
                ta = t % p
                tb = (t // p) % p
                
                # point = (xa, xb, ya, yb) + t * (v1, v2)
                # v1 = (1, 0) or (0, 1)
                # v2 = (ma, mb)
                
                # x = xa + ta*v1_a + tb*v1_b
                # y = ya + ta*v2_a + tb*v2_b
                
                # v1 = (1, 0) -> v1_a=1, v1_b=0
                # v1 = (0, 1) -> v1_a=0, v1_b=1
                
                if v1[0] == 1:
                    new_xa = (xa + ta) % p
                    new_xb = (xb + tb) % p
                else:
                    new_xa = xa
                    new_xb = xb
                    
                new_ya = (ya + ta*v2[0] + tb*v2[1]) % p
                new_yb = (yb + ta*v2[1] + tb*v2[0]) % p # Wait, field mult
                
                # Field mult: (a,b)*(c,d) = (ac + bd*k, ad + bc)
                # t*v2: (ta, tb) * (ma, mb)
                # ta*ma + tb*mb*k, ta*mb + tb*ma
                
                t_v2_a = (ta*v2[0] + tb*v2[1]*k_const) % p
                t_v2_b = (ta*v2[1] + tb*v2[0]) % p
                
                new_ya = (ya + t_v2_a) % p
                new_yb = (yb + t_v2_b) % p
                
                line_pts.append(get_idx(new_xa, new_xb, new_ya, new_yb))
                
            # Add to line_points
            line_id = len(line_points)
            line_points.append(line_pts)
            
            for pt in line_pts:
                point_lines[pt].append(line_id)
                visited_lines.add(pt)
                
        dir_id += 1
        
    # Initialize S with parabola y = x^2
    # x in F_q, y in F_q
    # x = (xa, xb), y = (ya, yb)
    # y = x*x
    S = set()
    for xa in range(p):
        for xb in range(p):
            # x = (xa, xb)
            # x^2 = (xa^2 - xb^2*k, 2*xa*xb)
            ya = (xa*xa - xb*xb*k_const) % p
            yb = (2*xa*xb) % p
            S.add(get_idx(xa, xb, ya, yb))
            
    # Initialize counts
    counts = np.zeros(len(line_points), dtype=np.int32)
    for idx in S:
        for line_id in point_lines[idx]:
            counts[line_id] += 1
            
    # Greedy addition
    all_indices = list(range(q*q))
    
    # Helper to check if a point can be added
    def can_add(p_idx):
        # 1. Check if p has a witness direction in current S
        # i.e., exists line L through p such that counts[L] == 0
        has_witness = False
        for line_id in point_lines[p_idx]:
            if counts[line_id] == 0:
                has_witness = True
                break
        if not has_witness:
            return False
            
        # 2. Tentatively add p
        # Save old counts for lines through p
        old_counts = {}
        for line_id in point_lines[p_idx]:
            old_counts[line_id] = counts[line_id]
            counts[line_id] += 1
            
        # 3. Check if all x in S still have a witness
        for x_idx in S:
            has_witness_x = False
            for line_id in point_lines[x_idx]:
                if counts[line_id] == 1:
                    has_witness_x = True
                    break
            if not has_witness_x:
                # Revert
                for line_id in point_lines[p_idx]:
                    counts[line_id] = old_counts[line_id]
                return False
                
        # If we reach here, p can be added
        return True
        
    # Main loop
    start_time = time.time()
    while True:
        if time.time() - start_time > 550:
            break
            
        added_any = False
        # Try to add points
        # Shuffle to avoid bias?
        # Just iterate
        for p_idx in all_indices:
            if p_idx in S:
                continue
            if can_add(p_idx):
                S.add(p_idx)
                added_any = True
                # Update counts (can_add already did it if it returned True? 
                # No, can_add reverts if False. If True, counts are updated.
                # Wait, my can_add implementation:
                # If it returns True, counts are updated.
                # If it returns False, counts are reverted.
                # So we don't need to update again.
                
        if not added_any:
            break
            
    # Construct result
    # N = F_q^2 \ S
    points = []
    for idx in all_indices:
        if idx not in S:
            xa, xb, ya, yb = get_coords(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
