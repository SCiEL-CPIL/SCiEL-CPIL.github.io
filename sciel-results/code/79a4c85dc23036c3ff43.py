# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses a systematic construction based on lines through a point,
    then applies greedy removal to minimize the set size.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines for each direction
    # Directions: v = (1, m) for m in F_49 (49 directions), and v = (0, 1) (1 direction)
    # Total 50 directions
    
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
    # Lines are vertical: same x, all y
    vert_lines = {}
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        key = (xa, xb)
        if key not in vert_lines:
            vert_lines[key] = []
        vert_lines[key].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                # Line through (xa,xb, ya,yb) with direction (1, m)
                # c = y - x*m
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[-1] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Construction: Start with all points, then remove points greedily
    # A point P can be removed if for every direction d, the line through P in direction d
    # has at least one other point in N that can serve as witness for P... 
    # Wait, that's not right. Let me re-read the problem.
    
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, 
    #  with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N"
    
    # So for each point x (whether in N or not), there must exist a direction v such that
    # all other points on the line through x in direction v are in N.
    
    # This means: for each point x, there exists a direction d such that
    # |line(x, d) \ {x}| ∩ N^c = 0, i.e., all other points on that line are in N.
    
    # Strategy: Start with N = all points. Then try to remove points one by one.
    # When we remove point P, we need to check that for every point Q (in or out of N),
    # Q still has at least one witness direction.
    
    # A point Q has direction d as witness iff all other points on line(Q, d) are in N.
    # When we remove P, the only points whose witness status might change are those on lines through P.
    # For a point Q on line(P, d), if P was the only missing point on line(Q, d) besides Q itself,
    # then removing P doesn't help Q (Q still needs all OTHER points to be in N).
    # Actually, removing P makes it harder for Q to have d as witness, because now P is missing.
    
    # So removing P can only break witnesses, not create them.
    # Therefore, we should start with a small initial N and add points, or start with all and remove carefully.
    
    # Let's think differently. The condition is:
    # For each x in F_49^2, there exists v in V (the 50 directions) such that
    # {x + t*v : t in F_49, t != 0} ⊆ N
    
    # This is equivalent to: for each x, min over v of |{x + t*v : t != 0} \ N| = 0
    
    # Let's use a greedy approach: start with N = all points, and try to remove points.
    # When removing P, check if any point Q loses all its witness directions.
    
    in_N = [True] * n
    
    # For each point Q and direction d, track how many points on line(Q,d) \ {Q} are NOT in N
    # missing_count[Q][d] = number of points on line(Q,d) other than Q that are not in N
    # Q has d as witness iff missing_count[Q][d] == 0
    
    missing_count = [[0] * 50 for _ in range(n)]
    # Initially all in N, so missing_count is all 0.
    
    # witness_count[Q] = number of directions d for which Q has d as witness
    witness_count = [50] * n  # initially all 50
    
    # Try to remove points in some order
    # Points on a line with many other points might be safer to remove?
    # Or points that are "redundant" - i.e., their removal doesn't break any witness.
    
    # Let's try removing points in a systematic order
    # We'll iterate through all points and try to remove each one
    
    removed = 0
    for attempt in range(10):  # multiple passes
        if time.time() > deadline - 30:
            break
        
        progress = False
        for pt_idx in range(n):
            if time.time() > deadline - 30:
                break
            
            if not in_N[pt_idx]:
                continue
            
            # Try to remove pt_idx
            # Check if removing pt_idx breaks any point's witness
            # Only points on lines through pt_idx are affected
            
            affected_points = set()
            for dir_idx in range(50):
                line_idx = point_line[pt_idx][dir_idx]
                line = direction_lines[dir_idx][line_idx]
                for Q in line:
                    if Q != pt_idx:
                        affected_points.add(Q)
            
            # For each affected point Q, check if it would lose a witness direction
            # Q loses direction d if:
            #   Before: missing_count[Q][d] == 0 (Q had d as witness)
            #   After: missing_count[Q][d] > 0 (Q loses d as witness)
            # This happens if pt_idx is on line(Q, d) and pt_idx is not Q
            # i.e., if dir_idx is one of the directions through pt_idx and Q
            
            broken = False
            for Q in affected_points:
                for dir_idx in range(50):
                    # Is pt_idx on line(Q, dir_idx)?
                    # pt_idx is on line(Q, dir_idx) iff point_line[pt_idx][dir_idx] == point_line[Q][dir_idx]
                    if point_line[pt_idx][dir_idx] != point_line[Q][dir_idx]:
                        continue
                    
                    # pt_idx is on line(Q, dir_idx) and pt_idx != Q
                    # Before removal: missing_count[Q][dir_idx]
                    # After removal: missing_count[Q][dir_idx] + 1 (since pt_idx is now not in N)
                    
                    old_missing = missing_count[Q][dir_idx]
                    new_missing = old_missing + 1
                    
                    had_witness = (old_missing == 0)
                    has_witness = (new_missing == 0)
                    
                    if had_witness and not has_witness:
                        # Q loses this direction
                        # Check if Q has other witness directions
                        if witness_count[Q] <= 1:
                            broken = True
                            break
                
                if broken:
                    break
            
            if not broken:
                # Remove pt_idx
                in_N[pt_idx] = False
                removed += 1
                progress = True
                
                # Update missing_count and witness_count
                for Q in affected_points:
                    for dir_idx in range(50):
                        if point_line[pt_idx][dir_idx] != point_line[Q][dir_idx]:
                            continue
                        
                        old_missing = missing_count[Q][dir_idx]
                        new_missing = old_missing + 1
                        missing_count[Q][dir_idx] = new_missing
                        
                        had_witness = (old_missing == 0)
                        has_witness = (new_missing == 0)
                        
                        if had_witness and not has_witness:
                            witness_count[Q] -= 1
                        elif not had_witness and has_witness:
                            witness_count[Q] += 1
        
        if not progress:
            break
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
