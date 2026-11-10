import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600  # seconds
    p = 7
    q = 49
    k = 3  # alpha^2 = 3

    # Precompute field operations using integer indices 0..48
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            r0, r1 = (a0 + b0) % p, (a1 + b1) % p
            add_tbl[i, j] = r0 * p + r1
            r0, r1 = (a0 - b0) % p, (a1 - b1) % p
            sub_tbl[i, j] = r0 * p + r1
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q  # 2401

    # Build lines: 49 vertical + 49*49 slanted = 2450 lines
    lines = []
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
    
    num_lines = len(lines)
    
    # Precompute: for each point, which lines contain it
    point_lines = [[] for _ in range(total_points)]
    for lid, L in enumerate(lines):
        for pt in L:
            point_lines[pt].append(lid)
    
    # State tracking for fast validity
    S_mask = np.zeros(total_points, dtype=bool)
    line_S_count = np.zeros(num_lines, dtype=np.int32)
    
    # For each point, track how many lines through it satisfy the requirement
    # If point is in S: need count of lines with S_count==1
    # If point is in N: need count of lines with S_count==0
    # valid_line_count[x] = number of "good" lines for point x
    valid_line_count = np.zeros(total_points, dtype=np.int32)
    
    # Initialize: all points in N, all lines have S_count=0
    # So for every point, all lines through it have S_count=0, which is "good" for N
    for pt in range(total_points):
        valid_line_count[pt] = len(point_lines[pt])
    
    # Number of points that are currently invalid (should be 0)
    invalid_count = 0
    
    def try_add(pt):
        """Try adding pt to S. Returns True if valid, False otherwise (and reverts)."""
        nonlocal invalid_count
        # Tentatively add pt to S
        S_mask[pt] = True
        for lid in point_lines[pt]:
            old_count = line_S_count[lid]
            line_S_count[lid] = old_count + 1
            # For all other points on this line, update their valid_line_count
            for other_pt in lines[lid]:
                if other_pt == pt:
                    continue
                if not S_mask[other_pt]:
                    # other_pt is in N: it was good if old_count==0, bad if old_count==1
                    if old_count == 0:
                        valid_line_count[other_pt] -= 1
                        if valid_line_count[other_pt] == 0:
                            invalid_count += 1
                    # if old_count >= 1: still bad (count was 1 -> 2, or 2->3, etc.)
                    # Actually: N-point needs a line with count 0. 
                    # Before: if old_count==0, this line was good. After: count=1, not good.
                    # If old_count>=1: this line was already not good, still not good.
                else:
                    # other_pt is in S: it was good if old_count==1, bad otherwise
                    if old_count == 1:
                        valid_line_count[other_pt] -= 1
                        if valid_line_count[other_pt] == 0:
                            invalid_count += 1
                    # if old_count==0: was bad (0!=1), after: 1==1, now good!
                    if old_count == 0:
                        valid_line_count[other_pt] += 1
                        if valid_line_count[other_pt] == 1:
                            invalid_count -= 1
        
        # Now check pt itself: pt is now in S, needs a line with S_count==1
        # The lines through pt now have their updated counts
        # But we need to check if any line through pt has S_count==1
        # Note: pt is on all those lines, and pt is in S, so each line through pt has count >= 1
        # We need count==1, meaning no other S-point on that line
        pt_good = False
        for lid in point_lines[pt]:
            if line_S_count[lid] == 1:
                pt_good = True
                break
        if pt_good:
            valid_line_count[pt] = 1  # at least one good line
        else:
            valid_line_count[pt] = 0
            invalid_count += 1
        
        return invalid_count == 0
    
    def revert_add(pt):
        """Revert adding pt to S."""
        nonlocal invalid_count
        # First, fix pt's own validity
        if valid_line_count[pt] == 0:
            invalid_count -= 1
        # Now revert the line counts and fix other points
        for lid in point_lines[pt]:
            new_count = line_S_count[lid]  # this is the count WITH pt
            old_count = new_count - 1      # count without pt
            line_S_count[lid] = old_count
            for other_pt in lines[lid]:
                if other_pt == pt:
                    continue
                if not S_mask[other_pt]:
                    # other_pt in N: line went from new_count to old_count
                    # If old_count==0: now good (was bad)
                    if old_count == 0:
                        valid_line_count[other_pt] += 1
                        if valid_line_count[other_pt] == 1:
                            invalid_count -= 1
                    # If new_count==1: was bad (1!=0), still bad if old_count==0? No wait.
                    # N-point: good iff count==0
                    # Before revert: count=new_count. After: count=old_count.
                    # If old_count==0 and new_count==1: was bad, now good -> +1
                    # If old_count>=1 and new_count==old_count+1: 
                    #   if old_count==0: handled above
                    #   if old_count>=1: was bad, still bad
                else:
                    # other_pt in S: good iff count==1
                    # Before revert: count=new_count. After: count=old_count.
                    if new_count == 1:
                        # was good, now count=0, bad -> -1
                        valid_line_count[other_pt] -= 1
                        if valid_line_count[other_pt] == 0:
                            invalid_count += 1
                    elif old_count == 1:
                        # was bad (new_count>=2), now good -> +1
                        valid_line_count[other_pt] += 1
                        if valid_line_count[other_pt] == 1:
                            invalid_count -= 1
        
        S_mask[pt] = False
    
    # Candidate ordering: prioritize points that are on fewer lines or have more "slack"
    # Points in U (unital) have geometric structure
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)
    
    # Score each candidate: prefer points where adding them is "safe"
    # A point is safer if fewer N-points rely on its lines having count 0
    # and fewer S-points rely on its lines having count 1
    # Initially all counts are 0, so adding any point makes all N-points on its lines lose one good line
    # Points with fewer lines through them are safer (but all have ~50 lines)
    # Let's just try all points, but use a priority queue based on impact
    
    # For speed, do multiple passes
    added_count = 0
    max_iterations = 100000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 30:
            break
        if added_count == 0 and iteration > 5000:
            # No progress, break
            break
        
        progress = False
        # Try adding points, prioritize U points first in early iterations
        if iteration < 1000:
            candidates = sorted(U, key=lambda pt: len(point_lines[pt]))
        else:
            candidates = sorted(range(total_points), key=lambda pt: (0 if pt in U else 1, len(point_lines[pt])))
        
        for pt in candidates:
            if S_mask[pt]:
                continue
            if time.time() - start_time > DEADLINE - 30:
                break
            
            if try_add(pt):
                added_count += 1
                progress = True
            else:
                revert_add(pt)
        
        if not progress:
            break
    
    # Construct N = all_points - S
    N_points = []
    for pt in range(total_points):
        if not S_mask[pt]:
            x = pt // q
            y = pt % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)
