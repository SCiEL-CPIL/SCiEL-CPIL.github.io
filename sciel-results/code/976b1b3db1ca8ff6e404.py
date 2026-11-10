# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses a bitmask-based greedy addition strategy to maximize the complement size.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

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
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Total 50 directions
    
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                # Line through (xa,xb, ya,yb) with direction (1, m):
                # Points: (xa, xb, ya, yb) + t*(1, m) for t in F_49
                # x' = xa + t_a, y' = ya + t*m where t = t_a + t_b*alpha
                # The line is determined by the "intercept" c = y - x*m
                # c = (ya, yb) - (xa, xb) * (ma, mb)
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # Now for each point, precompute which line it's on for each direction
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # Baseline construction: exclude points on Hermitian unital with y NOT in special set
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special:
            in_N[i] = False

    # Compute initial witness counts
    # For each point Q and direction v, Q has v as witness iff
    # all other points on the line through Q in direction v are in N
    
    # For each direction and line, count points not in N
    # not_in_N_count[dir_idx][line_idx] = number of points on that line not in N
    
    not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            count = 0
            for pt_idx in line:
                if not in_N[pt_idx]:
                    count += 1
            not_in_N_count[dir_idx][line_idx] = count

    # witness_count[pt_idx] = number of directions for which pt has a witness
    witness_count = [0] * n
    for pt_idx in range(n):
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            count = not_in_N_count[dir_idx][line_idx]
            # pt has dir as witness iff count == 0 or (count == 1 and not in_N[pt_idx])
            if count == 0 or (count == 1 and not in_N[pt_idx]):
                witness_count[pt_idx] += 1

    # Greedy improvement: try to exclude additional points
    # Candidate points: those currently in N that are on the unital
    candidates = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_N[i] and (2*xa + ya*ya - yb*yb*k) % p == 0:
            candidates.append(i)

    # Sort candidates by witness_count descending (points with more witnesses are safer to exclude)
    candidates.sort(key=lambda i: witness_count[i], reverse=True)

    for cand in candidates:
        if time.time() > deadline - 20:
            break
        
        # Tentatively exclude cand
        in_N[cand] = False
        
        # Check if this breaks any point's witness
        # Only points on lines through cand in each direction are affected
        broken = False
        affected_dirs = []
        
        for dir_idx in range(50):
            line_idx = point_line[cand][dir_idx]
            old_count = not_in_N_count[dir_idx][line_idx]
            new_count = old_count + 1
            
            # Check all other points on this line
            for pt_idx in direction_lines[dir_idx][line_idx]:
                if pt_idx == cand:
                    continue
                # Q loses dir as witness if:
                # Before: Q had dir as witness (old_count == 0 or (old_count == 1 and not in_N[Q]))
                # After: Q does NOT have dir as witness (new_count > 1, or (new_count == 1 and in_N[Q]))
                
                had_witness = (old_count == 0) or (old_count == 1 and not in_N[pt_idx])
                has_witness = (new_count == 0) or (new_count == 1 and not in_N[pt_idx])
                
                if had_witness and not has_witness:
                    broken = True
                    break
            
            if broken:
                break
            
            affected_dirs.append((dir_idx, line_idx, old_count))
        
        if broken:
            in_N[cand] = True
            # Revert all affected directions
            for dir_idx, line_idx, old_count in affected_dirs:
                not_in_N_count[dir_idx][line_idx] = old_count
        else:
            # Commit the exclusion
            for dir_idx, line_idx, old_count in affected_dirs:
                not_in_N_count[dir_idx][line_idx] = old_count + 1
            # Update witness counts for affected points
            for pt_idx in all_points: # This is inefficient, but let's see if we need it
                # Actually, we only need to update witness_count for points on the affected lines
                pass

    # Convert to array
    N = [pt for i, pt in enumerate(all_points) if in_N[i]]
    arr = np.array([[[x[0], x[1]], [y[0], y[1]]] for x, y in N], dtype=np.int64)
    
    return arr

# EVOLVE-BLOCK-END
