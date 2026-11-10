# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Hermitian-unital baseline
    followed by aggressive multi-pass greedy removal with constraint propagation.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401

    # Precompute lines for each direction
    # Direction 0: vertical v = (0,1)
    # Directions 1-49: v = (1, m) for m in F_49
    direction_lines = []

    # Vertical direction
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))

    # Slope directions
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))

    # point_line[pt_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # Baseline: exclude points on Hermitian unital
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            in_N[i] = False

    # Compute initial not_in_N_count per direction/line
    not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            count = sum(1 for pt_idx in line if not in_N[pt_idx])
            not_in_N_count[dir_idx][line_idx] = count

    # witness_count[pt_idx] = number of directions where pt has a witness
    witness_count = [0] * n
    for pt_idx in range(n):
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            count = not_in_N_count[dir_idx][line_idx]
            if count == 0 or (count == 1 and not in_N[pt_idx]):
                witness_count[pt_idx] += 1

    # Greedy removal with constraint propagation
    # We try to remove points from N in multiple passes
    # A point can be removed if removing it doesn't leave any point with 0 witnesses

    removed_in_this_pass = True
    max_passes = 200
    pass_count = 0

    while removed_in_this_pass and pass_count < max_passes:
        if time.time() > deadline - 30:
            break
        pass_count += 1
        removed_in_this_pass = False

        # Build candidate list: points in N, sorted by witness_count ascending
        # (points with fewer witnesses are more "critical" to keep, so we try to remove
        #  points with more witnesses first - they're less likely to be sole witnesses)
        # Actually, we want to remove points that are LEAST useful as witnesses for others.
        # Let's compute for each point how many other points depend on it.
        
        # Simpler approach: try removing in order of decreasing witness_count
        # Points with many witness directions are safer to remove.
        candidates = [i for i in range(n) if in_N[i]]
        candidates.sort(key=lambda i: -witness_count[i])

        for cand in candidates:
            if time.time() > deadline - 30:
                break

            if not in_N[cand]:
                continue

            # Check if removing cand breaks any point's witness
            # For each direction, the line through cand gets one more point not in N
            # Points on that line may lose this direction as witness
            
            broken = False
            affected = []  # list of (pt_idx, dir_idx) that lose witness
            
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                line = direction_lines[dir_idx][line_idx]
                
                old_count = not_in_N_count[dir_idx][line_idx]
                new_count = old_count + 1
                
                for Q in line:
                    if Q == cand:
                        continue
                    # Q had witness for dir_idx iff:
                    #   old_count == 0, or (old_count == 1 and not in_N[Q])
                    had_witness = (old_count == 0) or (old_count == 1 and not in_N[Q])
                    # Q has witness for dir_idx iff:
                    #   new_count == 0, or (new_count == 1 and not in_N[Q])
                    has_witness = (new_count == 0) or (new_count == 1 and not in_N[Q])
                    
                    if had_witness and not has_witness:
                        # Q loses this direction
                        if witness_count[Q] <= 1:
                            broken = True
                            break
                        else:
                            affected.append((Q, dir_idx))
                
                if broken:
                    break
            
            if broken:
                continue
            
            # Remove cand
            in_N[cand] = False
            removed_in_this_pass = True
            
            # Update not_in_N_count
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                not_in_N_count[dir_idx][line_idx] += 1
            
            # Update witness_count for affected points
            for Q, dir_idx in affected:
                witness_count[Q] -= 1

    # Final verification pass: ensure validity
    # Every point must have at least one witness direction
    # If not, we need to add back some points. For simplicity, if invalid, return all points.
    valid = True
    for pt_idx in range(n):
        has_witness = False
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            count = not_in_N_count[dir_idx][line_idx]
            if count == 0 or (count == 1 and not in_N[pt_idx]):
                has_witness = True
                break
        if not has_witness:
            valid = False
            break

    if not valid:
        # Fallback: return all points (always valid)
        result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in all_points], dtype=np.int64)
        return result

    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
