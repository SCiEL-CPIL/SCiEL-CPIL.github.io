# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy constraint-satisfaction approach.
    
    Strategy: Start with N = entire plane (2401 points). Greedily remove points from N
    (adding to S) if doing so doesn't immediately violate the Nikodym property for too many
    points. For each point x, it's "covered" if there exists a line direction v such that
    all other points on the line through x in direction v are in N. We greedily remove the
    point whose removal causes the fewest new violations (points that lose their last covering line).
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Field operations: F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # We'll use flat index: idx = (xa * q + xb) * q + (ya * q + yb) ... 
    # Actually, let's use a simpler indexing: each point is (x, y) where x, y in F_49
    # x = (xa, xb), y = (ya, yb)
    # Flat index: x_idx = xa * p + xb, y_idx = ya * p + yb, point_idx = x_idx * q + y_idx
    
    # Precompute field multiplication table for F_49
    # mul[(a,b),(c,d)] = (a,b)*(c,d) = (ac + bd*3, ad + bc) mod 7
    mul_table = np.zeros((q, q, 2), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    ia = a * p + b
                    ic = c * p + d
                    # (a + b*alpha)(c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
                    # = (ac + 3*bd) + (ad + bc)*alpha
                    mul_table[ia, ic, 0] = (a * c + 3 * b * d) % p
                    mul_table[ia, ic, 1] = (a * d + b * c) % p
    
    # Addition table
    add_table = np.zeros((q, q, 2), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    ia = a * p + b
                    ic = c * p + d
                    add_table[ia, ic, 0] = (a + c) % p
                    add_table[ia, ic, 1] = (b + d) % p
    
    # Precompute all lines. A line is defined by a point and a direction v = (1, m) or (0, 1).
    # For each point x, we need to know which lines pass through it and in which direction.
    # 
    # Key insight: For point x, the Nikodym property requires: there exists direction v 
    # (either (1, m) for some m in F_49, or (0,1)) such that for all t in F_49, t != 0:
    #   x + t*v is in N.
    #
    # So point x is "covered" by direction v if all q-1 other points on the line {x + t*v : t in F_49} \ {x} are in N.
    #
    # There are q + 1 = 50 possible directions: (1, m) for m in F_49 (49 directions) plus (0, 1).
    #
    # For each point, maintain:
    #   - cover_count[x][dir] = number of points on the line through x in direction dir (excluding x) that are in N
    #   - point is covered if max over dirs of cover_count[x][dir] == q - 1 = 48
    #
    # When we remove a point y from N, for every point x and direction v such that y is on the line through x in direction v,
    #   cover_count[x][v] decreases by 1.
    #
    # This is expensive. Let's think differently.
    
    # Alternative: For each point x, precompute the set of (x, v) pairs where v is a direction.
    # For a fixed direction v, the line through x in direction v is {x + t*v : t in F_49}.
    # Point y = x + t*v for some t. So y is on the line through x in direction v iff x = y - t*v for some t.
    # 
    # When we remove y, it affects cover_count[x][v] for all x, v such that y is on the line through x in direction v.
    # That means: for each direction v, for each t in F_49, t != 0: x = y - t*v. 
    # So for each direction v, removing y affects q-1 points x (one for each t != 0).
    # Total affected (x, v) pairs: 50 * 48 = 2400.
    
    # Let's precompute for each point y and direction v_idx, the list of points x such that y is on the line through x in direction v.
    # Actually, it's simpler: for direction v, the line through x in direction v contains x + t*v for all t.
    # y is on this line iff y - x is a scalar multiple of v.
    # If v = (1, m): y - x = t*(1, m) => t = y_a - x_a, and we need y_b - x_b = t*m.
    # If v = (0, 1): y - x = t*(0, 1) => y_a = x_a, and t = y_b - x_b.
    
    # Let's index directions: dir 0 to 48 are (1, m) for m in F_49 (m_idx = 0..48), dir 49 is (0, 1).
    num_dirs = q + 1  # 50
    
    # For each direction, precompute the "slope" m or flag for vertical.
    # dir_to_m: for dir 0..48, m = dir (as field element index). dir 49 is vertical.
    
    # Precompute for each point y, for each direction v, the list of points x such that y is on the line through x in direction v.
    # Actually, we need: when removing y, for each direction v, which points x have y on their line in direction v?
    # Answer: x = y - t*v for t in F_49, t != 0.
    
    # Let's precompute: for each y (2401) and each dir (50), the list of x indices (48 each).
    # This is 2401 * 50 * 48 = ~5.7M entries. Might be too much memory/time.
    
    # Alternative approach: Instead of tracking cover counts per direction, track for each point x:
    #   - how many directions v have all q-1 other points in N (i.e., x is covered by v)
    #   - the point is "safe" (can be removed without making x uncovered) if it has at least 2 covering directions
    #     (so removing one point from one line still leaves another covering direction)
    #
    # Actually, we want to remove points from N. A point x is "vulnerable" if it has exactly 1 covering direction.
    # If we remove a point y that is on the unique covering line of x, then x becomes uncovered.
    #
    # Let's try a simpler greedy: 
    # 1. Start with all points in N.
    # 2. For each point y, compute how many points x would become uncovered if we remove y.
    # 3. Remove the point y with the minimum such count (prefer 0).
    # 4. Repeat until no more points can be removed without causing violations, or until time runs out.
    
    # But computing "how many x become uncovered" is expensive.
    
    # Let's try yet another approach: Use the fact that the current baseline has size p^4 - p^2*(p+1)/2.
    # p^2*(p+1)/2 = 49*8/2 = 196. So baseline N has 2401 - 196 = 2205 points, complement 196.
    # 
    # The task says to make N as small as possible (complement as large as possible).
    # 
    # Let's try a different structural approach: Use a probabilistic / random removal with verification.
    # 
    # New idea: Start with N = all points. Randomly order the points. For each point in order,
    # tentatively remove it and check if the Nikodym property still holds for all points.
    # This is O(n^2 * q) which is too slow.
    
    # Let's go back to the cover count approach but optimize it.
    
    # For each point x and direction v, define line_members[x][v] = set of points on the line through x in direction v, excluding x.
    # cover_count[x][v] = |line_members[x][v] ∩ N|
    # x is covered if max_v cover_count[x][v] == q - 1.
    
    # When we remove point y:
    #   For each direction v and each t in F_49, t != 0:
    #     x = y - t*v
    #     cover_count[x][v] -= 1
    #     If cover_count[x][v] was q-1 and is now q-2, and no other direction has cover_count == q-1, then x becomes uncovered.
    
    # We need to track for each x: how many directions v have cover_count[x][v] == q-1.
    # Let's call this num_full_dirs[x].
    # x is covered iff num_full_dirs[x] >= 1.
    
    # Precompute for each y and direction v, the list of (x, v) pairs affected.
    # Actually, for fixed y and v, the affected x's are y - t*v for t != 0.
    # For each such x, we decrement cover_count[x][v].
    
    # Let's precompute the mapping: for each y (0..2400) and each dir (0..49), 
    #   the list of (x_idx) such that x = y - t*v_dir for t in 1..q-1.
    
    # First, let's define the directions as vectors.
    # dir 0..48: v = (1, m) where m = dir_idx (field element)
    # dir 49: v = (0, 1)
    
    def point_to_idx(xa, xb, ya, yb):
        x_idx = xa * p + xb
        y_idx = ya * p + yb
        return x_idx * q + y_idx
    
    def idx_to_point(idx):
        y_idx = idx % q
        x_idx = idx // q
        xb = y_idx % p
        ya = y_idx // p
        xb2 = x_idx % p
        xa2 = x_idx // p
        return (xa2, xb2, ya, xb)
    
    # Precompute field subtraction: sub[a][b] = a - b
    sub_table = np.zeros((q, q, 2), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    ia = a * p + b
                    ic = c * p + d
                    sub_table[ia, ic, 0] = (a - c) % p
                    sub_table[ia, ic, 1] = (b - d) % p
    
    # Precompute scalar multiplication: for each t in F_49 and v in F_49^2, t*v
    # Actually, we need: for direction v, and scalar t, compute t*v.
    # v is a 2D vector over F_49. t*v = (t*v0, t*v1) component-wise.
    
    # For each direction index, store the vector (v0, v1) as field element indices.
    dir_vectors = []
    for m_idx in range(q):
        # v = (1, m)
        dir_vectors.append((0, m_idx))  # (v0_idx, v1_idx) where v0 = (1,0) = idx 1*p+0 = 7
    # Wait, 1 in F_49 is (1, 0), so idx = 1*p + 0 = 7.
    # Let me fix: field element 1 is (1, 0) -> idx = 1*7 + 0 = 7.
    dir_vectors = []
    one_idx = 1 * p + 0  # 7
    for m_idx in range(q):
        dir_vectors.append((one_idx, m_idx))
    # Vertical: v = (0, 1). 0 is (0,0) idx=0, 1 is (1,0) idx=7.
    dir_vectors.append((0, one_idx))
    
    # Now, for each direction, precompute t*v for all t in F_49.
    # t*v = (t*v0, t*v1) where multiplication is component-wise in F_49.
    dir_tv = []  # dir_tv[d][t] = (tv0_idx, tv1_idx)
    for d in range(num_dirs):
        v0, v1 = dir_vectors[d]
        tv_list = []
        for t_idx in range(q):
            tv0 = mul_table[t_idx, v0]
            tv1 = mul_table[t_idx, v1]
            tv_list.append((tv0[0], tv0[1], tv1[0], tv1[1]))  # (a0, b0, a1, b1)
        dir_tv.append(tv_list)
    
    # Now, for each point y and direction d, compute the list of x = y - t*v for t != 0.
    # y = (ya, yb) as field element index.
    # x = y - t*v means: x0 = y0 - tv0, x1 = y1 - tv1.
    
    # affected[y][d] = list of x indices (48 elements)
    # Memory: 2401 * 50 * 48 * 8 bytes (int64) = ~46 MB. Acceptable.
    
    # Actually, let's compute on the fly to save memory, or use a more compact representation.
    # Let's precompute it.
    
    affected = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    for y_idx in range(q):
        ya, yb = y_idx // p, y_idx % p
        for d in range(num_dirs):
            for t_idx in range(1, q):  # t != 0
                tv0_a, tv0_b, tv1_a, tv1_b = dir_tv[d][t_idx]
                # x0 = y0 - tv0: (ya - tv0_a, yb - tv0_b) mod p
                x0_a = (ya - tv0_a) % p
                x0_b = (yb - tv0_b) % p
                x1_a = (ya - tv1_a) % p  # Wait, y is a point (x, y) in F_49^2.
                # I'm confusing myself. Let me restart the indexing.
    
    # Let me redefine clearly:
    # A point in F_49^2 is (X, Y) where X, Y in F_49.
    # X = (xa, xb), Y = (ya, yb).
    # Flat index: X_idx = xa*p + xb, Y_idx = ya*p + yb.
    # Point index: pt_idx = X_idx * q + Y_idx.
    
    # A direction v = (v0, v1) where v0, v1 in F_49.
    # Line through point (X, Y) in direction v: {(X + t*v0, Y + t*v1) : t in F_49}.
    
    # When we remove point (Xy, Yy), it affects cover_count[(Xx, Yx)][d] for all (Xx, Yx) and d such that
    # (Xy, Yy) is on the line through (Xx, Yx) in direction d.
    # That is: (Xy, Yy) = (Xx + t*v0_d, Yx + t*v1_d) for some t != 0.
    # So: Xx = Xy - t*v0_d, Yx = Yy - t*v1_d.
    
    # Let me redo the precomputation with correct indexing.
    
    # affected[pt_y][d][t_offset] = pt_x, where pt_x corresponds to (Xy - t*v0_d, Yy - t*v1_d)
    
    affected = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    for Xy_idx in range(q):
        Xya, Xyb = Xy_idx // p, Xy_idx % p
        for Yy_idx in range(q):
            Yya, Yyb = Yy_idx // p, Yy_idx % p
            pt_y = Xy_idx * q + Yy_idx
            
            for d in range(num_dirs):
                v0_idx, v1_idx = dir_vectors[d]
                v0a, v0b = v0_idx // p, v0_idx % p
                v1a, v1b = v1_idx // p, v1_idx % p
                
                for t_idx in range(1, q):
                    ta, tb = t_idx // p, t_idx % p
                    # t*v0 = (ta*v0a + tb*v0b*3, ta*v0b + tb*v0a) mod p
                    tv0a = (ta * v0a + tb * v0b * 3) % p
                    tv0b = (ta * v0b + tb * v0a) % p
                    # t*v1
                    tv1a = (ta * v1a + tb * v1b * 3) % p
                    tv1b = (ta * v1b + tb * v1a) % p
                    
                    # Xx = Xy - t*v0
                    Xxa = (Xya - tv0a) % p
                    Xxb = (Xyb - tv0b) % p
                    Xx_idx = Xxa * p + Xxb
                    # Yx = Yy - t*v1
                    Yxa = (Yya - tv1a) % p
                    Yxb = (Yyb - tv1b) % p
                    Yx_idx = Yxa * p + Yxb
                    
                    pt_x = Xx_idx * q + Yx_idx
                    affected[pt_y][d][t_idx - 1] = pt_x
    
    # Now, initialize cover counts.
    # cover_count[pt][d] = number of points on the line through pt in direction d (excluding pt) that are in N.
    # Initially, all points are in N, so cover_count[pt][d] = q - 1 = 48 for all pt, d.
    
    cover_count = np.full((total_points, num_dirs), q - 1, dtype=np.int64)
    num_full_dirs = np.full(total_points, num_dirs, dtype=np.int64)  # Initially all 50 directions are "full"
    
    in_N = np.ones(total_points, dtype=bool)
    
    # Greedy removal
    # For each point y, compute the "cost" of removing it: how many points x would become uncovered.
    # A point x becomes uncovered if num_full_dirs[x] goes from 1 to 0.
    # When we remove y, for each d and each t != 0:
    #   x = affected[y][d][t-1]
    #   cover_count[x][d] -= 1
    #   If cover_count[x][d] was q-1 and is now q-2: num_full_dirs[x] -= 1
    #   If num_full_dirs[x] becomes 0: x is uncovered (violation)
    
    # To compute the cost efficiently, we can simulate the removal and count violations.
    # But we want to avoid actually removing if it causes violations.
    
    # Alternative: Only remove points that don't cause any violations.
    # A point y can be safely removed if for all x affected by y's removal, num_full_dirs[x] >= 2
    # (so that even if one direction loses a point, x still has at least one full direction).
    # Wait, that's not quite right. x might have num_full_dirs[x] = 1, and the one full direction is d,
    # and y is on that line. Then removing y makes cover_count[x][d] = 47, so num_full_dirs[x] = 0. Violation.
    # If x has num_full_dirs[x] = 2, and y is on one of those lines, then num_full_dirs[x] becomes 1. Still covered.
    
    # So: y can be safely removed if for all x in the affected set of y, 
    #   if cover_count[x][d] == q-1 (i.e., d is a full direction for x), then num_full_dirs[x] >= 2.
    # Equivalently: no affected x has num_full_dirs[x] == 1 and cover_count[x][d] == q-1 for the relevant d.
    
    # Let's implement this check.
    
    # Priority queue: (cost, pt_idx) where cost is 0 if safe, or infinity if not.
    # Actually, let's just iterate: find any point that can be safely removed, remove it, repeat.
    
    # To speed up, maintain a list of "removable" points.
    
    # Let's use a simpler approach: repeatedly scan for removable points.
    # This is O(n^2 * 50 * 48) in the worst case, which is ~2401^2 * 2400 ≈ 1.4e10. Too slow.
    
    # Better: maintain a priority queue of points with their "safety margin".
    # For each point y, define safety[y] = min over affected (x, d) of num_full_dirs[x] (where cover_count[x][d] == q-1).
    # If safety[y] >= 2, y is safe to remove.
    
    # This is still complex. Let's just do a simple greedy with a time limit.
    
    # Simple approach: 
    # 1. Compute for each point y whether it's safe to remove.
    # 2. Remove all safe points simultaneously.
    # 3. Repeat.
    
    # Step 1: For each y, check if it's safe.
    # y is safe if: for all d in 0..49, for all t in 1..48:
    #   x = affected[y][d][t-1]
    #   if cover_count[x][d] == q-1:
    #     if num_full_dirs[x] < 2:
    #       y is not safe
    
    # This is 50 * 48 = 2400 checks per y, times 2401 y's = 5.7M checks per round.
    # Each round removes some points. Number of rounds is at most 2401.
    # Total: 5.7M * 2401 ≈ 1.4e10. Still too slow.
    
    # Let's limit the number of rounds and use a time budget.
    
    max_rounds = 50
    removed_in_round = 0
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 60:
            break
        
        # Find all safe points
        safe = np.ones(total_points, dtype=bool)
        
        for y_idx in range(total_points):
            if not in_N[y_idx]:
                safe[y_idx] = False
                continue
            
            is_safe = True
            for d in range(num_dirs):
                for t_offset in range(q - 1):
                    x_idx = affected[y_idx][d][t_offset]
                    if cover_count[x_idx][d] == q - 1:
                        if num_full_dirs[x_idx] < 2:
                            is_safe = False
                            break
                if not is_safe:
                    break
            safe[y_idx] = is_safe
        
        # Remove all safe points
        safe_indices = np.where(safe)[0]
        removed_in_round = len(safe_indices)
        
        if removed_in_round == 0:
            break
        
        for y_idx in safe_indices:
            in_N[y_idx] = False
            # Update cover counts
            for d in range(num_dirs):
                for t_offset in range(q - 1):
                    x_idx = affected[y_idx][d][t_offset]
                    if cover_count[x_idx][d] > 0:
                        cover_count[x_idx][d] -= 1
                        if cover_count[x_idx][d] == q - 2:  # Was q-1, now q-2
                            num_full_dirs[x_idx] -= 1
        
        if time.time() > deadline - 60:
            break
    
    # Collect points in N
    points = []
    for pt_idx in range(total_points):
        if in_N[pt_idx]:
            X_idx = pt_idx // q
            Y_idx = pt_idx % q
            xa = X_idx // p
            xb = X_idx % p
            ya = Y_idx // p
            yb = Y_idx % p
            points.append([[xa, xb], [ya, yb]])
    
    if len(points) == 0:
        # Fallback: return all points
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
