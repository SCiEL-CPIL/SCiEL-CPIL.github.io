import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field elements
    # alpha^2 = 3 is a non-residue mod 7?
    # Squares mod 7: 0, 1, 4, 2 (3^2=9=2), 4 (4^2=16=2? No 4^2=16=2 mod 7? 16=14+2. Yes.)
    # 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
    # Residues: 0, 1, 2, 4.
    # Non-residues: 3, 5, 6.
    # Smallest is 3. So alpha^2 = 3.
    
    # Map (a, b) to int 0..48
    def to_int(a, b):
        return a * p + b
        
    def to_pair(i):
        return (i // p, i % p)
        
    # Precompute field ops? Not strictly necessary if we work with pairs, but ints are faster.
    # Let's stick to pairs for clarity in line generation, or precompute line points.
    
    # Generate all points in F_q^2
    # Point is (x, y) where x, y in F_q.
    # x = (xa, xb), y = (ya, yb).
    # Total q^2 = 2401 points.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Index mapping
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Directions
    # v = (1, m) for m in F_q, and v = (0, 1).
    # m is an element of F_q.
    # Let's list all 50 directions as pairs (va, vb).
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma * 0 + mb)) # Wait, v=(1, m). m=(ma, mb).
            # v = (1, 0) + (0, m)? No.
            # v is a vector in F_q^2.
            # v = (vx, vy).
            # Case 1: vx = 1, vy = m.
            # Case 2: vx = 0, vy = 1.
            pass
            
    # Correct direction generation:
    dirs = []
    for ma in range(p):
        for mb in range(p):
            # v = (1, m)
            dirs.append(((1, 0), (ma, mb))) # vx=(1,0), vy=(ma,mb)
    # v = (0, 1)
    dirs.append(((0, 0), (1, 0))) # vx=(0,0), vy=(1,0)
    
    # Precompute line points for each point y and direction v
    # line_points[y_idx][v_idx] = set of point indices on the line, excluding y
    # This is 2401 * 50 sets of 48 ints.
    # Memory: 2401 * 50 * 48 * 4 bytes ~ 23 MB. OK.
    
    line_points = [[None for _ in range(len(dirs))] for _ in range(len(points))]
    
    for y_idx, y in enumerate(points):
        for v_idx, v in enumerate(dirs):
            pts_on_line = set()
            for t_idx, t in enumerate(points): # t is in F_q, so t is a pair
                if t_idx == 0: # t=0
                    continue
                # x = y + t*v
                # y = (ya, yb) is pair of pairs.
                # Let's flatten.
                pass
                
    # This nested loop is too slow in Python.
    # We need to compute lines faster.
    # x = y + t v.
    # x_a = y_a + t_a v_a, x_b = y_b + t_b v_b.
    # Since we iterate t over all F_q, and v is fixed, the set of points is just a permutation of F_q^2 shifted?
    # No, it's a line.
    
    # Let's precompute addition and multiplication in F_q.
    # F_q elements are 0..48.
    # add[i][j] = i+j
    # mul[i][j] = i*j
    
    add = np.zeros((q, q), dtype=np.int8)
    mul = np.zeros((q, q), dtype=np.int8)
    
    for i in range(q):
        ia, ib = to_pair(i)
        for j in range(q):
            ja, jb = to_pair(j)
            # Add
            sa = (ia + ja) % p
            sb = (ib + jb) % p
            add[i, j] = to_int(sa, sb)
            # Mul
            # (ia + ib alpha)(ja + jb alpha) = ia*ja + (ia*jb + ib*ja)alpha + ib*jb alpha^2
            # alpha^2 = 3
            ma = (ia * ja + ib * jb * 3) % p
            mb = (ia * jb + ib * ja) % p
            mul[i, j] = to_int(ma, mb)
            
    # Now generate lines
    # y is index. v is index (0..48 for m, 49 for vertical).
    # We need to map direction v to (vx, vy) indices in F_q.
    # vx, vy are elements of F_q (0..48).
    
    v_coords = []
    for ma in range(p):
        for mb in range(p):
            vx = to_int(1, 0)
            vy = to_int(ma, mb)
            v_coords.append((vx, vy))
    # Vertical
    vx = to_int(0, 0)
    vy = to_int(1, 0)
    v_coords.append((vx, vy))
    
    line_points = []
    for y_idx in range(q * q):
        y_pt = points[y_idx]
        # Flatten y to (y_x, y_y) where y_x, y_y are F_q elements (0..48)
        yx = to_int(y_pt[0][0], y_pt[0][1])
        yy = to_int(y_pt[1][0], y_pt[1][1])
        
        lines_for_y = []
        for vx, vy in v_coords:
            pts = set()
            for t in range(q):
                if t == 0:
                    continue
                tx = mul[t, vx]
                ty = mul[t, vy]
                px = add[yx, tx]
                py = add[yy, ty]
                # Map (px, py) to point index
                # px, py are 0..48.
                # point index = px * q + py
                pts.add(px * q + py)
            lines_for_y.append(pts)
        line_points.append(lines_for_y)
        
    # Initial N
    # Use the seed construction
    k = 3
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    N = [True] * (q * q)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = to_int(xa, xb) * q + to_int(ya, yb)
                        N[idx] = False
                        
    # Compute V_y and I_y
    # V_y is list of v_idx such that line is valid
    # I_y is intersection of lines in V_y
    
    V_y = []
    I_y = []
    
    for y_idx in range(q * q):
        valid_v = []
        for v_idx in range(len(v_coords)):
            line = line_points[y_idx][v_idx]
            # Check if all points in line are in N
            valid = True
            for p_idx in line:
                if not N[p_idx]:
                    valid = False
                    break
            if valid:
                valid_v.append(v_idx)
                
        if not valid_v:
            # Should not happen for initial N
            continue
            
        # Intersection of lines
        inter = set(line_points[y_idx][valid_v[0]])
        for v_idx in valid_v[1:]:
            inter = inter.intersection(line_points[y_idx][v_idx])
            
        V_y.append(valid_v)
        I_y.append(inter)
        
    # Greedy removal
    start_time = time.time()
    deadline = 600
    
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find removable x
        # x is removable if x not in I_y for all y != x
        removable = None
        # Iterate through N
        # To speed up, we can check random points or iterate.
        # Let's iterate.
        for x_idx in range(q * q):
            if not N[x_idx]:
                continue
                
            is_removable = True
            for y_idx in range(q * q):
                if y_idx == x_idx:
                    continue
                if x_idx in I_y[y_idx]:
                    is_removable = False
                    break
                    
            if is_removable:
                removable = x_idx
                break
                
        if removable is None:
            break
            
        # Remove removable
        N[removable] = False
        
        # Update V_y and I_y
        # For each y, if removable was on any line in V_y, update
        for y_idx in range(q * q):
            if y_idx == removable:
                continue
                
            # Check if removable is on any valid line for y
            # We can check if removable in union of lines in V_y?
            # Or just check each v in V_y.
            changed = False
            new_V = []
            for v_idx in V_y[y_idx]:
                if removable in line_points[y_idx][v_idx]:
                    changed = True
                else:
                    new_V.append(v_idx)
                    
            if changed:
                V_y[y_idx] = new_V
                if not new_V:
                    # Invalid state, should not happen
                    break
                # Recompute I_y
                inter = set(line_points[y_idx][new_V[0]])
                for v_idx in new_V[1:]:
                    inter = inter.intersection(line_points[y_idx][v_idx])
                I_y[y_idx] = inter
                
    # Convert N to array
    pts = []
    for i in range(q * q):
        if N[i]:
            x = points[i]
            pts.append([x[0], x[1]])
            
    return np.array(pts, dtype=np.int64)
