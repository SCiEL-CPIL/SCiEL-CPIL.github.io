import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3  # alpha^2 = 3

    # Precompute field tables
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 - b1) % p)
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q  # 2401

    # Construct Unital U: 2x + y^2 = 0 in F_49
    # 2(xa + xb*alpha) + (ya + yb*alpha)^2 = 0
    # (2*xa + ya^2 + 3*yb^2) + (2*xb + 2*ya*yb)*alpha = 0
    U = set()
    for ya in range(p):
        for yb in range(p):
            # x_b = -y_a * y_b
            xb = (-ya * yb) % p
            # x_a = 4 * (y_a^2 - 3 * y_b^2)
            xa = (4 * (ya * ya - k * yb * yb)) % p
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            U.add(x_idx * q + y_idx)
    
    # U has 49 points

    all_points_set = set(range(total_points))
    outside_U = all_points_set - U

    # Build all lines
    lines = []
    line_points = []
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)
        line_points.append(set(pts))
        
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
            line_points.append(set(pts))
            
    num_lines = len(lines)
    
    # Precompute line intersections with U
    line_U_intersections = []
    for L in lines:
        I = set(pt for pt in L if pt in U)
        line_U_intersections.append(I)
        
    H = set()
    
    def check_U_points():
        # Check if all x in U are satisfied
        for x in U:
            satisfied = False
            for i in range(num_lines):
                if x not in line_points[i]:
                    continue
                I = line_U_intersections[i]
                if x in H:
                    # Need L \ {x} subset N => L \cap (U \ H) = empty
                    # Since x in H, x not in U\H.
                    # So we need I \ {x} subset H? No.
                    # N = (F \ U) U H.
                    # L \ {x} subset N iff for all y in L \ {x}, y in N.
                    # y in N iff y not in U\H.
                    # So we need (L \ {x}) \cap (U \ H) = empty.
                    # L \cap (U \ H) = (L \cap U) \ (L \cap H) = I \ H.
                    # We need (I \ H) \ {x} = empty?
                    # If x in H, x not in U\H.
                    # So we need I \ H = empty.
                    if I.issubset(H):
                        satisfied = True
                        break
                else:
                    # x not in H. x not in N.
                    # Need L \ {x} subset N.
                    # So (L \ {x}) \cap (U \ H) = empty.
                    # L \cap (U \ H) = I \ H.
                    # x not in H, so x in U\H.
                    # So I \ H contains x.
                    # We need (I \ H) \ {x} = empty.
                    # i.e., I \ H = {x}.
                    if (I - H) == {x}:
                        satisfied = True
                        break
            if not satisfied:
                return False, x
        return True, None
        
    def check_outside_U():
        # Greedy set cover for outside_U using lines that avoid U\H
        uncovered = set(outside_U)
        if not uncovered:
            return True
        chosen = []
        while uncovered:
            best_line = -1
            best_cov = 0
            for i in range(num_lines):
                I = line_U_intersections[i]
                # Line must avoid U\H
                if not I.issubset(H):
                    continue
                cov = 0
                for pt in lines[i]:
                    if pt in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_line = i
            if best_line == -1 or best_cov == 0:
                return False
            chosen.append(best_line)
            for pt in lines[best_line]:
                if pt in uncovered:
                    uncovered.remove(pt)
        return True

    # Iterative refinement
    while time.time() - start_time < DEADLINE - 10:
        ok, x_fail = check_U_points()
        if not ok:
            H.add(x_fail)
            continue
            
        if check_outside_U():
            break
        else:
            # Need to add more to H
            # Find a point in U\H that blocks many lines
            # Simple heuristic: add a random point from U\H
            candidates = list(U - H)
            if not candidates:
                break
            # Pick one that is in many "bad" lines?
            # Just pick the first one for now
            H.add(candidates[0])
            
    # Construct N
    S = U - H
    N = all_points_set - S
    
    points = []
    for pt in N:
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)
