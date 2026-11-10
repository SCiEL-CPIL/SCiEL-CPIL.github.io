import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) is a + b*alpha
    # idx(a, b) = a * 7 + b
    
    def idx(a, b):
        return a * p + b
        
    def pt_idx(xa, xb, ya, yb):
        x_i = idx(xa, xb)
        y_i = idx(ya, yb)
        return x_i * q + y_i

    # Precompute field ops for multiplication and squaring
    # We need to compute x^2 for parabola
    # x = xa + xb*alpha
    # x^2 = xa^2 + 2xa xb alpha + xb^2 * 3
    # ya = (xa^2 + 3 xb^2) % 7
    # yb = (2 xa xb) % 7
    
    # Generate Parabola
    C = np.zeros(q2, dtype=bool)
    parabola_pts = []
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            pi = pt_idx(xa, xb, ya, yb)
            C[pi] = True
            parabola_pts.append((xa, xb, ya, yb, pi))
            
    # Line Structure
    # Total lines: 49 vertical + 49*49 slanted = 2450
    # Line ID:
    # 0..48: Vertical x = i (i is x_idx)
    # 49..2449: Slanted y = m*x + c
    #   m_idx = 0..48, c_idx = 0..48
    #   id = 49 + m_idx * 49 + c_idx
    
    num_lines = q + q * q
    lines_through_point = [[] for _ in range(q2)]
    points_on_line = [[] for _ in range(num_lines)]
    
    # Build Vertical Lines
    for x_i in range(q):
        li = x_i
        for y_i in range(q):
            pi = x_i * q + y_i
            lines_through_point[pi].append(li)
            points_on_line[li].append(pi)
            
    # Build Slanted Lines
    # y = m x + c
    # m = ma + mb alpha
    # c = ca + cb alpha
    # y = (ma+mb a)(xa+xb a) + (ca+cb a)
    # y = (ma xa + 3 mb xb + ca) + (ma xb + mb xa + cb) alpha
    # ya = (ma xa + 3 mb xb + ca) % 7
    # yb = (ma xb + mb xa + cb) % 7
    
    for ma in range(p):
        for mb in range(p):
            m_i = idx(ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c_i = idx(ca, cb)
                    li = 49 + m_i * q + c_i
                    for xa in range(p):
                        for xb in range(p):
                            x_i = idx(xa, xb)
                            ya = (ma * xa + 3 * mb * xb + ca) % p
                            yb = (ma * xb + mb * xa + cb) % p
                            y_i = idx(ya, yb)
                            pi = x_i * q + y_i
                            lines_through_point[pi].append(li)
                            points_on_line[li].append(pi)
                            
    # Helper to get line ID for tangent of (xa, xb)
    # Tangent to y=x^2 at x0=xa+xb*a
    # y = 2 x0 x - x0^2
    # m = 2 x0 = 2xa + 2xb a
    # c = -x0^2 = -(xa^2 + 3xb^2) + 0 a ? No.
    # x0^2 = (xa^2 + 3 xb^2) + (2 xa xb) a
    # -x0^2 = -(xa^2 + 3 xb^2) + 0 a
    # So ca = -(xa^2 + 3 xb^2) % 7
    #    cb = 0
    
    def get_tangent_line_id(xa, xb):
        ma = (2 * xa) % p
        mb = (2 * xb) % p
        m_i = idx(ma, mb)
        ca = (-(xa * xa + 3 * xb * xb)) % p
        cb = 0
        c_i = idx(ca, cb)
        return 49 + m_i * q + c_i

    # Initialize Blocked
    blocked = np.zeros(q2, dtype=bool)
    
    # Mark points on tangents of initial C (Parabola)
    for xa, xb, ya, yb, pi in parabola_pts:
        li = get_tangent_line_id(xa, xb)
        for p_on_line in points_on_line[li]:
            blocked[p_on_line] = True
            
    # Greedy Add
    # We iterate through all points not in C and not blocked
    # To ensure determinism, we iterate in fixed order
    for pi in range(q2):
        if time.time() > _DEADLINE - _MARGIN:
            break
        if C[pi] or blocked[pi]:
            continue
            
        # Check if there is a line through pi with empty intersection with C
        found = False
        for li in lines_through_point[pi]:
            # Check if any point on li is in C
            # Optimization: if we store count of C points on line, we can check O(1)
            # Let's maintain c_count_on_line
            pass
        
    # Actually, let's maintain c_count_on_line for speed
    c_count_on_line = np.zeros(num_lines, dtype=np.int32)
    for pi in range(q2):
        if C[pi]:
            for li in lines_through_point[pi]:
                c_count_on_line[li] += 1
                
    # Re-initialize blocked? No, blocked is for witness lines.
    # Wait, the logic:
    # 1. blocked indicates points that cannot be added because they lie on a witness line of an existing point.
    # 2. We need to find a line L for new point x such that L contains NO points of C.
    #    i.e., c_count_on_line[li] == 0.
    
    # Re-run loop with c_count_on_line
    for pi in range(q2):
        if time.time() > _DEADLINE - _MARGIN:
            break
        if C[pi] or blocked[pi]:
            continue
            
        # Find a clean line
        clean_li = -1
        for li in lines_through_point[pi]:
            if c_count_on_line[li] == 0:
                clean_li = li
                break
                
        if clean_li != -1:
            # Add pi
            C[pi] = True
            # Update c_count_on_line for all lines through pi
            for li in lines_through_point[pi]:
                c_count_on_line[li] += 1
            # Update blocked: all points on the witness line (clean_li) are now blocked
            for p_on_line in points_on_line[clean_li]:
                blocked[p_on_line] = True

    # Build N
    N = []
    for i in range(q2):
        if not C[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)
