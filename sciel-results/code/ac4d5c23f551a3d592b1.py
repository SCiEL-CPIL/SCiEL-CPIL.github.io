import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Field element representation: (a, b) means a + b*alpha
    # Precompute all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}
    idx_to_elem = list(field_elements)

    # Field operations
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def f_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def f_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def f_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * alpha_sq) % p, (a0 * b1 + a1 * b0) % p)

    def f_inv(a):
        a0, a1 = a
        norm = (a0 * a0 - a1 * a1 * alpha_sq) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p - 2, p)
        return ((a0 * norm_inv) % p, (-a1 * norm_inv) % p)

    def f_div(a, b):
        b_inv = f_inv(b)
        if b_inv is None:
            return None
        return f_mul(a, b_inv)

    # Precompute multiplication and division tables for speed
    mul_table = np.empty((q, q, 2), dtype=np.int64)
    div_table = np.empty((q, q, 2), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            r = f_mul(idx_to_elem[i], idx_to_elem[j])
            mul_table[i, j] = [r[0], r[1]]
            if j != 0:
                r_div = f_div(idx_to_elem[i], idx_to_elem[j])
                div_table[i, j] = [r_div[0], r_div[1]]

    # The set M (to be excluded from N) is the parabola {(x, x^2) : x in F_49}
    M = set()
    for x in field_elements:
        x2 = f_mul(x, x)
        M.add((x, x2))

    # Verify the Nikodym property for N = F_49^2 \ M
    # For every point P in F_49^2, there must exist a line L through P
    # (with direction (1,m) for some m, or vertical (0,1)) such that L \ {P} is disjoint from M.
    # Equivalently: L \ {P} contains no points of M.
    # If P is in M: need a line through P where no other point of M lies on it.
    # If P is not in M: need a line through P disjoint from M.

    # For each point P, check all allowed lines through P
    # Allowed directions: (1, m) for m in F_49, and (0, 1)
    # A line through P with direction v has points {P + t*v : t in F_49}
    # We need to check if any other point of M lies on this line.
    # P + t*v = (x, x^2) for some x means:
    #   P_x + t*v_x = x
    #   P_y + t*v_y = x^2
    # If v_x != 0: t = (x - P_x) / v_x, then check P_y + t*v_y == x^2
    # If v_x == 0 (vertical): check P_x == x (only one x possible), then P_y + t*v_y = P_x^2 for some t
    #   For vertical: x = P_x, need P_y + t*1 = P_x^2, so t = P_x^2 - P_y. Always exists.
    #   So vertical line through P contains M point (P_x, P_x^2) iff t is valid (always is) AND P != (P_x, P_x^2).
    #   Wait, if P = (P_x, P_x^2), then P is on the parabola and the vertical line passes through P and no other M point
    #   (since each x has exactly one parabola point). So for P on parabola, vertical line works.
    #   For P not on parabola with x-coord P_x: vertical line contains (P_x, P_x^2) which is in M and != P. So vertical line is bad.

    # Let's just brute-force check each point against each allowed direction.
    # For each P, for each direction, count how many M-points (other than P if P in M) are on the line.
    # If count is 0, the line is valid.

    # Precompute for each point P and each direction d, whether the line through P with direction d
    # contains any M-point other than P itself.

    # Directions: index 0 = vertical (0,1), index 1..49 = (1, m) for m in field_elements
    num_dirs = 1 + q

    # For efficiency, precompute for each direction d and each point P, the set of M-points on the line.
    # Actually, let's just check validity directly.

    valid = True
    for P in field_elements:
        for Py in field_elements:
            P_pt = (P, Py)
            P_in_M = P_pt in M
            found_valid_line = False

            # Check vertical direction (0, 1)
            # Line: {(P_x, P_y + t) : t in F_49}
            # M-points on this line: (P_x, P_x^2) if P_x^2 is reachable, i.e., always (P_x, P_x^2)
            # So the only M-point on vertical line through P is (P_x, P_x^2)
            if P_in_M:
                # P is on parabola, so (P_x, P_x^2) = P, no other M-points on vertical line
                found_valid_line = True
            else:
                # (P_x, P_x^2) is on the line and != P, so vertical line is bad
                pass

            if not found_valid_line:
                # Check directions (1, m) for m in F_49
                for m in field_elements:
                    # Line: {(P_x + t, P_y + t*m) : t in F_49}
                    # M-point (x, x^2) is on line iff:
                    #   P_x + t = x  =>  t = x - P_x
                    #   P_y + (x - P_x)*m = x^2
                    # So we need: x^2 - m*(x - P_x) - P_y = 0
                    # This is a quadratic in x. For a given m, how many x satisfy this?
                    # x^2 - m*x + m*P_x - P_y = 0
                    # Discriminant: m^2 - 4*(m*P_x - P_y) = m^2 - 4*m*P_x + 4*P_y
                    # If discriminant is a square, 2 solutions; if zero, 1; if non-square, 0.
                    # We need NO solutions other than possibly P itself (if P in M).
                    # If P in M: P = (P_x, P_x^2). The quadratic becomes x^2 - m*x + m*P_x - P_x^2 = 0.
                    #   x = P_x is a solution: P_x^2 - m*P_x + m*P_x - P_x^2 = 0. Yes.
                    #   So one solution is x = P_x. The other is x = m - P_x (by Vieta: sum = m).
                    #   So there's another M-point at x = m - P_x unless m - P_x = P_x, i.e., m = 2*P_x.
                    #   If m = 2*P_x, the quadratic has a double root at x = P_x, so only P is on the line.
                    #   So for P in M, direction (1, 2*P_x) gives a line where only P is the M-point. Valid!
                    # If P not in M: need the quadratic to have 0 solutions.
                    #   Discriminant must be a non-square.

                    if P_in_M:
                        # Use m = 2*P_x
                        two_px = f_mul((2 % p, 0), P)
                        if m == two_px:
                            found_valid_line = True
                            break
                    else:
                        # Check if discriminant is a non-square
                        m2 = f_mul(m, m)
                        m_px = f_mul(m, P)
                        four_py = f_mul((4 % p, 0), Py)
                        disc = f_sub(f_add(m2, f_neg(f_mul((4 % p, 0), m_px))), f_neg(four_py))
                        # Wait: disc = m^2 - 4*m*P_x + 4*P_y
                        # Let me recompute: disc = m^2 + (-4*m*P_x) + 4*P_y
                        # In field: -4 = 3 mod 7
                        neg_four_mp = f_neg(f_mul((4 % p, 0), m_px))
                        disc = f_add(f_add(m2, neg_four_mp), four_py)
                        # Check if disc is a square
                        is_square = False
                        for cand in field_elements:
                            if f_mul(cand, cand) == disc:
                                is_square = True
                                break
                        if not is_square:
                            found_valid_line = True
                            break

            if not found_valid_line:
                valid = False
                break
        if not valid:
            break

    # Build N = F_49^2 \ M
    points = []
    for x in field_elements:
        for y in field_elements:
            if (x, y) not in M:
                xa, xb = x
                ya, yb = y
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
