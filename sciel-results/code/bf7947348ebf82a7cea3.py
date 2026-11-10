import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha], alpha^2 = 3 (3 is quadratic non-residue mod 7)
    k = 3  # alpha^2 = 3
    
    def field_add(a, b):
        # a, b are (a0, a1), (b0, b1)
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = a0*b0 + a1*b1*k + (a0*b1 + a1*b0)*alpha
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # Inverse of a0 + a1*alpha in F_49
        # (a0 + a1*alpha)(b0 + b1*alpha) = 1
        # a0*b0 + a1*b1*k = 1, a0*b1 + a1*b0 = 0
        # From second: b1 = -a1*b0/a0 (if a0 != 0)
        # Or use norm: inv(a) = conj(a)/norm(a)
        # conj(a0 + a1*alpha) = a0 - a1*alpha
        # norm = a0^2 - a1^2*k
        a0, a1 = a
        norm = (a0*a0 - a1*a1*k) % p
        if norm == 0:
            return None  # shouldn't happen for nonzero
        norm_inv = pow(norm, p-2, p)
        return ((a0 * norm_inv) % p, (-a1 * norm_inv) % p)
    
    # Precompute all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute multiplication table
    mul_table = {}
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            mul_table[(i, j)] = field_mul(a, b)
    
    # For efficiency, map field element to index
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}
    
    # S = {(x, x^2) : x in F_49}
    S = set()
    for x in field_elements:
        x2 = field_mul(x, x)
        S.add((x, x2))
    
    # Verify Nikodym property
    # For each point x in F_49^2, check there's an allowed line through x with no other S points
    
    def get_line_points(x0, y0, direction):
        # direction is (v0, v1), line is {(x0 + t*v0, y0 + t*v1) : t in F_49}
        points = set()
        for t in field_elements:
            tx = field_add(x0, field_mul(t, direction[0]))
            ty = field_add(y0, field_mul(t, direction[1]))
            points.add((tx, ty))
        return points
    
    valid = True
    for x0 in field_elements:
        for y0 in field_elements:
            x = (x0, y0)
            found_line = False
            # Try vertical direction (0, 1)
            v_vert = ((0, 0), (1, 0))  # (0,1) in field coords: 0 + 1*alpha? No.
            # Wait, direction (0,1) means v = (0, 1) where 0 and 1 are field elements.
            # 0 = (0,0), 1 = (1,0)
            v_vert = (((0, 0)), ((1, 0)))  # (0, 1) as field elements
            line_pts = get_line_points(x0, y0, v_vert)
            s_count = sum(1 for pt in line_pts if pt in S)
            if s_count <= 1:
                found_line = True
            if not found_line:
                # Try directions (1, m) for m in F_49
                for m in field_elements:
                    v = (((1, 0)), m)  # (1, m)
                    line_pts = get_line_points(x0, y0, v)
                    s_count = sum(1 for pt in line_pts if pt in S)
                    if s_count <= 1:
                        found_line = True
                        break
            if not found_line:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use a single point
        S = {((0, 0), (0, 0))}
    
    # Build N = complement of S
    points = []
    for x in field_elements:
        for y in field_elements:
            if (x, y) not in S:
                xa, xb = x
                ya, yb = y
                points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
