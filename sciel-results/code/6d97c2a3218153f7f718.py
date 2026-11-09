import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field arithmetic
    # alpha^2 = 3
    k = 3
    
    # Precompute field elements
    # Represented as (a, b) for a + b*alpha
    # We need to iterate over all points in F_49^2
    # Point: ((xa, xb), (ya, yb))
    
    # Construct the Unital U
    # Equation: 2*xa + ya^2 - k*yb^2 = 0 (mod p)
    # xa is determined by ya, yb.
    # xb is free.
    
    U = set()
    for ya in range(p):
        for yb in range(p):
            # Calculate required xa
            # 2*xa = k*yb^2 - ya^2
            rhs = (k * yb * yb - ya * ya) % p
            # 2*xa = rhs => xa = rhs * inv(2)
            # inv(2) mod 7 is 4 (2*4=8=1)
            xa = (rhs * 4) % p
            for xb in range(p):
                U.add(((xa, xb), (ya, yb)))
                
    # Check validity of C = U
    # A point x is "covered" if there exists a direction m
    # such that the line L_x,m intersects U only at x.
    
    # Directions:
    # m in F_49 (49 values) and infinity (vertical)
    # We represent m as (ma, mb) or None for infinity
    
    def get_line_points(x, m):
        """
        x: ((xa, xb), (ya, yb))
        m: (ma, mb) or None
        Returns list of points on the line, excluding x.
        """
        xa, xb = x[0]
        ya, yb = x[1]
        points = []
        for t_a in range(p):
            for t_b in range(p):
                if t_a == 0 and t_b == 0:
                    continue
                t = (t_a, t_b)
                if m is None:
                    # v = (0, 1)
                    # x' = x, y' = y + t
                    nx_a = xa
                    nx_b = xb
                    ny_a = (ya + t_a) % p
                    ny_b = (yb + t_b) % p
                else:
                    ma, mb = m
                    # v = (1, m)
                    # x' = x + t
                    # y' = y + t*m
                    nx_a = (xa + t_a) % p
                    nx_b = (xb + t_b) % p
                    # y + t*m = (ya + b*alpha) + (ta + tb*alpha)(ma + mb*alpha)
                    # = ya + b*alpha + (ta*ma + tb*mb*k) + (ta*mb + tb*ma)*alpha
                    prod_a = (t_a * ma + t_b * mb * k) % p
                    prod_b = (t_a * mb + t_b * ma) % p
                    ny_a = (ya + prod_a) % p
                    ny_b = (yb + prod_b) % p
                points.append(((nx_a, nx_b), (ny_a, ny_b)))
        return points

    # This is too slow if we generate points for every check.
    # Better: For a given x and m, check if ANY point on the line is in U.
    
    def is_line_disjoint_except_x(x, m, C):
        xa, xb = x[0]
        ya, yb = x[1]
        for t_a in range(p):
            for t_b in range(p):
                if t_a == 0 and t_b == 0:
                    continue
                if m is None:
                    nx_a = xa
                    nx_b = xb
                    ny_a = (ya + t_a) % p
                    ny_b = (yb + t_b) % p
                else:
                    ma, mb = m
                    nx_a = (xa + t_a) % p
                    nx_b = (xb + t_b) % p
                    prod_a = (t_a * ma + t_b * mb * k) % p
                    prod_b = (t_a * mb + t_b * ma) % p
                    ny_a = (ya + prod_a) % p
                    ny_b = (yb + prod_b) % p
                
                if ((nx_a, nx_b), (ny_a, ny_b)) in C:
                    return False
        return True

    bad_points = set()
    
    # Iterate over all points in U
    for x in U:
        covered = False
        # Check all 49+1 directions
        for ma in range(p):
            for mb in range(p):
                m = (ma, mb)
                if is_line_disjoint_except_x(x, m, U):
                    covered = True
                    break
            if covered:
                break
        if not covered:
            # Check vertical
            if is_line_disjoint_except_x(x, None, U):
                covered = True
        
        if not covered:
            bad_points.add(x)
            
    C = U - bad_points
    
    # Construct N
    N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    if pt not in C:
                        N.append([[xa, xb], [ya, yb]])
                        
    return np.array(N, dtype=np.int64)
