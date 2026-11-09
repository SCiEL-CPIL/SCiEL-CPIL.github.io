import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Build field operations
    # Element is (a, b) representing a + b*alpha, alpha^2 = 3
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def field_mul(x, y):
        # (a+bα)(c+dα) = ac + (ad+bc)α + bd·α² = (ac+3bd) + (ad+bc)α
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)
    
    def field_neg(x):
        return ((-x[0])%p, (-x[1])%p)
    
    def field_sub(x, y):
        return field_add(x, field_neg(y))
    
    def field_inv(x):
        # Inverse of a+bα: (a-bα)/(a²-3b²)
        det = (x[0]*x[0] - k*x[1]*x[1]) % p
        if det == 0:
            return None
        det_inv = pow(det, p-2, p)
        return ((x[0]*det_inv)%p, (-x[1]*det_inv)%p)
    
    # Enumerate all field elements
    field_elements = [(a,b) for a in range(p) for b in range(p)]
    field_elements_set = set(field_elements)
    
    # All points in F_49^2
    all_points = [((xa,xb),(ya,yb)) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Build lines in allowed directions
    # Direction (1,m) for m in F_49, and (0,1)
    # A line through point x with direction v: {x + t*v : t in F_49}
    
    # For efficiency, precompute lines
    # Lines with direction (0,1): for each (xa,xb), the line is {(xa,xb,ya,yb) : ya,yb in F_7}
    # That's 7*7=49 points per line, and 7*7=49 such lines
    
    # Lines with direction (1,m): for each starting point and each m, 
    # but many lines are the same. Total lines with direction (1,m) for fixed m: 
    # 49*49/49 = 49 lines (since each line has 49 points and there are 49*49 points total)
    # Total lines: 49 (vertical) + 49*49 (slanted) = 49 + 2401 = 2450 lines
    
    # Actually, for direction (1,m), a line is determined by its "intercept" in y.
    # y = m*x + c for some c in F_49. So 49 lines per m, 49 values of m, total 2401 slanted lines.
    # Plus 49 vertical lines. Total 2450 lines.
    
    # Let's precompute all lines as sets of point indices
    lines = []  # list of sets of point indices
    
    # Vertical lines: direction (0,1)
    for xa in range(p):
        for xb in range(p):
            line_pts = set()
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    line_pts.add(point_to_idx[pt])
            lines.append(line_pts)
    
    # Slanted lines: direction (1,m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # y = m*x + c, for each c in F_49
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = set()
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            # y = m*x + c
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            line_pts.add(point_to_idx[pt])
                    lines.append(line_pts)
    
    # Each line has 49 points
    # For the Nikodym property: for each point x, there exists a line L through x 
    # such that L \ {x} ⊆ N, i.e., (L \ {x}) ∩ C = ∅
    
    # Precompute for each point, which lines pass through it
    point_lines = [[] for _ in range(2401)]
    for li, line in enumerate(lines):
        for pt_idx in line:
            point_lines[pt_idx].append(li)
    
    # Initial complement: Hermitian unital points with ya not in squares
    squares_mod7 = {0, 1, 2, 4}
    C = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
                    if in_unital and ya not in squares_mod7:
                        C.add(point_to_idx[((xa,xb),(ya,yb))])
    
    # Verify Nikodym property for current C
    def verify_nikodym(C_set):
        """Check if N = all_points - C_set is a Nikodym set."""
        C_frozen = frozenset(C_set)
        for pt_idx in range(2401):
            found = False
            for li in point_lines[pt_idx]:
                line = lines[li]
                # Check if all other points on this line are in N (not in C)
                valid = True
                for other in line:
                    if other != pt_idx and other in C_frozen:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Verify initial C
    assert verify_nikodym(C), "Initial C should be valid"
    
    best_C = set(C)
    best_size = len(C)
    
    # Try to grow C greedily
    # For each candidate point, check if adding it to C still maintains the Nikodym property
    # This is expensive, so use a smarter approach
    
    # Alternative: try different blocking sets
    # Try: ya in {0} only (special set = {(0,v) for v in F_7})
    # C = unital points with ya not in {0}
    for special_ya_set in [
        {0},
        {0, 1},
        {0, 1, 2},
        {0, 1, 2, 4},  # current
        {0, 1, 2, 3, 4, 5, 6},  # no exclusion
    ]:
        C_new = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
                        if in_unital and ya not in special_ya_set:
                            C_new.add(point_to_idx[((xa,xb),(ya,yb))])
        
        if len(C_new) > best_size:
            if verify_nikodym(C_new):
                best_C = set(C_new)
                best_size = len(C_new)
        
        if time.time() - start_time > deadline:
            break
    
    # Convert best_C to array
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in best_C:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)
