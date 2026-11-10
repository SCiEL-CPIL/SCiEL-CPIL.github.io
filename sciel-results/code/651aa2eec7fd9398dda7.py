# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Hermitian unital baseline
    with efficient greedy refinement using coverage counting.
    """
    p = 7
    q = p * p
    deadline = time.time() + 580
    
    k_const = 3
    
    # Field element: (a, b) represents a + b*alpha, alpha^2 = 3
    # Point: (xa, xb, ya, yb)
    
    # Precompute all field elements as flat indices 0..48
    # flat_idx = a + b*p
    def f_to_flat(f):
        return f[0] + f[1] * p
    
    def flat_to_f(idx):
        return (idx % p, idx // p)
    
    def f_add(f1, f2):
        return ((f1[0] + f2[0]) % p, (f1[1] + f2[1]) % p)
    
    def f_mul(f1, f2):
        return ((f1[0]*f2[0] + f1[1]*f2[1]*k_const) % p,
                (f1[0]*f2[1] + f1[1]*f2[0]) % p)
    
    def f_mul_flat(f1_idx, f2_idx):
        f1 = flat_to_f(f1_idx)
        f2 = flat_to_f(f2_idx)
        return f_to_flat(f_mul(f1, f2))
    
    # Point as flat index: xa + xb*p + ya*p^2 + yb*p^3
    def pt_to_flat(pt):
        return pt[0] + pt[1]*p + pt[2]*p*p + pt[3]*p*p*p
    
    def flat_to_pt(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p*p)) % p
        yb = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    # Generate all points
    all_pts_flat = list(range(q*q))
    all_pts = [flat_to_pt(i) for i in all_pts_flat]
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Direction as (vxa, vxb, vya, vyb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 1, 0, 0))
    n_dirs = len(directions)
    
    # Precompute for each direction and each t != 0: the offset (dx_a, dx_b, dy_a, dy_b)
    # t*v where t = (ta, tb), v = (vxa, vxb, vya, vyb)
    # t*v = (ta*vxa + tb*vxb*k, ta*vxb + tb*vxa, ta*vya + tb*vyb*k, ta*vyb + tb*vya)
    offsets = []  # offsets[d][t_idx] = (da, db, dc, dd)
    for d in range(n_dirs):
        vxa, vxb, vya, vyb = directions[d]
        d_offsets = []
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                da = (ta*vxa + tb*vxb*k_const) % p
                db = (ta*vxb + tb*vxa) % p
                dc = (ta*vya + tb*vyb*k_const) % p
                dd = (ta*vyb + tb*vya) % p
                d_offsets.append((da, db, dc, dd))
        offsets.append(d_offsets)
    
    # Precompute for each point index and each direction: the list of other point indices on the line
    # line_pts[pt_idx][d] = list of (other_pt_idx, t_idx) 
    # Actually, let's precompute: for each direction d and point pt, the set of other points
    # This is 2401 * 50 * 48 = too much memory? 2401*50*48 = 5,762,400 entries. That's fine.
    
    # Actually, let's compute on the fly but cache the line membership.
    # For each direction d, the line through point pt contains pt + t*v for all t.
    # Two points are on the same line in direction d iff their difference is in the span of v.
    
    # Better: precompute for each direction d, a mapping from (line_id, pt_idx) to line_id
    # For direction v=(1,m): lines are parameterized by the y-intercept.
    # For direction v=(0,1): lines are vertical, parameterized by x.
    
    # Let's precompute line_id for each point in each direction.
    # For v = (1, m): line through (x,y) has equation y = m*(x - x0) + y0, so y - m*x = y0 - m*x0
    # The line id is (y - m*x) in F_49
    # For v = (0, 1): line id is x in F_49
    
    # line_id[pt_idx][d] = flat index of the line identifier
    line_id = np.zeros((q*q, n_dirs), dtype=np.int32)
    
    for d in range(n_dirs):
        vxa, vxb, vya, vyb = directions[d]
        for pt_idx in range(q*q):
            xa, xb, ya, yb = flat_to_pt(pt_idx)
            if vxa == 0:  # v = (0, 1)
                # Line id = x = (xa, xb)
                lid = xa + xb * p
            else:
                # v = (1, m) where m = (vya, vyb)
                # Line id = y - m*x
                # y - m*x = (ya, yb) - (vya, vyb)*(xa, xb)
                mxa = (vya * xa + vyb * xb * k_const) % p
                mxb = (vya * xb + vyb * xa) % p
                lda = (ya - mxa) % p
                ldb = (yb - mxb) % p
                lid = lda + ldb * p
            line_id[pt_idx, d] = lid
    
    # For each direction d and each line_id, get the list of points on that line
    # line_points[d][lid] = list of point indices
    line_points = [dict() for _ in range(n_dirs)]
    for d in range(n_dirs):
        for pt_idx in range(q*q):
            lid = line_id[pt_idx, d]
            if lid not in line_points[d]:
                line_points[d][lid] = []
            line_points[d][lid].append(pt_idx)
    
    # Verify: each line should have exactly q = 49 points
    for d in range(n_dirs):
        for lid, pts in line_points[d].items():
            assert len(pts) == q, f"Line {lid} in dir {d} has {len(pts)} points, expected {q}"
    
    # Baseline: Hermitian unital construction
    # C = { (x,y) : 2*xa + ya^2 - yb^2*3 = 0 mod 7, and (ya, yb) not in Y_special }
    # where Y_special = { (u^2 mod 7, v) : u,v in 0..6 }
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    C = set()
    for pt_idx in range(q*q):
        xa, xb, ya, yb = flat_to_pt(pt_idx)
        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            C.add(pt_idx)
    
    # Now, for each point not in C, check if it can be added to C.
    # A point pt can be added to C if for every direction d, there exists t != 0
    # such that pt + t*v is in C.
    
    # Efficient check: for each direction d, the line through pt in direction d has
    # q-1 = 48 other points. We need at least one of them to be in C.
    
    # Precompute: for each direction d and point pt_idx, count how many other points on the line are in C.
    # coverage[pt_idx][d] = count of points in C on the line through pt_idx in direction d, excluding pt_idx itself.
    
    coverage = np.zeros((q*q, n_dirs), dtype=np.int32)
    
    def update_coverage(pt_idx, add=True):
        """Update coverage counts when pt_idx is added to or removed from C."""
        sign = 1 if add else -1
        for d in range(n_dirs):
            lid = line_id[pt_idx, d]
            for other_idx in line_points[d][lid]:
                if other_idx != pt_idx:
                    coverage[other_idx, d] += sign
    
    # Initialize coverage
    for pt_idx in C:
        update_coverage(pt_idx, add=True)
    
    # A point pt can be added to C if min_d coverage[pt][d] >= 1
    # i.e., for every direction, at least one other point on the line is in C.
    
    # Greedily add points: repeatedly find a point not in C with min coverage >= 1,
    # add it to C, and update coverage.
    
    # To be efficient, maintain a list of candidate points and their min coverage.
    # When we add a point to C, we update coverage for all points on the same lines.
    
    in_C = np.zeros(q*q, dtype=bool)
    for pt_idx in C:
        in_C[pt_idx] = True
    
    # Candidate points: those not in C
    # We'll iterate until no more points can be added.
    
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        if time.time() > deadline - 5:
            break
        
        # Find all points that can be added
        # min_cov = minimum over directions of coverage
        min_cov = np.min(coverage, axis=1)
        
        # Candidates: not in C, and min_cov >= 1
        candidates = np.where((~in_C) & (min_cov >= 1))[0]
        
        if len(candidates) == 0:
            break
        
        # Add all candidates at once? No, that might break the condition for others.
        # Actually, if we add a point to C, it only increases coverage for others,
        # so it can't break the condition for other points. So we can add all at once!
        
        for pt_idx in candidates:
            in_C[pt_idx] = True
            C.add(int(pt_idx))
            update_coverage(int(pt_idx), add=True)
        
        changed = True
    
    # Build N = all points not in C
    N_indices = [i for i in range(q*q) if not in_C[i]]
    
    # Convert to array
    points = np.array([[[flat_to_pt(i)[0], flat_to_pt(i)[1]], 
                        [flat_to_pt(i)[2], flat_to_pt(i)[3]]] for i in N_indices], dtype=np.int64)
    
    return points

# EVOLVE-BLOCK-END
