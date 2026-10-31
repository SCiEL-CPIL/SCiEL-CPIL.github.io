"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_sum_r = 0.0
    best_result = None
    
    # We search over aspect ratios.
    # Perimeter 4 => 2(w+h)=4 => h = 2-w.
    # Due to symmetry, w and h are interchangeable. We can assume w <= 1.0 (so w <= h).
    # However, to be safe, we will sample w in [0.5, 1.5] and rely on the optimizer.
    # Actually, since w+h=2, if w>1, h<1. The problem is symmetric under rotation.
    # So we can restrict w to [0.5, 1.0] and cover all unique rectangles.
    
    # Vectorized unpacking helper
    def unpack(v):
        # v: [x1, y1, r1, ..., xN, yN, rN]
        xs = v[0:3*N:3]
        ys = v[1:3*N:3]
        rs = v[2:3*N:3]
        return xs, ys, rs

    def make_constraints(w, h):
        # Returns a list of constraint dictionaries for SLSQP
        # Constraints must be >= 0
        
        def cons_inside(v):
            xs, ys, rs = unpack(v)
            # x - r >= 0
            # w - x - r >= 0
            # y - r >= 0
            # h - y - r >= 0
            return np.concatenate([
                xs - rs,
                w - xs - rs,
                ys - rs,
                h - ys - rs
            ])

        def cons_non_overlap(v):
            xs, ys, rs = unpack(v)
            # dist_ij - ri - rj >= 0
            # To vectorize, we can compute all pairwise distances.
            # However, for N=21, N*(N-1)/2 = 210 constraints.
            # A fully vectorized approach using broadcasting creates a 21x21 matrix.
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = rs[:, None] + rs[None, :]
            
            # We only need i < j
            # dist[i, j] - min_dist[i, j] >= 0 for i < j
            # We can flatten the upper triangle
            i_idx, j_idx = np.triu_indices(N, k=1)
            return dist[i_idx, j_idx] - min_dist[i_idx, j_idx]

        def cons_positive_r(v):
            _, _, rs = unpack(v)
            return rs

        return [
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    def objective(v):
        _, _, rs = unpack(v)
        return -np.sum(rs)

    def validate(xs, ys, rs, w, h, tol=1e-8):
        if np.any(rs < 0):
            return False
        if np.any(xs < rs - tol) or np.any(xs > w - rs + tol):
            return False
        if np.any(ys < rs - tol) or np.any(ys > h - rs + tol):
            return False
        
        # Check overlaps
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = rs[:, None] + rs[None, :]
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        if np.any(dist[i_idx, j_idx] < min_dist[i_idx, j_idx] - tol):
            return False
        return True

    # Search over aspect ratios
    # w in [0.5, 1.0] covers all unique rectangles due to symmetry
    w_values = np.linspace(0.5, 1.0, 12)
    
    # For each w, try multiple initializations
    # Init types: Grid, Hexagonal, Random
    
    for w in w_values:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 - w
        constraints = make_constraints(w, h)
        
        # Try 3 different initializations for this aspect ratio
        for init_type in ['grid', 'hex', 'random']:
            if time.time() > deadline - 5:
                break
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            
            if init_type == 'grid':
                # 5x5 grid is 25 spots, take first 21
                # 4x6 grid is 24 spots
                # Let's use a 5x5 grid centered in the rectangle
                cols = 5
                rows = 5
                # Adjust spacing to fit N=21
                # We'll just place 21 points in a 5x5 grid pattern
                # Row 0: 5, Row 1: 5, Row 2: 5, Row 3: 5, Row 4: 1 (centered)
                # Or simpler: 4 rows of 5 + 1 row of 1?
                # Let's stick to a dense grid and let the optimizer sort it out.
                # 5 cols x 5 rows
                x_sp = w / 5.0
                y_sp = h / 5.0
                count = 0
                for r in range(5):
                    for c in range(5):
                        if count < N:
                            xs_init[count] = (c + 0.5) * x_sp
                            ys_init[count] = (r + 0.5) * y_sp
                            count += 1
            elif init_type == 'hex':
                # Hexagonal packing
                # Approximate rows
                # Area per circle ~ pi*r^2. 
                # Let's estimate r ~ min(w, h) / 5 * 0.8
                est_r = min(w, h) / 5.0 * 0.8
                # Vertical spacing for hex is sqrt(3)*r
                # Horizontal spacing is 2*r
                # Number of rows
                n_rows = int(np.ceil(h / (np.sqrt(3) * est_r)))
                if n_rows < 3: n_rows = 3
                if n_rows > 8: n_rows = 8
                
                # Fill rows
                count = 0
                for r in range(n_rows):
                    y = (r + 0.5) * np.sqrt(3) * est_r
                    if y > h - est_r: break
                    # Offset odd rows
                    offset = 0.0
                    if r % 2 == 1:
                        offset = est_r
                    # Number of circles in this row
                    x_start = est_r + offset
                    x_end = w - est_r
                    if x_start > x_end:
                        continue
                    n_in_row = int((x_end - x_start) / (2 * est_r)) + 1
                    for c in range(n_in_row):
                        if count >= N: break
                        x = x_start + c * 2 * est_r
                        xs_init[count] = x
                        ys_init[count] = y
                        count += 1
                # If we didn't fill all, fill remaining with random
                if count < N:
                    xs_init[count:] = np.random.uniform(est_r, w-est_r, N-count)
                    ys_init[count:] = np.random.uniform(est_r, h-est_r, N-count)
            else: # random
                # Random initialization with small radii
                xs_init = np.random.uniform(0.1, w-0.1, N)
                ys_init = np.random.uniform(0.1, h-0.1, N)

            # Initial radii: small enough to be valid, large enough to grow
            # Estimate max possible r
            # Min distance to wall
            min_wall_dist = np.min([np.min(xs_init), np.min(w-xs_init), np.min(ys_init), np.min(h-ys_init)])
            # Min distance between circles
            dx = xs_init[:, None] - xs_init[None, :]
            dy = ys_init[:, None] - ys_init[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            min_circ_dist = np.min(dist)
            
            r_init = 0.4 * min(min_wall_dist, min_circ_dist)
            if r_init < 0.01: r_init = 0.01
            rs_init = np.full(N, r_init)
            
            v0 = np.zeros(3*N)
            v0[0:3*N:3] = xs_init
            v0[1:3*N:3] = ys_init
            v0[2:3*N:3] = rs_init
            
            try:
                # Use SLSQP
                # maxiter: 500 is usually enough for this size if well-initialized
                # ftol: 1e-8 for high precision
                res = minimize(
                    objective, v0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if res.x is not None:
                    xs, ys, rs = unpack(res.x)
                    # Validate the result
                    if validate(xs, ys, rs, w, h):
                        sum_r = np.sum(rs)
                        if sum_r > best_sum_r:
                            best_sum_r = sum_r
                            best_result = np.column_stack([xs, ys, rs])
                
            except Exception:
                pass

    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Simple grid in square
        w = 1.0
        h = 1.0
        xs = []
        ys = []
        rs = []
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                if len(xs) < N:
                    xs.append((i+0.5)*0.2)
                    ys.append((j+0.5)*0.2)
                    rs.append(0.09) # Slightly less than 0.1
        best_result = np.column_stack([xs, ys, rs])

    return best_result

# EVOLVE-BLOCK-END
