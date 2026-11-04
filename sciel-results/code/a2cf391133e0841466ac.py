"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21


def _hex_positions_21():
    """Generate 21 positions on a hexagonal lattice, centered at origin, unit spacing."""
    pts = []
    # Ring 0: center
    pts.append((0.0, 0.0))
    # Ring 1: 6 points
    for k in range(6):
        a = k * np.pi / 3.0
        pts.append((np.cos(a), np.sin(a)))
    # Ring 2: 12 points
    for k in range(6):
        a = k * np.pi / 3.0
        pts.append((2 * np.cos(a), 2 * np.sin(a)))
        a2 = a + np.pi / 6.0
        pts.append((np.sqrt(3) * np.cos(a2), np.sqrt(3) * np.sin(a2)))
    pts = np.array(pts[:N], dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _make_rect(w, h):
    """Return half-widths and half-heights for a perimeter-4 rectangle."""
    # Perimeter = 2*(W+H) = 4 => W+H = 2
    # Let W = w, H = h with w+h=2. Use w = W/2, h = H/2 so 2w+2h=2 => w+h=1.
    return w, h


def construct():
    t0 = time.time()
    deadline = 110.0  # safety margin before 120s kill

    hex_pts = _hex_positions_21()
    n = len(hex_pts)

    # Normalize hex lattice to unit "diameter" scale: nearest neighbor distance = 1
    # So each circle has radius r = 0.5 * s where s is the spacing scale.
    # We'll optimize positions x (n,2) to maximize the minimum of:
    #   - pairwise distances / 2  (radius of circle i if all same radius)
    #   - wall clearance
    # But we want to maximize SUM of radii. For a fixed lattice shape, all radii can be equal
    # and the sum is maximized when the minimum clearance is maximized.
    #
    # Strategy: 
    #   1. Choose rectangle aspect ratio (w+h=1, w,h>0).
    #   2. Scale hex lattice to fit, then optimize positions to maximize min clearance.
    #   3. The common radius = min clearance; sum = n * r.
    #
    # We'll optimize positions x (n*2) with objective = -min_clearance (maximize min clearance).
    # min_clearance = min over i<j of (||x_i - x_j|| - 2r) and over i of (wall clearance - r)
    # But r depends on positions... This is non-convex.
    #
    # Simpler: fix r as a variable, maximize r such that all constraints hold.
    # This is a feasibility problem for a given r; we can binary search on r.
    # For a fixed r, constraints are:
    #   ||x_i - x_j|| >= 2r for all i<j
    #   x_i[0] >= -w + r, x_i[0] <= w - r, x_i[1] >= -h + r, x_i[1] <= h - r
    # These are non-convex (pairwise distances) but we can use SLSQP or trust-constr.
    #
    # Alternative: maximize r directly with SLSQP treating r as a variable.
    # Objective: maximize r
    # Variables: x (n*2) + r (1) => (n*2+1) variables
    # Constraints: 
    #   ||x_i - x_j||^2 - (2r)^2 >= 0 for all i<j  (non-convex)
    #   x_i[0] + r - w <= 0, w - x_i[0] - r <= 0, etc. (linear)
    #
    # Let's try SLSQP with multiple aspect ratios.

    best_result = None
    best_val = -1.0

    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    # w + h = 1, so w = ar/(1+ar), h = 1/(1+ar) where ar = w/h
    # Actually let's parameterize: w = t, h = 1-t, t in (0,1)
    t_vals = [0.5, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.25, 0.75, 0.2, 0.8]

    for t in t_vals:
        if time.time() - t0 > deadline:
            break
        w = t
        h = 1.0 - t

        # Initial guess: scale hex_pts to fit in [-w, w] x [-h, h]
        # Hex lattice extent: roughly [-2,2] in x, [-2,2] in y after centering
        scale_x = (w - 0.5) / 2.5  # leave margin
        scale_y = (h - 0.5) / 2.5
        s = min(scale_x, scale_y)
        x0 = hex_pts * s
        x0_flat = x0.flatten()
        r0 = 0.3 * s
        z0 = np.concatenate([x0_flat, [r0]])

        def objective(z):
            r = z[-1]
            return -r

        def constraint_func(z, i, j):
            r = z[-1]
            xi = z[2*i:2*i+2]
            xj = z[2*j:2*j+2]
            d2 = np.sum((xi - xj)**2)
            return d2 - 4*r*r

        def wall_constraints(z, i):
            r = z[-1]
            xi = z[2*i:2*i+2]
            return [
                xi[0] + r - w,      # should be <= 0
                w - xi[0] - r,      # should be <= 0  => xi[0] <= w - r
                xi[1] + r - h,
                h - xi[1] - r,
            ]

        # Build constraints
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i, j=j: constraint_func(z, i, j)})
        for i in range(n):
            for c in range(4):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i, c=c: -wall_constraints(z, i)[c]})

        # Bounds: r >= 0, positions within rectangle
        bounds = []
        for i in range(n):
            bounds.append((-(w), w))
            bounds.append((-(h), h))
        bounds.append((0.0, min(w, h)))

        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_val:
                r = res.x[-1]
                if r > 0:
                    val = n * r
                    if val > best_val:
                        best_val = val
                        best_result = res.x
        except Exception:
            pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract positions and radius
    r = best_result[-1]
    positions = best_result[:n*2].reshape(n, 2)

    # Convert to (cx, cy, r) in [0, W] x [0, H] where W=2w, H=2h
    # Our positions are centered at origin in [-w,w] x [-h,h]
    # Shift to [0, 2w] x [0, 2h]
    positions[:, 0] += w
    positions[:, 1] += h

    result = np.zeros((n, 3))
    result[:, 0] = positions[:, 0]
    result[:, 1] = positions[:, 1]
    result[:, 2] = r

    return result
