"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3.0) / 2.0  # height of equilateral triangle with side 1

# EVOLVE-BLOCK-START

def _to_bary(p):
    """Convert Cartesian points to barycentric coordinates (relative to triangle vertices)."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, H)
    # Barycentric: p = a*A + b*B + c*C, a+b+c=1, a,b,c >= 0
    # x = b + 0.5*c
    # y = H*c
    # So c = y/H, b = x - 0.5*c, a = 1 - b - c
    c = p[:, 1] / H
    b = p[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _from_bary(bary):
    """Convert barycentric coordinates to Cartesian."""
    a = bary[:, 0]
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _min_triangle_area(p):
    """Compute the minimum area of any triangle formed by 3 points from p (n x 2)."""
    n = p.shape[0]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = p[j] - p[i]
                v2 = p[k] - p[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return min(areas)


def _min_triangle_area_fast(p):
    """Vectorized min triangle area computation."""
    n = p.shape[0]
    # Compute all pairwise cross products
    # For each triple (i,j,k), area = 0.5 * |det([pj-pi, pk-pi])|
    # Use broadcasting
    min_area = np.inf
    for i in range(n):
        vi = p[i]
        # All vectors from p[i] to others
        diffs = p - vi  # (n, 2)
        # Cross product of all pairs j<k where j,k != i
        # For fixed i, we need cross(diffs[j], diffs[k]) for all j<k, j,k != i
        idx = [m for m in range(n) if m != i]
        if len(idx) < 2:
            continue
        sub = diffs[idx]  # (n-1, 2)
        # All pairwise cross products
        # cross(sub[a], sub[b]) = sub[a,0]*sub[b,1] - sub[a,1]*sub[b,0]
        x = sub[:, 0]
        y = sub[:, 1]
        # Outer cross: (n-1) x (n-1) matrix
        cross = np.abs(np.outer(x, y) - np.outer(y, x))
        # Only take upper triangle (j < k)
        iu = np.triu_indices(len(idx), k=1)
        vals = cross[iu]
        if len(vals) > 0:
            local_min = np.min(vals) * 0.5
            if local_min < min_area:
                min_area = local_min
    return min_area


def _penalty(bary, p, margin=0.001):
    """Penalty for points outside the triangle (barycentric coords should be >= 0)."""
    return np.sum(np.clip(-bary, 0, None) ** 2) + margin * 0  # just boundary penalty


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin from 150s limit

    # Strategy: 3x3 barycentric grid + 2 free points
    # The 3x3 grid in barycentric space: positions (i/4, j/4, k/4) where i+j+k=4
    # Actually, let's use a more standard approach:
    # 9 grid points at barycentric (i/2, j/2, k/2) for i+j+k=2 gives 6 points
    # Let's use a different parameterization.
    
    # Better approach: Use differential evolution on 22 free variables (11 points x 2 coords)
    # with a penalty for being outside the triangle.
    
    def objective_flat(x):
        """Objective: minimize negative of min triangle area (with penalty for out-of-bounds)."""
        elapsed = time.time() - t0
        if elapsed > deadline - 5:
            return 1e10  # signal timeout
        
        p = x.reshape(11, 2)
        # Check if all points are inside the triangle
        # Point (x,y) is inside if: y >= 0, y <= H*x*2 (left edge from (0,0) to (0.5,H): y = 2*H*x... wait)
        # Edge AB: from (0,0) to (1,0): y >= 0
        # Edge BC: from (1,0) to (0.5,H): line is y = -2*H*(x-1) = 2*H*(1-x)
        #   So y <= 2*H*(1-x)
        # Edge CA: from (0.5,H) to (0,0): line is y = 2*H*x
        #   So y <= 2*H*x
        
        # Actually let me recompute:
        # A=(0,0), B=(1,0), C=(0.5, H)
        # Edge AB: y=0, interior is y >= 0
        # Edge BC: from B(1,0) to C(0.5, H). Slope = (H-0)/(0.5-1) = H/(-0.5) = -2H
        #   Line: y - 0 = -2H(x - 1) => y = -2Hx + 2H => y = 2H(1-x)
        #   Interior: y <= 2H(1-x)
        # Edge CA: from C(0.5, H) to A(0,0). Slope = (0-H)/(0-0.5) = -H/(-0.5) = 2H
        #   Line: y - 0 = 2H(x - 0) => y = 2Hx
        #   Interior: y <= 2Hx
        
        y = p[:, 1]
        x = p[:, 0]
        
        # Constraints:
        # y >= 0
        # y <= 2*H*(1-x)  =>  y + 2*H*x <= 2*H
        # y <= 2*H*x       =>  y - 2*H*x <= 0
        
        viol1 = np.clip(-y, 0, None)
        viol2 = np.clip(y + 2*H*x - 2*H, 0, None)
        viol3 = np.clip(y - 2*H*x, 0, None)
        
        penalty = 100.0 * (np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2))
        
        if penalty > 1e-6:
            return 1e10 + penalty
        
        min_area = _min_triangle_area_fast(p)
        return -min_area
    
    # Initial bounds for each point: x in [0,1], y in [0,H]
    bounds = [(0.0, 1.0), (0.0, H)] * 11
    
    # Run differential evolution
    best_result = None
    best_score = -np.inf
    
    try:
        de_result = differential_evolution(
            objective_flat,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mut=(0.3, 1.0),
            reccomb=0.8,
            polish=True,
            updating='deferred',
            init='sobol',
            seed=42
        )
        if de_result.success or de_result.fun < 0:
            best_result = de_result.x
            best_score = -de_result.fun
    except Exception:
        pass
    
    # Additional local optimization with L-BFGS-B
    if best_result is not None:
        elapsed = time.time() - t0
        if elapsed < deadline - 10:
            try:
                def obj_l(x):
                    p = x.reshape(11, 2)
                    y = p[:, 1]
                    x_c = p[:, 0]
                    viol1 = np.clip(-y, 0, None)
                    viol2 = np.clip(y + 2*H*x_c - 2*H, 0, None)
                    viol3 = np.clip(y - 2*H*x_c, 0, None)
                    penalty = 1000.0 * (np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2))
                    if penalty > 1e-4:
                        return 1e10 + penalty
                    min_area = _min_triangle_area_fast(p)
                    return -min_area
                
                res = minimize(
                    obj_l,
                    best_result,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < obj_l(best_result):
                    best_result = res.x
                    best_score = -res.fun
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        return np.array(points)
    
    p = best_result.reshape(11, 2)
    return p

# EVOLVE-BLOCK-END
