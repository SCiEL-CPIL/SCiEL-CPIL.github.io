"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_smooth(vertices_list, beta=50.0):
        """Compute the minimum side length S of the outer hexagon using a smooth max (log-sum-exp)."""
        # Outer hexagon centered at 0,0, rot 0.
        # Normals are at pi/6 + k*pi/3.
        # S = max |proj| / apothem
        # proj = v . n
        # We want max over all v in all hexes, all normals k of |v . n_k|
        # This is equivalent to max of (v . n_k, -v . n_k) for all v, k.
        
        max_val = 0.0
        # Collect all projections
        # To avoid memory issues, we can iterate, but log-sum-exp needs the sum.
        # We can accumulate the log-sum-exp online.
        # log(sum exp(x)) = max(x) + log(sum exp(x - max(x)))
        
        # Let's just collect the absolute projections into a list/array if N is small (11*6*6 = 396 values).
        projs = []
        for verts in vertices_list:
            for v in verts:
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
                    projs.append(abs(proj))
        
        projs = np.array(projs)
        if len(projs) == 0:
            return 0.0
            
        # Smooth max
        m = np.max(projs)
        if m == 0:
            return 0.0
        # log-sum-exp: m + log(sum(exp(projs - m)))
        # sum(exp(projs - m)) is sum of exp of numbers <= 0.
        s = np.sum(np.exp(projs - m))
        smooth_max = m + np.log(s)
        
        apothem = np.sqrt(3) / 2
        return smooth_max / apothem

    def closest_point_on_segment(p, a, b):
        """Find the closest point on segment ab to point p."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab

    def min_distance_convex(poly1, poly2):
        """
        Compute the minimum distance between two convex polygons.
        Returns negative value if they overlap (approximated by max penetration depth).
        Actually, standard min distance is >= 0.
        To handle overlap, we check SAT first. If they overlap, we return a negative penalty.
        For differentiability, we can return the minimum distance if disjoint, 
        and a measure of overlap if intersecting.
        
        A simpler differentiable proxy for overlap:
        If they overlap, return - (minimum penetration depth).
        Penetration depth can be approximated by the minimum over all SAT axes of the overlap length.
        This is the "minimum of maximums" which is non-smooth.
        
        Let's use the vertex-edge distance for disjoint, and for overlap, 
        we can just use a large constant penalty if SAT detects overlap, 
        but that's not differentiable.
        
        Alternative: Use the **squared distance** between the two polygons.
        If they overlap, distance is 0. We need to push them apart.
        If we use `max(0, -dist)^2`, we need `dist` to be negative when overlapping.
        
        Let's compute the "signed distance" using SAT.
        For each axis, we have an interval [min1, max1] and [min2, max2].
        If they are separated, gap = min(min1 - max2, min2 - max1).
        If they overlap, gap is negative?
        Overlap length on axis k = min(max1, max2) - max(min1, min2).
        If Overlap > 0 for all axes, they intersect.
        The minimum penetration depth is min(Overlap_k).
        
        Let's compute `min_overlap` over all axes.
        If `min_overlap` > 0, they are disjoint? No, if min_overlap > 0, it means there is at least one axis where they are separated?
        Wait.
        SAT: If there exists an axis where the projections are disjoint, they are disjoint.
        If for ALL axes, the projections overlap, they intersect.
        
        So, for each axis, calculate the gap.
        gap_k = max(min1 - max2, min2 - max1).
        If gap_k > 0, they are separated along axis k.
        If gap_k < 0, they overlap along axis k.
        
        If ANY gap_k > 0, the polygons are disjoint. The minimum distance is min(gap_k) over all k?
        Actually, the minimum distance is the minimum over all axes of the gap?
        No, the minimum distance is the minimum over all axes of the gap ONLY IF the axis is the separating axis.
        Actually, the minimum distance between two convex polygons is the minimum over all axes (from both polygons) of the gap.
        
        So, `dist = min(gap_k)`.
        If `dist > 0`, they are disjoint.
        If `dist <= 0`, they overlap.
        
        This `dist` function is continuous but not differentiable at 0 (kink).
        We can use `smooth_min` or just use `dist` directly in a penalty like `max(0, -dist)^2`.
        `max(0, x)` is not differentiable at 0.
        We can use `softplus` or similar.
        `penalty = (softplus(-dist * beta) / beta)^2`?
        
        Let's just use `dist` and check if it's negative.
        For the optimizer, we will use `method='BFGS'`.
        We will define the objective as `S_smooth + penalty`.
        `penalty = sum( max(0, -dist_ij)^2 )`.
        To make `max(0, x)` smooth, we can use `0.5 * (x + sqrt(x^2 + eps))`.
        
        """
        gaps = []
        polys = [poly1, poly2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                # Normalize normal to get true distance? 
                # SAT gap is in units of the normal vector length.
                # If normal is not unit, gap is scaled.
                # Let's normalize.
                norm_len = np.linalg.norm(normal)
                if norm_len == 0: continue
                normal = normal / norm_len
                
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                # Gap is positive if separated.
                # Separated if max1 < min2 or max2 < min1.
                gap = min(min2 - max1, min1 - max2)
                # If gap > 0, separated.
                # If gap < 0, overlapping.
                gaps.append(gap)
        
        # The minimum distance is the minimum gap.
        # If min gap > 0, distance is min gap.
        # If min gap < 0, they overlap.
        dist = min(gaps)
        return dist

    def objective(params):
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_smooth(vertices_list)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Quick check: if centers are far, skip
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist_centers = np.linalg.norm(c1 - c2)
                if dist_centers > 2.5: # Max distance for overlap is 2*radius = 2
                    continue
                
                d = min_distance_convex(vertices_list[i], vertices_list[j])
                if d < 0:
                    # Smooth penalty for overlap
                    # Use softplus approximation for max(0, -d)
                    # max(0, x) approx 0.5 * (x + sqrt(x^2 + eps))
                    # Here x = -d
                    x = -d
                    smooth_overlap = 0.5 * (x + np.sqrt(x*x + 1e-6))
                    penalty += 100.0 * smooth_overlap**2
        
        return S + penalty

    def jac(params):
        # Numerical gradient for BFGS if analytical is too hard
        # BFGS can use numerical gradients if jac is not provided?
        # No, BFGS requires jac or uses numerical approximation if not provided?
        # scipy.optimize.minimize BFGS: "If jac is not provided, it is approximated by finite differences."
        # So we can just return the objective.
        pass

    # Initial guess: 11 hexagons
    # 1 center, 6 ring 1, 4 ring 2
    params = np.zeros(3 * N)
    
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1 (6 hexes)
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
        
    # Ring 2 (4 hexes) - Pick 4 to minimize bounding box
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) in axial?
    # Axial (q,r) -> x = sqrt(3)*(q + r/2), y = 1.5*r
    # (2,0) -> x=2sqrt(3), y=0
    # (0,2) -> x=sqrt(3), y=3
    # (-2,0) -> x=-2sqrt(3), y=0
    # (0,-2) -> x=sqrt(3), y=-3
    # This is a "diamond" shape.
    # Let's try these positions.
    pos2 = [
        (2*np.sqrt(3), 0),
        (np.sqrt(3), 3),
        (-2*np.sqrt(3), 0),
        (np.sqrt(3), -3)
    ]
    for cx, cy in pos2:
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    # Optimize
    # Use BFGS with numerical gradients
    result = minimize(objective, params, method='BFGS', 
                      options={'maxiter': 1000, 'maxfev': 10000})
    
    best_params = result.x
    best_score = result.fun
    
    # Multi-start
    for _ in range(3):
        p = params.copy()
        p += np.random.normal(0, 0.5, size=p.size)
        res = minimize(objective, p, method='BFGS', options={'maxiter': 1000, 'maxfev': 10000})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    opt_params = best_params
    inner = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = np.degrees(opt_params[3 * i + 2])
        inner.append([cx, cy, angle])
    
    # Compute final S (exact, not smooth)
    vertices_list = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = opt_params[3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
        
    S = 0.0
    apothem = np.sqrt(3) / 2
    for verts in vertices_list:
        for v in verts:
            for k in range(6):
                theta_n = np.pi / 6 + k * np.pi / 3
                proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
                S = max(S, abs(proj))
    S = S / apothem
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
