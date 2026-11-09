import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # Generate candidate centers from hexagonal lattice
    # For a side-3 hexagon (19 hexagons), axial coords satisfy max(|q|,|r|,|q+r|) <= 2
    # We'll also try larger radii for better packing
    centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
    
    # Sort by distance from origin to pick innermost 12
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:N]
    
    # Precompute unit hexagon vertices (side 1, circumradius 1)
    angles_unit = np.array([k * np.pi / 3.0 for k in range(6)])
    verts_unit = np.column_stack((np.cos(angles_unit), np.sin(angles_unit)))
    
    # For non-overlap of two unit hexagons with same orientation:
    # Minimum center distance = 2 * inradius = sqrt(3)
    min_dist = sqrt3 + 1e-6
    min_dist_sq = min_dist * min_dist
    
    def objective(p):
        """Minimize the maximum distance from origin to any vertex of any inner hexagon."""
        # p is flattened (N*2,)
        P = p.reshape(N, 2)
        # All vertices for all hexagons (same orientation, angle=0)
        # Vertex i,j = P[i] + verts_unit[j]
        # dist_sq = |P[i] + verts_unit[j]|^2
        diff = P[:, np.newaxis, :] + verts_unit[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        return np.max(dist_sq)
    
    def objective_with_grad(p):
        P = p.reshape(N, 2)
        diff = P[:, np.newaxis, :] + verts_unit[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        max_val = np.max(dist_sq)
        # Find which (i,j) gives the max
        max_idx = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
        i, j = max_idx
        # Gradient wrt P[i] is 2*(P[i] + verts_unit[j])
        grad = np.zeros(N * 2)
        grad[2*i] = 2 * (P[i, 0] + verts_unit[j, 0])
        grad[2*i+1] = 2 * (P[i, 1] + verts_unit[j, 1])
        return max_val, grad
    
    # Constraints: non-overlap
    def make_constraint(i, j):
        def con(p):
            P = p.reshape(N, 2)
            diff = P[i] - P[j]
            return min_dist_sq - np.dot(diff, diff)
        return con
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = [(-6.0, 6.0)] * (N * 2)
    
    # Initial position: scale the 12 selected centers
    p0 = np.zeros(N * 2)
    scale = 1.0
    for i, (x, y) in enumerate(selected_centers):
        p0[2*i] = x * scale
        p0[2*i+1] = y * scale
    
    best_side = float('inf')
    best_p = p0.copy()
    
    # First pass: SLSQP optimization
    try:
        res = minimize(objective, p0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
        if res.fun < best_side:
            best_side = res.fun
            best_p = res.x.copy()
    except Exception:
        pass
    
    # Second pass: try different initial configurations
    # Try rotating the initial configuration
    for trial_angle in [0, math.pi/6, math.pi/3, math.pi/4, math.pi/12]:
        if time.time() > deadline:
            break
        
        p0_rot = np.zeros(N * 2)
        cos_a = math.cos(trial_angle)
        sin_a = math.sin(trial_angle)
        for i, (x, y) in enumerate(selected_centers):
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            p0_rot[2*i] = xr
            p0_rot[2*i+1] = yr
        
        try:
            res = minimize(objective, p0_rot, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_side:
                best_side = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Third pass: try COBYLA for potentially different local minima
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_p, method='COBYLA', 
                          constraints=constraints, options={'maxiter': 500, 'rhobeg': 0.1})
            if res.fun < best_side:
                best_side = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Compute the outer hexagon side length
    # For a regular hexagon centered at origin, a point at distance d from center
    # with angle theta requires the hexagon to have inradius >= d * |cos(theta - k*pi/3)|
    # The minimum side length L such that all vertices fit:
    # For a hexagon with side L, the inradius is L*sqrt(3)/2
    # A vertex at polar (d, theta) fits if d*max_k |cos(theta - k*pi/3)| <= L*sqrt(3)/2
    # So L >= d * max_k |cos(theta - k*pi/3)| * 2/sqrt(3)
    
    P = best_p.reshape(N, 2)
    # All vertices
    all_verts = P[:, np.newaxis, :] + verts_unit[np.newaxis, :, :]
    # For each vertex, compute the required side length
    dists = np.sqrt(np.sum(all_verts**2, axis=2))
    thetas = np.arctan2(all_verts[:, :, 1], all_verts[:, :, 0])
    
    # For a regular hexagon with flat top (angle 0), the support function is:
    # h(theta) = L * sqrt(3)/2 / cos((theta mod pi/3) - pi/6) for theta near 0
    # More precisely, for a hexagon with vertices at angles k*pi/3,
    # the distance from center to edge at angle phi is L*sqrt(3)/(2*cos(phi - k*pi/3))
    # where k is chosen so that |phi - k*pi/3| <= pi/6
    
    # Required side: for a point at (d, theta), the hexagon side must satisfy
    # d <= L * sqrt(3) / (2 * cos(delta)) where delta = angle to nearest edge normal
    # Edge normals are at angles (k+1/2)*pi/3 for k=0..5, i.e., pi/6, pi/2, 5pi/6, ...
    
    # Actually, for a regular hexagon with side s, the inradius (apothem) is s*sqrt(3)/2
    # The support function h(phi) = apothem / cos(phi - phi_k) where phi_k is the nearest edge normal
    # Edge normals are at angles pi/6 + k*pi/3 for k=0..5
    
    edge_normals = np.array([math.pi/6 + k * math.pi/3 for k in range(6)])
    
    max_required_side = 0
    for i in range(N):
        for j in range(6):
            d = dists[i, j]
            theta = thetas[i, j]
            if d < 1e-10:
                continue
            # Find closest edge normal
            diffs = np.abs(theta - edge_normals)
            diffs = np.minimum(diffs, 2*math.pi - diffs)
            min_diff = np.min(diffs)
            if min_diff > math.pi/6 - 1e-10:
                # Shouldn't happen for valid hexagon, but handle gracefully
                min_diff = math.pi/6
            required = d * 2.0 / (sqrt3 * math.cos(min_diff))
            max_required_side = max(max_required_side, required)
    
    side = max_required_side
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([best_p[2*i], best_p[2*i+1], 0.0])
    
    outer_center = (0.0, 0.0)
    angle = 0.0
    
    return inner, outer_center, side, angle
