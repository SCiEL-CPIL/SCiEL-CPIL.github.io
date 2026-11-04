"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11


def _hex_vertices(angle_deg):
    """Vertices of a unit hexagon centered at origin, rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for i in range(6):
        base = i * math.pi / 3.0
        x = ca * math.cos(base) - sa * math.sin(base)
        y = sa * math.cos(base) + ca * math.sin(base)
        verts.append((x, y))
    return np.array(verts)


def _sat_overlap(v1, v2):
    """Return the minimum overlap depth between two convex hexagons using SAT.
    Returns 0 if no overlap, positive if overlapping (depth of penetration)."""
    min_overlap = float('inf')
    for verts in (v1, v2):
        for i in range(6):
            # Edge normal
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
            if overlap < 0:
                return 0.0  # Separating axis found
            min_overlap = min(min_overlap, overlap)
    return min_overlap


def _total_overlap_penalty(centers, angles, smooth=True, beta=50.0):
    """Compute smooth total overlap penalty between all pairs of hexagons.
    centers: (n, 2) array, angles: (n,) array in radians."""
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        v1 = centers[i] + _hex_vertices(np.degrees(angles[i]))
        for j in range(i + 1, n):
            v2 = centers[j] + _hex_vertices(np.degrees(angles[j]))
            # Quick reject: if center distance > 2, no overlap possible
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.05:
                continue
            depth = _sat_overlap(v1, v2)
            if depth > 0:
                if smooth:
                    penalty += math.exp(beta * depth) - 1.0
                else:
                    penalty += depth
    return penalty


def _max_radial_projection(centers, angles, beta=30.0):
    """Compute smooth max radial projection (related to outer hexagon side length).
    The outer hexagon has normals at 0, 60, 120, 180, 240, 300 degrees.
    Side length S = (2/sqrt(3)) * max over all vertices and all normals of |v . n|."""
    k_angles = np.arange(6) * math.pi / 3.0
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)  # (6, 2)
    
    all_dots = []
    for i in range(len(centers)):
        verts = centers[i] + _hex_vertices(np.degrees(angles[i]))
        dots = verts @ normals.T  # (6, 6)
        all_dots.append(np.abs(dots).max(axis=0))  # (6,)
    
    max_per_normal = np.max(np.array(all_dots), axis=0)  # (6,)
    
    if beta > 1:
        # Smooth max via log-sum-exp
        m = max_per_normal.max()
        smooth_max = m + math.log(np.sum(np.exp(beta * (max_per_normal - m)))) / beta
    else:
        smooth_max = max_per_normal.max()
    
    S = (2.0 / math.sqrt(3.0)) * smooth_max
    return S


def _objective(params, beta_outer=30.0, beta_overlap=50.0, weight_overlap=10.0):
    """Combined objective: outer size + overlap penalty."""
    n = N
    centers = params[:2 * n].reshape(n, 2)
    angles = params[2 * n:]
    
    S = _max_radial_projection(centers, angles, beta=beta_outer)
    overlap = _total_overlap_penalty(centers, angles, smooth=True, beta=beta_overlap)
    
    return S + weight_overlap * overlap


def _exact_objective(params):
    """Exact (nonsmooth) objective for final polish."""
    n = N
    centers = params[:2 * n].reshape(n, 2)
    angles = params[2 * n:]
    
    S = _max_radial_projection(centers, angles, beta=1.0)
    overlap = _total_overlap_penalty(centers, angles, smooth=False, beta=1.0)
    
    if overlap > 1e-6:
        return S + 100.0 * overlap
    return S


def _is_valid(centers, angles, S, tol=1e-4):
    """Check if configuration is valid: no overlap, all inside outer hexagon of side S."""
    # Check overlap
    for i in range(len(centers)):
        v1 = centers[i] + _hex_vertices(np.degrees(angles[i]))
        for j in range(i + 1, len(centers)):
            v2 = centers[j] + _hex_vertices(np.degrees(angles[j]))
            if _sat_overlap(v1, v2) > tol:
                return False
    
    # Check containment in outer hexagon
    k_angles = np.arange(6) * math.pi / 3.0
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    apothem = S * math.sqrt(3) / 2.0
    
    for i in range(len(centers)):
        verts = centers[i] + _hex_vertices(np.degrees(angles[i]))
        for v in verts:
            for nk in normals:
                if abs(v @ nk) > apothem + tol:
                    return False
    return True


def _seed_solution():
    """Generate honeycomb seed configuration."""
    d = math.sqrt(3)
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
    
    # Ring 2: need 4 more to make 11 total
    # Axial coords: max(|q|,|r|,|q+r|) == 2, pick 4
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    # Pick 4 from ring2 (12 available) - choose symmetric ones
    # Sort by angle and pick every 3rd
    ring2.sort(key=lambda p: math.atan2(p[1], p[0]))
    picked = [ring2[i] for i in range(0, 12, 3)]
    centers.extend(picked[:4])
    
    centers = np.array(centers)
    angles = np.zeros(N)
    return centers, angles


def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    best_S = float('inf')
    best_params = None
    
    # Seed configuration
    centers0, angles0 = _seed_solution()
    params0 = np.concatenate([centers0.ravel(), angles0])
    
    # Phase 1: Coarse optimization with smooth surrogate
    # Try multiple initial configurations
    initial_configs = [params0]
    
    # Perturbed seeds
    np.random.seed(42)
    for trial in range(3):
        c = centers0.copy()
        a = angles0.copy()
        c += np.random.normal(0, 0.1, c.shape)
        a += np.random.uniform(-0.1, 0.1, a.shape)
        initial_configs.append(np.concatenate([c.ravel(), a]))
    
    # Also try slightly rotated/translated honeycomb
    for theta in [0.1, -0.1, 0.2]:
        c = centers0.copy()
        rot = np.array([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
        c = c @ rot.T
        a = angles0 + theta
        initial_configs.append(np.concatenate([c.ravel(), a]))
    
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4,
                                    'maxiter': 500, 'maxfev': 1500})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = params0
        best_S = _exact_objective(best_params)
    
    # Phase 2: Refinement with tighter smooth parameters
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           args=(20.0, 40.0, 5.0),
                           options={'xatol': 1e-5, 'fatol': 1e-5,
                                    'maxiter': 500, 'maxfev': 1000})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with exact objective
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(_exact_objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6,
                                    'maxiter': 300, 'maxfev': 800})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Verify and adjust S if needed
    centers = best_params[:2 * N].reshape(N, 2)
    angles_rad = best_params[2 * N:]
    
    # Compute exact S
    S_exact = _max_radial_projection(centers, angles_rad, beta=1.0)
    
    # Add small buffer for numerical safety
    S_final = S_exact + 0.001
    
    # Verify validity
    if not _is_valid(centers, angles_rad, S_final, tol=1e-3):
        # Try to fix: increase S slightly
        S_final += 0.01
        if not _is_valid(centers, angles_rad, S_final, tol=1e-3):
            # Fallback: use larger S
            S_final = S_exact + 0.05
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(angles_rad[i]))])
    
    return inner, (0.0, 0.0), float(S_final), 0.0
