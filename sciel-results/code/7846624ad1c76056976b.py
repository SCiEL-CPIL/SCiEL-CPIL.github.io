"""Pack 11 unit hexagons into a regular hexagon, minimizing outer side length."""
import time
import numpy as np
from scipy.optimize import linprog

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, rot):
    """Vertices of regular hexagon with side s, center (cx,cy), rotation rot (radians)."""
    angles = rot + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _unit_hex_verts(angle_deg):
    """Vertices of unit hexagon at origin, rotated by angle_deg."""
    rot = np.radians(angle_deg)
    a = rot + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(a), np.sin(a)])

def _min_side_lp(inner_pts, n_angles=12):
    """
    For a fixed set of inner hexagon centers (unit hexagons, angle 0),
    find minimal outer side s over outer center (cx,cy) and outer rotation theta.
    LP: min s s.t. for each inner hex vertex v_i and each outer face k:
        n_k . (v_i + t) <= s + cx_offset ... 
    Actually: outer hexagon with center (cx,cy), side s, rotation theta.
    Inequality form: for each of 6 faces, n_k . (p - c) <= s * apothem_factor... 
    For regular hexagon side s, apothem = s*sqrt(3)/2.
    Face k normal n_k (unit), point p inside iff n_k.(p-c) <= apothem = s*sqrt(3)/2.
    So: s*sqrt(3)/2 - n_k.(p-c) >= 0  =>  n_k.c - n_k.p + (sqrt(3)/2)*s >= 0.
    Variables: cx, cy, s. But theta is discrete (we try several).
    """
    sqrt3_2 = np.sqrt(3) / 2.0
    verts = []
    for (x, y) in inner_pts:
        v = _unit_hex_verts(0.0)
        verts.append(v + np.array([x, y]))
    verts = np.vstack(verts)  # (6N, 2)
    
    best_s = np.inf
    for theta in np.linspace(0, np.pi/3, n_angles):
        # Outer face normals
        angles = theta + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
        
        # Constraints: for each vertex v and each face k:
        # normals[k].(v - c) <= s*sqrt3_2
        # => -normals[k].v + normals[k].c <= s*sqrt3_2
        # => normals[k].cx + normals[k].cy - sqrt3_2*s - normals[k].v >= 0
        # In linprog form (A_ub x <= b_ub):
        # -normals[k].cx - normals[k].cy + sqrt3_2*s + normals[k].v <= 0
        # Variables x = [cx, cy, s]
        m = len(verts)
        A = np.zeros((6 * m, 3))
        b = np.zeros(6 * m)
        idx = 0
        for k in range(6):
            nk = normals[k]
            for i in range(m):
                A[idx, 0] = -nk[0]
                A[idx, 1] = -nk[1]
                A[idx, 2] = sqrt3_2
                b[idx] = nk[0] * verts[i, 0] + nk[1] * verts[i, 1]
                idx += 1
        
        c_obj = np.array([0.0, 0.0, 1.0])  # minimize s
        bounds = [(None, None), (None, None), (0.0, None)]
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            if res.x[2] < best_s:
                best_s = res.x[2]
    return best_s

def _generate_candidates():
    """Generate candidate configurations of 11 unit hexagons on honeycomb lattice."""
    # Honeycomb lattice basis vectors (side=1)
    # Using axial coordinates: neighbors at 60-degree spacing
    a1 = np.array([1.5, np.sqrt(3)/2])
    a2 = np.array([0.0, np.sqrt(3)])
    
    # Generate a pool of lattice points
    candidates_pts = []
    # We want compact clusters. Generate points in a region.
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            candidates_pts.append(p)
    candidates_pts = np.array(candidates_pts)
    
    # Generate subsets of 11 that are "connected" (compact)
    # For simplicity, generate all subsets of size 11 from a smaller pool,
    # but that's C(81,11) which is huge. Instead, use a greedy/beam approach.
    
    # Better: generate specific compact shapes.
    # A 11-hexagon cluster: e.g., a 4x3 block minus 1, or a "flower" shape.
    
    # Let's generate by starting from a center and adding neighbors.
    # Use a simple approach: generate random connected subsets.
    
    rng = np.random.default_rng(42)
    seen = set()
    results = []
    
    # Precompute adjacency
    # Two points are adjacent if distance = sqrt(3) (for side=1 hexagons, centers are sqrt(3) apart)
    # Actually for unit hexagons in honeycomb, center distance = sqrt(3).
    dist_thresh = np.sqrt(3) + 1e-9
    
    # Generate many random connected clusters
    for trial in range(2000):
        # Start from a random point
        start = rng.integers(len(candidates_pts))
        cluster = {start}
        # Grow to size 11
        for _ in range(10):
            if len(cluster) >= 11:
                break
            # Find all neighbors of cluster points not in cluster
            neighbors = set()
            for idx in cluster:
                p = candidates_pts[idx]
                for jdx in range(len(candidates_pts)):
                    if jdx in cluster:
                        continue
                    q = candidates_pts[jdx]
                    if np.linalg.norm(p - q) <= dist_thresh:
                        neighbors.add(jdx)
            if not neighbors:
                break
            new_pt = rng.choice(list(neighbors))
            cluster.add(new_pt)
        
        if len(cluster) == 11:
            # Canonical form: translate so min point is at origin, sort
            pts = np.array([candidates_pts[i] for i in sorted(cluster)])
            # Center
            center = pts.mean(axis=0)
            pts_centered = pts - center
            # Round for hashing
            key = tuple(np.round(pts_centered, 4).flatten().tolist())
            if key not in seen:
                seen.add(key)
                results.append(pts_centered.tolist())
    
    return results

def construct():
    t0 = time.time()
    
    # Generate candidate configurations
    candidates = _generate_candidates()
    
    if not candidates:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_s = np.inf
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for i, cand in enumerate(candidates):
        if time.time() - t0 > DEADLINE:
            break
        # cand is list of (x,y) centered at origin
        s = _min_side_lp(cand)
        if s < best_s:
            best_s = s
            best_inner = cand
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Convert to required format: (x, y, angle_degrees)
    inner_fmt = [[p[0], p[1], 0.0] for p in best_inner]
    
    return inner_fmt, best_center, best_s, best_angle
