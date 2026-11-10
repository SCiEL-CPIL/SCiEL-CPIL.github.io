# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _outer_side_for_points(pts):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0

    # The "flower" arrangement: 1 center, 6 around it, 4 more.
    # This is a known dense packing for hexagons.
    # Center at (0,0).
    # Neighbors at distance 2 (touching).
    # Hexagon vertices distance from center is 1.
    # So centers must be at least 2 apart.
    
    # Standard hex grid centers:
    # (0,0)
    # (2,0), (-2,0)
    # (1, sqrt(3)), (-1, sqrt(3))
    # (1, -sqrt(3)), (-1, -sqrt(3))
    # This is 7 hexagons.
    
    # We need 4 more.
    # Place them in the "gaps" of the flower.
    # The gaps are at angles 30, 90, 150, 210, 270, 330? No.
    # The flower has 6-fold symmetry.
    # Adding 4 breaks symmetry.
    
    # Let's try a specific configuration that is known to be compact.
    # A 3x3 grid has 9. Add 2.
    # Or a 2x2x2?
    
    # Let's use a deterministic search over a few candidate topologies.
    
    # Candidate 1: The "3-4-3" or similar?
    # Actually, let's just generate a few good starting positions and refine with a 1D rotation search.
    
    # The key insight: If we fix the relative positions of the 11 hexagons (the cluster),
    # the optimal outer hexagon is found by rotating the cluster.
    # The side length s(theta) is a function of the rotation theta.
    # We want to minimize s(theta).
    
    # Let's define a few cluster configurations.
    
    def _make_cluster(config_type):
        if config_type == 0:
            # Standard 7-flower + 4 satellites
            # Center (0,0)
            # Ring 1: 6 hexagons
            # Ring 2: 4 hexagons placed in the "armpits" of the ring 1?
            # Actually, the "armpits" are where 3 hexagons meet.
            # In a hex grid, 3 hexagons meet at a vertex.
            # The "gap" is not a place for a hexagon.
            
            # Let's try a 3x3 grid (9 hexagons) + 2.
            # 3x3 grid:
            # (-2, 0), (0, 0), (2, 0)
            # (-1, 1.732), (1, 1.732)
            # (-3, 1.732), (-1, 1.732), (1, 1.732), (3, 1.732) -- wait, this is not a grid.
            
            # Hex grid coordinates (axial):
            # (0,0)
            # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            # Convert to cartesian:
            # x = 1.5 * q
            # y = sqrt(3) * (r + q/2)
            
            def axial_to_cartesian(q, r):
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2.0)
                return x, y
            
            # 7 flower:
            # (0,0)
            # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            
            # Add 4 more.
            # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
            # Let's pick 4 that are symmetric-ish.
            # (2,0), (0,2), (-2,0), (0,-2)
            
            centers = [
                axial_to_cartesian(0, 0),
                axial_to_cartesian(1, 0),
                axial_to_cartesian(0, 1),
                axial_to_cartesian(-1, 1),
                axial_to_cartesian(-1, 0),
                axial_to_cartesian(0, -1),
                axial_to_cartesian(1, -1),
                axial_to_cartesian(2, 0),
                axial_to_cartesian(0, 2),
                axial_to_cartesian(-2, 0),
                axial_to_cartesian(0, -2),
            ]
            return np.array(centers)
        
        elif config_type == 1:
            # Try a more compact cluster
            # Maybe a 2x5 rectangle?
            # (0,0), (2,0), (4,0)
            # (1, 1.732), (3, 1.732)
            # ...
            # Let's try a "snake" or "zigzag"
            centers = [
                (0, 0),
                (2, 0),
                (1, 1.732),
                (3, 1.732),
                (2, 3.464),
                (0, 3.464),
                (-1, 1.732),
                (-2, 3.464),
                (-3, 1.732),
                (-1, -1.732),
                (1, -1.732),
            ]
            return np.array(centers)
        
        else:
            # Random perturbation of config 0
            c0 = _make_cluster(0)
            rng = np.random.default_rng(config_type)
            return c0 + rng.normal(0, 0.1, size=c0.shape)

    best_val = -np.inf
    best_result = None

    for config_type in range(5):
        if time.time() > deadline - 10.0:
            break
            
        centers = _make_cluster(config_type)
        
        # Center the cluster
        centers -= centers.mean(axis=0)
        
        # The inner hexagons are at these centers, with angle 0 (flat top or pointy top doesn't matter for separation if they are far enough, but for containment it does. However, we assume the cluster is rigid).
        # Actually, the angle of the inner hexagons matters for separation.
        # In a hex grid, all hexagons have the same orientation.
        
        inner = np.column_stack((centers[:, 0], centers[:, 1], np.zeros(11)))
        
        # Now, find the best outer hexagon.
        # We can rotate the cluster (and thus the inner hexagons) and find the minimum enclosing hexagon.
        # Or, fix the cluster and rotate the outer hexagon.
        # These are equivalent.
        
        # Let's optimize the rotation of the cluster.
        # theta is the rotation angle of the cluster.
        
        def _objective(theta):
            # Rotate centers
            c = np.cos(theta)
            s = np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rot_centers = centers @ R.T
            
            # Update inner
            inner_rot = np.column_stack((rot_centers[:, 0], rot_centers[:, 1], np.zeros(11)))
            
            # Find min enclosing hexagon for these points
            # We can use the _outer_side_for_points logic but it's slow.
            # Let's do a quick check of the 6 normals.
            # The optimal outer hexagon will have its normals aligned with the directions that maximize the support function.
            
            # For a fixed cluster, the min enclosing hexagon side length s(a) is:
            # s(a) = 2/sqrt(3) * max_k (max_i (p_i . n_k(a)))
            # where n_k(a) are the 6 normals of the outer hexagon at angle a.
            
            # We want to minimize s(a) over a.
            # This is a 1D optimization.
            
            # Let's sample a few angles for a to find the min.
            # Since the cluster is fixed, we can just scan a.
            min_s = np.inf
            best_a = 0
            
            # We can restrict a to 0..60 due to symmetry
            for a_deg in np.linspace(0, 60, 100):
                a = np.deg2rad(a_deg)
                th = a + np.deg2rad(60.0) * np.arange(6)
                normals = np.column_stack((np.cos(th), np.sin(th)))
                proj = rot_centers @ normals.T
                maxproj = proj.max(axis=0)
                apothem = maxproj.max()
                s_val = 2.0 * apothem / np.sqrt(3.0)
                if s_val < min_s:
                    min_s = s_val
                    best_a = a_deg
            
            return min_s

        # Optimize theta
        res = minimize_scalar(_objective, bounds=(0, 2*np.pi), method='bounded', options={'xatol': 1e-6})
        
        theta_opt = res.x
        s_opt = res.fun
        
        # Reconstruct the solution
        c = np.cos(theta_opt)
        s = np.sin(theta_opt)
        R = np.array([[c, -s], [s, c]])
        rot_centers = centers @ R.T
        
        inner = np.column_stack((rot_centers[:, 0], rot_centers[:, 1], np.zeros(11)))
        
        # Find the best outer hexagon angle
        best_s = np.inf
        best_a = 0
        for a_deg in np.linspace(0, 60, 361):
            a = np.deg2rad(a_deg)
            th = a + np.deg2rad(60.0) * np.arange(6)
            normals = np.column_stack((np.cos(th), np.sin(th)))
            proj = rot_centers @ normals.T
            maxproj = proj.max(axis=0)
            apothem = maxproj.max()
            s_val = 2.0 * apothem / np.sqrt(3.0)
            if s_val < best_s:
                best_s = s_val
                best_a = a_deg
        
        outer = np.array([0.0, 0.0, best_a])
        
        # Check validity
        if _all_valid(inner, outer, best_s):
            val = 1.0 / best_s
            if val > best_val:
                best_val = val
                best_result = (inner, outer, best_s)

    if best_result is None:
        # Fallback to the seed program's result if possible, or a valid loose packing
        # Let's just return the first valid config we found, or a known valid one.
        # The seed program returned a valid result.
        # We can replicate the seed's fallback.
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    return best_result

# EVOLVE-BLOCK-END
