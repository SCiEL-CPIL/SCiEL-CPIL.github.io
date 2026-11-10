# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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
    deadline = t0 + 590.0  # Safety margin

    # Define a few candidate cluster topologies (relative positions of 11 hexagons)
    # All hexagons have the same orientation (angle 0) in the local frame.
    # We will optimize the global rotation of this cluster and the outer hexagon.
    
    # Topology 1: "Flower" (7) + 4 extensions
    # Axial coords (q, r) -> Cartesian (1.5q, sqrt(3)(r + q/2))
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
        
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # Let's pick a symmetric set of 4: (2,0), (-2,0), (0,2), (0,-2)
    # Wait, (0,2) is distance 2 from (0,1) and (1,1)? No.
    # Distance between (0,1) and (0,2) is 1 axial unit = sqrt(3) cartesian? No.
    # Axial distance formula: max(|dq|, |dr|, |dq+dr|)
    # (0,1) to (0,2): dq=0, dr=1, sum=1. Dist 1. Touching.
    # (1,0) to (0,2): dq=-1, dr=2, sum=1. Dist 2. Not touching.
    
    # Let's use a known compact packing: "Hexagonal number" H_3 is 19. H_2 is 7.
    # We need 11.
    # A common shape is a hexagon of side 2 (19 cells) minus 8 corners?
    # Or just take the 7 flower and add 4 in the "gaps" of the second ring?
    # The second ring has 12 positions.
    # If we add 4 at (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2), we have 6.
    # Let's try adding 4 at (2,0), (-2,0), (1,1), (-1,-1)?
    # (1,1) is adjacent to (0,1) and (1,0).
    # (1,1) to (1,0): dq=0, dr=1. Dist 1. Touching.
    # (1,1) to (0,1): dq=1, dr=0. Dist 1. Touching.
    # So (1,1) fits in the "armpit" of the first ring.
    # Similarly (-1,-1) fits in the opposite armpit.
    # And (0,2) fits in the top armpit?
    # (0,2) to (0,1): Dist 1.
    # (0,2) to (1,1): dq=-1, dr=1. Dist 1.
    # (0,2) to (-1,1): dq=1, dr=1. Dist 1.
    # So (0,2) is surrounded by (0,1), (1,1), (-1,1).
    # If we include (1,1) and (-1,1), then (0,2) is well supported.
    
    # Let's define Cluster A:
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): (1,1), (-1,1), (0,2), (-1,-1), (1,-1) is already in R1.
    # Let's pick: (1,1), (-1,1), (0,2), (0,-2)?
    # (0,-2) to (0,-1): Dist 1.
    # (0,-2) to (1,-1): Dist 1.
    # (0,-2) to (-1,-1): Dist 1.
    # So Cluster A:
    coords_A = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    
    # Cluster B: More elongated?
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,-1), (1,-1), (2,-1)
    # Total 11.
    coords_B = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,-1), (1,-1), (2,-1)
    ]
    
    # Cluster C: Random perturbation of A
    coords_C = coords_A + np.random.default_rng(42).normal(0, 0.1, size=(11, 2)) # Just for variety, but we'll use exact A and B
    
    clusters = [
        np.array([axial_to_cartesian(q, r) for q, r in coords_A]),
        np.array([axial_to_cartesian(q, r) for q, r in coords_B])
    ]
    
    # Center each cluster
    for i in range(len(clusters)):
        clusters[i] -= clusters[i].mean(axis=0)
        
    best_val = -np.inf
    best_result = None
    
    for c_idx, cluster in enumerate(clusters):
        if time.time() > deadline - 5.0:
            break
            
        # Variables: theta (cluster rotation), phi (outer hex rotation)
        # The outer hexagon is centered at the origin (0,0) for simplicity, 
        # but we can also optimize its center? 
        # The task allows outer (x, y, angle). 
        # However, the optimal enclosing hexagon for a centered cluster is usually centered.
        # Let's optimize (theta, phi, ox, oy) to be safe, but start with ox=oy=0.
        
        def objective(params):
            theta, phi, ox, oy = params
            c = np.cos(theta)
            s = np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rot_cluster = cluster @ R.T
            
            # Outer hexagon vertices
            # Side length s_side is determined by the support function.
            # We want to minimize s_side.
            # s_side = 2/sqrt(3) * max_k (max_i (p_i . n_k) - min_i (p_i . n_k)) / 2? No.
            # For a regular hexagon centered at (ox, oy) with rotation phi:
            # Normals n_k(phi) = (cos(phi + 60k), sin(phi + 60k))
            # The hexagon is defined by: n_k . (p - center) <= apothem
            # apothem = s_side * sqrt(3) / 2
            # So s_side = 2/sqrt(3) * max_k (max_i (n_k . (p_i - center)))
            # To minimize s_side, we should center the cluster optimally.
            # The optimal center for a given orientation is the midpoint of the min and max projections?
            # Actually, for a fixed orientation, the minimal enclosing hexagon has its center at the "center" of the cluster in some sense.
            # Let's just compute the required side length for a hexagon centered at (ox, oy).
            # s_side(ox, oy, phi) = 2/sqrt(3) * max_k ( max_i (n_k . (p_i - [ox, oy])) )
            # Note: We assume the hexagon is symmetric, so we only need to check the max projection.
            # Wait, if the cluster is not centered, the max projection from the center determines the size.
            
            phi_rad = np.deg2rad(phi)
            normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                                       np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
            
            # Projections of rotated cluster points onto normals
            # p_i is (x, y). We want n_k . (p_i - center)
            # = n_k . p_i - n_k . center
            proj = rot_cluster @ normals.T  # (11, 6)
            center_proj = np.array([ox, oy]) @ normals.T  # (6,)
            
            # For each normal k, the max distance from center is max_i (proj[i,k] - center_proj[k])
            # The side length is 2/sqrt(3) * max_k ( max_i (proj[i,k] - center_proj[k]) )
            # BUT, this assumes the hexagon is "tight" on one side. 
            # A regular hexagon is defined by 6 half-planes.
            # The condition is: for all k, max_i (n_k . (p_i - center)) <= apothem
            # AND min_i (n_k . (p_i - center)) >= -apothem
            # So 2 * apothem >= max_k ( max_i (n_k . p_i - n_k . center) - min_i (n_k . p_i - n_k . center) )
            # apothem >= 0.5 * max_k ( range_k )
            # s_side = 2/sqrt(3) * apothem = 1/sqrt(3) * max_k ( range_k )
            
            ranges = proj.max(axis=0) - proj.min(axis=0)
            max_range = ranges.max()
            s_side = max_range / np.sqrt(3.0)
            
            return s_side
            
        # Initial guess
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Try a few random initial rotations to escape local minima
        best_local_val = -np.inf
        best_local_params = None
        
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
                
            if trial == 0:
                x_start = x0
            else:
                x_start = np.array([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 60), 0, 0])
                
            res = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < best_local_val:
                best_local_val = res.fun
                best_local_params = res.x
                
        if best_local_params is None:
            continue
            
        theta_opt, phi_opt, ox_opt, oy_opt = best_local_params
        s_opt = best_local_val
        
        # Reconstruct inner and outer
        c = np.cos(theta_opt)
        s = np.sin(theta_opt)
        R = np.array([[c, -s], [s, c]])
        rot_cluster = cluster @ R.T
        
        # Inner hexagons: centers at rot_cluster, angle 0 (relative to global frame? No, angle is absolute)
        # The cluster rotation rotates the hexagons too.
        # So inner angle should be theta_opt?
        # The hexagon vertices are defined by (cx + s*cos(a + 60i), ...).
        # If we rotate the center by theta, and keep the hexagon orientation fixed at 0, 
        # the hexagon is NOT rotated. 
        # But in our cluster definition, the hexagons are part of the rigid body.
        # So if the cluster rotates by theta, the hexagons rotate by theta.
        # So inner angle = theta_opt (in degrees).
        
        inner = np.column_stack((rot_cluster[:, 0], rot_cluster[:, 1], np.full(11, np.rad2deg(theta_opt))))
        
        # Outer hexagon
        outer = np.array([ox_opt, oy_opt, phi_opt])
        
        # Check validity
        if _all_valid(inner, outer, s_opt):
            val = 1.0 / s_opt
            if val > best_val:
                best_val = val
                best_result = (inner, outer, s_opt)
                
    if best_result is None:
        # Fallback
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
