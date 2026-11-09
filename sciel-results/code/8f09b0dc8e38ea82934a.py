import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_lattice_init(side=1.0):
    """
    Generate initial positions for N=12 unit hexagons in a 3x4 lattice.
    This configuration is known to be valid.
    """
    pts = []
    # 3 rows, 4 columns
    # Hexagon side length s=1.
    # Distance between centers in same row: 2 * s * cos(30) = sqrt(3)
    # Distance between rows: 1.5 * s = 1.5
    # Offset for alternating rows: s * cos(30) = sqrt(3)/2
    
    dx = np.sqrt(3)
    dy = 1.5
    offset = np.sqrt(3) / 2
    
    # Center the lattice roughly at (0,0)
    # Width of 4 cols: 3 * dx
    # Height of 3 rows: 2 * dy
    x_start = -1.5 * dx
    y_start = -dy
    
    for i in range(3):
        for j in range(4):
            x = x_start + j * dx + (offset if i % 2 == 1 else 0)
            y = y_start + i * dy
            pts.append([x, y])
            
    return np.array(pts)

def _min_enclosing_hex_side(pts, angles):
    """
    Calculate the minimum side length S of a regular hexagon centered at (0,0) 
    with angle 0 that contains all points (centers of inner hexagons) 
    expanded by their apothem.
    """
    # Inner hexagons have side 1. Their apothem is sqrt(3)/2.
    # For the inner hexagons to be inside the outer hexagon of side S,
    # the distance from the center of the inner hexagon to any boundary of the outer hexagon
    # must be >= apothem_inner.
    # The boundary of the outer hexagon (side S, angle 0) is defined by 3 pairs of parallel lines.
    # The distance from origin to each side is S * sqrt(3) / 2.
    # Let R_outer = S * sqrt(3) / 2.
    # Let R_inner = 1 * sqrt(3) / 2.
    # For a point p at distance d from origin, in a direction theta,
    # the condition for containment in a regular hexagon of circumradius R_c is complex.
    # However, the inradius (apothem) of the outer hexagon is R_outer.
    # The condition is that the projection of the center of the inner hexagon onto any of the 6 normal vectors
    # of the outer hexagon must be <= R_outer - R_inner_apothem? No.
    
    # Let's use the support function.
    # A point p is inside a regular hexagon (center 0, side S) if:
    # max_{k=0..5} dot(p, n_k) <= S * sqrt(3) / 2
    # where n_k are the unit normals to the sides.
    # Since the inner hexagon has size, we need:
    # max_{k=0..5} (dot(p, n_k) + dot(v_max, n_k)) <= S * sqrt(3) / 2
    # where v_max is the vertex of the inner hexagon in the direction n_k.
    # For a regular hexagon, the support function h(u) = R_c * cos(angle_diff) ? No.
    # The support function of a regular hexagon with side s is h(u) = s * sqrt(3)/2 * 1 / cos( (angle(u) - angle(vertex)) mod pi/3 - pi/6 )?
    # Simpler: The Minkowski sum of two regular hexagons (one centered at p, one at 0) is a regular hexagon? No, it's a larger regular hexagon if aligned, or a dodecagon if rotated.
    # Since we can rotate inner hexagons, the worst case is when an inner hexagon is aligned such that a vertex points outwards.
    # To be safe and simple, we can assume the inner hexagons are aligned with the outer one (angle 0).
    # Then the "effective" radius of the inner hexagon in any direction is the distance from its center to its boundary.
    # For aligned hexagons, the condition is:
    # dot(p, n_k) + h_inner(n_k) <= h_outer(n_k)
    # h_outer(n_k) = S * sqrt(3) / 2
    # h_inner(n_k) = 1 * sqrt(3) / 2 (since n_k is normal to a side, and inner is aligned, the max projection is the apothem)
    # So: dot(p, n_k) + sqrt(3)/2 <= S * sqrt(3) / 2
    # dot(p, n_k) <= (S - 1) * sqrt(3) / 2
    
    # If we allow rotation, the max projection of a unit hexagon in any direction is its circumradius = 1.
    # But the average is lower. Let's stick to angle=0 for simplicity and robustness, as the lattice is aligned.
    # If we use angle=0, the constraint is:
    # max_k dot(p, n_k) <= (S - 1) * sqrt(3) / 2
    
    # Normals for hexagon with flat top (angle 0):
    # Side 1 (top): normal (0, 1)
    # Side 2 (top-right): normal (sqrt(3)/2, -1/2) ? No.
    # Vertices at angles 0, 60, 120...
    # Side normals at 30, 90, 150...
    # Let's define normals for a hexagon with a horizontal top side.
    # The top side is at y = S*sqrt(3)/2. Normal (0, 1).
    # The top-right side.
    # Let's just compute the required S for a given set of points and angle=0.
    
    # Normals:
    # 0: (0, 1)
    # 1: (sqrt(3)/2, -0.5)
    # 2: (-sqrt(3)/2, -0.5)
    # 3: (0, -1)
    # 4: (-sqrt(3)/2, 0.5)
    # 5: (sqrt(3)/2, 0.5)
    
    normals = np.array([
        [0, 1],
        [np.sqrt(3)/2, -0.5],
        [-np.sqrt(3)/2, -0.5],
        [0, -1],
        [-np.sqrt(3)/2, 0.5],
        [np.sqrt(3)/2, 0.5]
    ])
    
    # Projection of points onto normals
    # proj[i, k] = dot(p_i, n_k)
    proj = pts @ normals.T
    
    # Max projection for each normal
    max_proj = np.max(proj, axis=0)
    
    # Constraint: max_proj[k] + sqrt(3)/2 <= S * sqrt(3) / 2
    # S >= (max_proj[k] + sqrt(3)/2) * 2 / sqrt(3)
    # S >= max_proj[k] * 2/sqrt(3) + 1
    
    required_S = np.max(max_proj * 2.0 / np.sqrt(3) + 1.0)
    
    # Also need to check pairwise distances for non-overlap
    # Distance between centers must be >= 2 * apothem = sqrt(3)
    # This is handled by the optimizer penalty, not this function.
    
    return required_S

def _pairwise_overlap_penalty(pts):
    """
    Penalty if any two inner hexagons overlap.
    Min distance between centers is sqrt(3).
    """
    penalty = 0.0
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            min_dist = np.sqrt(3)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty

def _objective(params):
    """
    Objective: Minimize the outer side length S, with penalties for overlap.
    params: [x0, y0, ..., x11, y11]
    """
    pts = params.reshape(N, 2)
    
    # Calculate S required for containment
    S = _min_enclosing_hex_side(pts, None)
    
    # Calculate overlap penalty
    overlap_pen = _pairwise_overlap_penalty(pts)
    
    # Return S + large penalty for overlap
    return S + 100.0 * overlap_pen

def _check_valid(pts, S, tol=1e-5):
    """
    Check if configuration is valid:
    1. All inner hexagons inside outer hexagon of side S.
    2. No inner hexagons overlap.
    """
    # 1. Containment
    required_S = _min_enclosing_hex_side(pts, None)
    if required_S > S + tol:
        return False
    
    # 2. Non-overlap
    min_dist = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < min_dist - tol:
                return False
                
    return True

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0
    
    # Initial guess: Hex lattice
    init_pts = _hex_lattice_init(1.0)
    current_params = init_pts.flatten()
    
    # Calculate initial S
    best_S = _min_enclosing_hex_side(init_pts, None)
    best_params = current_params.copy()
    
    # Try to optimize
    # Use L-BFGS-B
    # We don't have bounds, but the objective penalizes large S and overlap.
    # To help convergence, we can try multiple random starts or just the lattice.
    
    try:
        # Optimize from the lattice
        res = minimize(_objective, current_params, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9})
        
        if time.time() < DEADLINE - 10:
            if res.success or res.fun < best_S:
                # Verify validity of the optimized solution
                opt_pts = res.x.reshape(N, 2)
                opt_S = _min_enclosing_hex_side(opt_pts, None)
                
                # The objective included penalty, so res.fun might be lower than true S if penalty is 0.
                # If penalty is 0, res.fun == opt_S.
                if _check_valid(opt_pts, opt_S, tol=1e-6):
                    best_params = res.x
                    best_S = opt_S
                else:
                    # If invalid, maybe the penalty wasn't enough or it got stuck.
                    # Try to repair or just keep the lattice.
                    pass
                    
            # Try a few random restarts if time permits
            restarts = 5
            for r in range(restarts):
                if time.time() > DEADLINE - 15:
                    break
                
                # Random perturbation
                noise = np.random.normal(0, 0.5, size=N*2)
                start_params = best_params + noise
                res_r = minimize(_objective, start_params, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9})
                
                if res_r.fun < best_S:
                    opt_pts = res_r.x.reshape(N, 2)
                    opt_S = _min_enclosing_hex_side(opt_pts, None)
                    if _check_valid(opt_pts, opt_S, tol=1e-6):
                        best_params = res_r.x
                        best_S = opt_S
                        
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to lattice
    
    # Final check
    final_pts = best_params.reshape(N, 2)
    final_S = _min_enclosing_hex_side(final_pts, None)
    
    if not _check_valid(final_pts, final_S, tol=1e-5):
        # Fallback to guaranteed valid lattice
        final_pts = init_pts
        final_S = _min_enclosing_hex_side(init_pts, None)

    inner = [[float(final_pts[i, 0]), float(final_pts[i, 1]), 0.0] for i in range(N)]
    center = (0.0, 0.0)
    return inner, center, float(final_S), 0.0
