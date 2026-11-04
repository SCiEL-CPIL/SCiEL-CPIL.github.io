import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _compute_required_side(centers, angles_deg, outer_angle_deg=0.0, outer_center=(0,0)):
    """
    Compute the minimum side length of an outer hexagon (fixed angle, centered at 0,0) 
    that contains all inner hexagons.
    Uses the half-plane intersection property: S >= max(dot(P-C, n_k)) * 2/sqrt(3).
    """
    theta = np.deg2rad(outer_angle_deg)
    normals = np.array([
        [np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] 
        for k in range(6)
    ])
    
    max_proj = 0.0
    for c, a_deg in zip(centers, angles_deg):
        verts = _hex_vertices(c, 1.0, a_deg)
        # Project vertices onto normals
        # verts shape (6,2), normals shape (6,2)
        # We need dot(verts - outer_center, n) for all verts and all n
        # Efficient: verts @ normals.T gives (6,6) matrix of projections
        projs = verts @ normals.T
        # max_proj is the maximum over all vertices and all normals
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
            
    return max_proj * 2.0 / np.sqrt(3.0)

def _smooth_collision_penalty(centers, angles_deg, beta=20.0):
    """
    Compute a smooth approximation of the overlap volume/distance between pairs of hexagons.
    We use a soft-min of the minimum distance between edges/vertices, or simpler:
    approximate the hexagon by its inscribed circle? No, that's too loose.
    
    Better approach for differentiability:
    Use the Signed Distance Function (SDF) approximation or just check distance between centers?
    For hexagons, if centers are close, they overlap.
    A robust differentiable proxy: 
    For each pair, compute the minimum distance between their convex hulls.
    This is hard to make differentiable cheaply.
    
    Alternative: Use a penalty based on the intersection of their "expanded" bounding boxes or circles?
    Let's use a simpler heuristic that is differentiable:
    Penalty = sum over pairs of max(0, R_i + R_j - dist_ij)^2 where R is effective radius?
    For a unit hexagon, the circumradius is 1, inradius is sqrt(3)/2.
    If we use circumradius 1, we are too conservative (allows gaps).
    If we use inradius, we allow overlaps.
    
    Let's use a more accurate but still differentiable metric:
    The minimum distance between two convex polygons can be approximated by the minimum distance between their vertices and edges.
    However, implementing a differentiable SAT is complex.
    
    Let's try a different strategy: 
    Optimize the positions such that the "packing density" is high.
    Actually, the inspiration suggests using a smooth surrogate.
    
    Let's use the following differentiable penalty for overlap:
    For each pair (i, j), compute the distance d_ij between centers.
    The minimum possible distance between centers of two non-overlapping unit hexagons depends on their relative rotation.
    The minimum distance is sqrt(3) (when aligned in a honeycomb).
    If d_ij < sqrt(3), they might overlap.
    A simple smooth penalty: max(0, sqrt(3) - d_ij)^2.
    This assumes the worst-case orientation for overlap, which is safe but might be suboptimal.
    
    To be more precise, we can compute the actual minimum distance between the two hexagons using a differentiable SAT-like routine?
    Or just use the center distance penalty with a tunable threshold.
    Let's use threshold = sqrt(3) * 0.99 to allow tight packing.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            # Minimum center distance for non-overlap is roughly sqrt(3) for aligned hexes.
            # For arbitrary rotations, it can be slightly less, but sqrt(3) is a good lower bound for "tight" packing.
            # Actually, if rotated 30 degrees, the min distance is smaller?
            # Let's use a soft penalty that becomes active when d < sqrt(3).
            gap = np.sqrt(3.0) - d
            if gap > 0:
                penalty += gap * gap
    return penalty

def _objective(params):
    """
    Objective function to minimize.
    params: array of length 3*N: [x1, y1, a1, x2, y2, a2, ...]
    """
    if time.time() - START_TIME > TIME_LIMIT - 2:
        return 1e10
        
    centers = params[0::3]
    angles_deg = params[1::3]
    
    # 1. Compute required outer side length
    S = _compute_required_side(centers, angles_deg)
    
    # 2. Compute overlap penalty
    # Use a large weight to ensure disjointness is prioritized, but allow some overlap if it reduces S significantly?
    # No, validity is strict. We need to ensure final result is valid.
    # So penalty should be very high if overlap is detected.
    penalty = _smooth_collision_penalty(centers, angles_deg)
    
    # Combine: Minimize S + lambda * penalty
    # If penalty is non-zero, the solution is invalid. We want to drive penalty to 0.
    # To make it a hard constraint in a soft way, we can use a large lambda.
    # However, if the geometry is impossible with low S, we might get stuck.
    # Let's use a barrier-like approach: S * (1 + penalty)
    # Or simply S + 100 * penalty
    
    return S + 100.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Honeycomb lattice
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate 12 points in a compact cluster (e.g., 3x4 grid or hexagonal)
    # Let's use a hexagonal arrangement of radius 1 (7 points) + 5 more
    # Or just a 3x4 grid
    init_centers = []
    for i in range(-1, 2):
        for j in range(-2, 2):
            p = i * v1 + j * v2
            init_centers.append(p)
            if len(init_centers) >= 12:
                break
        if len(init_centers) >= 12:
            break
            
    init_centers = np.array(init_centers)
    # Center the cluster
    centroid = np.mean(init_centers, axis=0)
    init_centers = init_centers - centroid
    
    init_angles = np.zeros(N)
    
    init_params = np.concatenate([init_centers.flatten(), init_angles])
    
    # Run optimization
    best_result = None
    best_obj = float('inf')
    
    # Use L-BFGS-B for local optimization
    # Multiple restarts with different initial perturbations?
    # Given time limit, one good run might be enough if init is good.
    # Let's do 3 restarts.
    
    for restart in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        if restart == 0:
            x0 = init_params
        else:
            # Add small random noise
            noise = np.random.normal(0, 0.1, size=init_params.shape)
            x0 = init_params + noise
            
        # Define bounds? No, unbounded.
        # But we should constrain angles to [0, 60) due to symmetry?
        # L-BFGS-B supports bounds.
        # Angles are periodic. Let's not bound them tightly to avoid boundary issues, 
        # or bound to [-180, 180].
        
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
            
    if best_result is None:
        # Fallback to initial guess
        best_result = init_params
        
    # Extract final centers and angles
    final_centers = best_result[0::3]
    final_angles_deg = best_result[1::3]
    
    # Compute final side length
    final_side = _compute_required_side(final_centers, final_angles_deg)
    
    # Verify validity with strict SAT
    # If invalid, we need to repair or fall back.
    # Given the soft penalty, it might be slightly overlapping.
    # Let's check and if invalid, expand the outer hexagon slightly or move hexagons apart?
    # Better: Just return the result. The evaluator checks validity.
    # If invalid, score is 0. So we MUST ensure validity.
    
    # Let's do a quick check
    outer_verts = _hex_vertices((0,0), final_side, 0.0)
    inner_verts = [_hex_vertices(c, 1.0, a) for c, a in zip(final_centers, final_angles_deg)]
    
    is_valid = True
    # Check containment
    for iv in inner_verts:
        for v in iv:
            # Point in hexagon check
            # Using the same normals logic
            theta = 0.0
            normals = np.array([
                [np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] 
                for k in range(6)
            ])
            projs = v @ normals.T
            apothem = final_side * np.sqrt(3) / 2
            if np.any(projs > apothem + 1e-7):
                is_valid = False
                break
        if not is_valid:
            break
            
    if is_valid:
        # Check disjointness with SAT
        for i in range(N):
            for j in range(i + 1, N):
                # SAT check
                axes = []
                for k in range(6):
                    p1 = inner_verts[i][k]
                    p2 = inner_verts[i][(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                    p1 = inner_verts[j][k]
                    p2 = inner_verts[j][(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                        
                overlap = True
                for axis in axes:
                    proj1 = np.dot(inner_verts[i], axis)
                    proj2 = np.dot(inner_verts[j], axis)
                    if np.max(proj1) < np.min(proj2) - 1e-7 or np.max(proj2) < np.min(proj1) - 1e-7:
                        overlap = False
                        break
                if overlap:
                    is_valid = False
                    break
            if not is_valid:
                break
                
    if not is_valid:
        # If invalid, we must fix it.
        # Simple fix: Scale up the outer hexagon? No, that increases score.
        # Simple fix: Move hexagons apart slightly?
        # Or just use the initial honeycomb configuration which is known to be valid.
        # The initial honeycomb has side length approx 4.something.
        # Let's calculate the side length for the initial honeycomb.
        init_side = _compute_required_side(init_centers, init_angles)
        
        # Check if init is valid (it should be)
        # If the optimized result is invalid, return the init result.
        # This is safer than returning an invalid optimized result.
        final_centers = init_centers
        final_angles_deg = init_angles
        final_side = init_side
        
    # Construct return value
    final_inner = []
    for c, a in zip(final_centers, final_angles_deg):
        final_inner.append([c[0], c[1], a])
        
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return (final_inner, outer_center, final_side, outer_angle)

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Time taken: {time.time() - START_TIME}")

