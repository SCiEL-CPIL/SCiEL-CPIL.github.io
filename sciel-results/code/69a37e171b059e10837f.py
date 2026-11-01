"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq


def _objective_reduced(flat):
    """
    Objective function for reduced parameterization.
    Fixes point 0 at (0,0,0) and point 1 at (1,0,0) to remove translational and rotational symmetry.
    """
    points = np.zeros((N, D))
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2:] = flat.reshape(N-2, D)
    
    return _compute_score(points)


def _generate_icosahedron_seed():
    """Generate a seed based on the 12 vertices of an icosahedron plus 2 extra points."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Scale and center
    verts = verts - np.mean(verts, axis=0)
    # Normalize to unit sphere roughly
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # We need 14 points. 
    # Strategy: Place 12 points near the icosahedron vertices.
    # Place 2 points on the z-axis, one above and one below, or slightly inside.
    # For N=14, a good config is often two "rings" or an icosahedron with 2 axial points.
    
    # Let's try placing 2 points on the z-axis at +/- z_val
    # We will optimize the positions of all 14 points, but start with this structure.
    
    points = np.zeros((N, D))
    points[:12] = verts
    points[12] = [0, 0, 1.5]
    points[13] = [0, 0, -1.5]
    
    # Flatten and apply symmetry reduction
    # We need to map these 14 points to the reduced parameter space (N-2)*D
    # This is tricky because the reduction fixes p0 and p1.
    # Instead, we will run optimization on the full 14 points but use a penalty for symmetry?
    # No, better to stick to the reduced space for efficiency and stability.
    # To use the reduced space, we need to align the seed with p0=(0,0,0) and p1=(1,0,0).
    
    # Let's pick two points from the seed to be p0 and p1.
    # Ideally, they should be a minimum distance pair.
    # In an icosahedron, all edges are equal. Let's pick p0=verts[0] and p1=verts[1].
    # Then we need to rotate/translate the other points relative to these.
    
    p0 = points[0]
    p1 = points[1]
    
    # Vector from p0 to p1
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        return None
        
    # We want to map this configuration such that p0 is at origin and p1 is at (1,0,0).
    # This requires a rigid transformation (rotation + translation).
    # However, the reduced space optimization expects the *other* points to be in the
    # coordinate system where p0=0 and p1=(1,0,0).
    
    # Let's construct the transformation.
    # 1. Translate so p0 is at origin.
    pts_trans = points - p0
    
    # 2. Rotate so v is along x-axis.
    # v is now pts_trans[1]
    v_trans = pts_trans[1]
    norm_v_trans = np.linalg.norm(v_trans)
    x_axis = np.array([1.0, 0.0, 0.0])
    
    # Compute rotation matrix to align v_trans with x_axis
    # Cross product to get rotation axis
    cross = np.cross(v_trans, x_axis)
    cross_norm = np.linalg.norm(cross)
    
    if cross_norm < 1e-9:
        # Already aligned or anti-aligned
        if np.dot(v_trans, x_axis) < 0:
            # 180 degree rotation around y-axis
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:
            R = np.eye(3)
    else:
        # Rodrigues' rotation formula
        v_hat = v_trans / norm_v_trans
        x_hat = x_axis
        dot = np.dot(v_hat, x_hat)
        cross_vec = np.cross(v_hat, x_hat)
        cross_norm_sq = np.dot(cross_vec, cross_vec)
        R = np.eye(3) + np.outer(cross_vec, cross_vec) * (1 - dot) / cross_norm_sq - np.sqrt(cross_norm_sq) * np.array([
            [0, -cross_vec[2], cross_vec[1]],
            [cross_vec[2], 0, -cross_vec[0]],
            [-cross_vec[1], cross_vec[0], 0]
        ])
        
    # Apply rotation
    pts_rot = pts_trans @ R.T
    
    # Now p0 is at (0,0,0) and p1 is at (norm_v, 0, 0).
    # We need p1 to be at (1, 0, 0). So we scale by 1/norm_v.
    pts_scaled = pts_rot / norm_v
    
    # Now pts_scaled[0] is (0,0,0) and pts_scaled[1] is (1,0,0).
    # The remaining points are pts_scaled[2:].
    # We flatten pts_scaled[2:] to get the initial guess for the reduced optimizer.
    init_flat = pts_scaled[2:].flatten()
    
    return init_flat


def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses deterministic local optimization (L-BFGS-B) from multiple structured seeds.
    """
    best_score = np.inf
    best_points = None
    
    # Define bounds for the reduced parameters
    # Points are roughly within [-2, 2]
    bounds = [(-3, 3)] * ((N-2) * D)
    
    # Seed 1: Icosahedron-based
    init_ico = _generate_icosahedron_seed()
    if init_ico is not None:
        res = minimize(
            _objective_reduced, 
            init_ico, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if res.fun < best_score:
            best_score = res.fun
            # Reconstruct points
            pts = np.zeros((N, D))
            pts[0] = [0, 0, 0]
            pts[1] = [1, 0, 0]
            pts[2:] = res.x.reshape(N-2, D)
            best_points = pts
    
    # Seed 2: Random but structured (e.g., points on a sphere with noise)
    # We can generate a few random seeds and optimize them
    rng = np.random.RandomState(42)
    for i in range(5):
        # Generate random points on a sphere of radius 1
        pts_rand = rng.randn(N, D)
        norms = np.linalg.norm(pts_rand, axis=1)
        pts_rand = pts_rand / norms[:, np.newaxis]
        
        # Apply the same reduction logic
        p0 = pts_rand[0]
        p1 = pts_rand[1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            continue
            
        pts_trans = pts_rand - p0
        v_trans = pts_trans[1]
        x_axis = np.array([1.0, 0.0, 0.0])
        
        cross = np.cross(v_trans, x_axis)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-9:
            if np.dot(v_trans, x_axis) < 0:
                R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            else:
                R = np.eye(3)
        else:
            v_hat = v_trans / norm_v
            x_hat = x_axis
            dot = np.dot(v_hat, x_hat)
            cross_vec = np.cross(v_hat, x_hat)
            cross_norm_sq = np.dot(cross_vec, cross_vec)
            R = np.eye(3) + np.outer(cross_vec, cross_vec) * (1 - dot) / cross_norm_sq - np.sqrt(cross_norm_sq) * np.array([
                [0, -cross_vec[2], cross_vec[1]],
                [cross_vec[2], 0, -cross_vec[0]],
                [-cross_vec[1], cross_vec[0], 0]
            ])
            
        pts_rot = pts_trans @ R.T
        pts_scaled = pts_rot / norm_v
        
        init_flat = pts_scaled[2:].flatten()
        
        res = minimize(
            _objective_reduced, 
            init_flat, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if res.fun < best_score:
            best_score = res.fun
            pts = np.zeros((N, D))
            pts[0] = [0, 0, 0]
            pts[1] = [1, 0, 0]
            pts[2:] = res.x.reshape(N-2, D)
            best_points = pts
            
    # If no valid points were found (should not happen), return a fallback
    if best_points is None:
        # Fallback: simple grid
        best_points = np.zeros((N, D))
        best_points[0] = [0,0,0]
        best_points[1] = [1,0,0]
        for i in range(2, N):
            best_points[i] = [1, 1, 0] + i*0.1
            
    return best_points
# EVOLVE-BLOCK-END
