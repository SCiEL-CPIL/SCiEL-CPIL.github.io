"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                pass
    # Explicit list
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    return v

def _face_center_opposite_faces(verts):
    """Find centers of two opposite faces of the icosahedron.
    For a regular icosahedron, opposite faces are related by central symmetry."""
    # The icosahedron has 20 faces. We'll find one face and its opposite.
    # A face consists of 3 mutually adjacent vertices.
    # For the standard icosahedron, face (0, 4, 8) is one face.
    # Its opposite face would be (2, 5, 11) or similar.
    # Let's compute: face center = average of 3 vertices.
    # Face 1: vertices 0, 4, 8
    f1 = (verts[0] + verts[4] + verts[8]) / 3.0
    # Opposite face: vertices 2, 5, 11 (centrally symmetric)
    f2 = (verts[2] + verts[5] + verts[11]) / 3.0
    return f1, f2

def _rotate_z(v, angle):
    """Rotate points around z-axis by angle."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return v @ R.T

def _objective(params):
    """Compute smooth surrogate for squared distance ratio.
    params = [r1, r2, theta] where r1, r2 are radial scales for the 2 extra points,
    theta is rotation angle."""
    r1, r2, theta = params
    verts = _icosahedron_vertices()
    f1, f2 = _face_center_opposite_faces(verts)
    
    # Rotate the icosahedron vertices
    verts_rot = _rotate_z(verts, theta)
    f1_rot = _rotate_z(f1.reshape(1, -1), theta)[0]
    f2_rot = _rotate_z(f2.reshape(1, -1), theta)[0]
    
    # Place the two extra points at scaled face centers
    p13 = r1 * f1_rot
    p14 = r2 * f2_rot
    
    points = np.vstack([verts_rot, p13.reshape(1, -1), p14.reshape(1, -1)])
    
    # Compute all pairwise squared distances
    n = len(points)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = sq_dists[iu]
    
    if len(d) == 0:
        return 1e10
    
    d_min = np.min(d)
    d_max = np.max(d)
    
    if d_min < 1e-12:
        return 1e10
    
    # Smooth surrogate: use log-sum-exp for max, and -logsumexp(-d) for min
    # But for Nelder-Mead, the raw ratio might work well enough.
    # Let's use a soft version:
    # soft_max = log(sum(exp(d / tau))) * tau
    # soft_min = -log(sum(exp(-d / tau))) * tau
    tau = 0.1
    soft_max = tau * np.log(np.sum(np.exp(d / tau)))
    soft_min = -tau * np.log(np.sum(np.exp(-d / tau)))
    
    if soft_min < 1e-12:
        return 1e10
    
    ratio_sq = (soft_max / soft_min) ** 2
    return ratio_sq

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s to be safe
    
    best_points = None
    best_score = np.inf
    
    verts = _icosahedron_vertices()
    f1, f2 = _face_center_opposite_faces(verts)
    
    # Initial guess: r1, r2 close to 1 (face centers are inside), theta = 0
    # Face centers have norm < 1, so scaling up may help
    f1_norm = np.linalg.norm(f1)
    f2_norm = np.linalg.norm(f2)
    
    initial_params = [f1_norm, f2_norm, 0.0]
    
    # Multi-start Nelder-Mead
    n_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        if i == 0:
            x0 = np.array(initial_params)
        else:
            # Perturb initial guess
            r1_pert = f1_norm * (0.5 + rng.random() * 1.0)
            r2_pert = f2_norm * (0.5 + rng.random() * 1.0)
            theta_pert = rng.uniform(-np.pi, np.pi)
            x0 = np.array([r1_pert, r2_pert, theta_pert])
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                r1, r2, theta = result.x
                verts_rot = _rotate_z(verts, theta)
                f1_rot = _rotate_z(f1.reshape(1, -1), theta)[0]
                f2_rot = _rotate_z(f2.reshape(1, -1), theta)[0]
                p13 = r1 * f1_rot
                p14 = r2 * f2_rot
                best_points = np.vstack([verts_rot, p13.reshape(1, -1), p14.reshape(1, -1)])
                
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to the icosahedron + unscaled face centers
    if best_points is None:
        best_points = np.vstack([verts, f1.reshape(1, -1), f2.reshape(1, -1)])
    
    return best_points
# EVOLVE-BLOCK-END
