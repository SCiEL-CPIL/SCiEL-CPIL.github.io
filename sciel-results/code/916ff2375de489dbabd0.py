import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_symmetric_points(params):
    """
    Generate 14 points with D3d symmetry:
    - 2 points on the z-axis (poles)
    - 12 points in 3 rings of 4, staggered by 45 degrees.
    
    params:
    p[0]: z-height of the poles (should be ~1 for unit sphere, but we allow optimization)
    p[1]: z-height of ring 0
    p[2]: z-height of ring 1
    p[3]: z-height of ring 2
    p[4]: radius of ring 0
    p[5]: radius of ring 1
    p[6]: radius of ring 2
    p[7]: radius of ring 3 (wait, 3 rings of 4 is 12 points. 2 poles + 12 = 14. 
          Actually, 3 rings of 4? 
          Ring 0: 4 points at angle 0, 90, 180, 270
          Ring 1: 4 points at angle 45, 135, 225, 315
          Ring 2: 4 points at angle 0, 90, 180, 270 (or 45?)
          
          Let's use 3 rings. 
          Ring 0: z=z0, r=r0, angles: 0, 90, 180, 270
          Ring 1: z=z1, r=r1, angles: 45, 135, 225, 315
          Ring 2: z=z2, r=r2, angles: 0, 90, 180, 270
          
          This gives 12 points. Plus 2 poles. Total 14.
          Symmetry group is D3d? No, D3d has 3-fold symmetry. 4-fold is D4h or D2d?
          4-fold symmetry with staggered rings is D4d or S4?
          Actually, 14 points often relate to the 14 vertices of the "small stellated dodecahedron" or similar?
          No, 14 points in 3D often refers to the 14 vertices of the "gyroelongated square bipyramid" or similar Johnson solids?
          Or simply 2 poles + 12 in a structure.
          
          Let's stick to 4-fold symmetry (D4d-like) which is a very strong candidate for packing 14 points.
          Points:
          (0,0,1), (0,0,-1)
          Ring 1: 4 pts at z=z1, r=r1, angles 0, 90, 180, 270
          Ring 2: 4 pts at z=z2, r=r2, angles 45, 135, 225, 315
          Ring 3: 4 pts at z=z3, r=r3, angles 0, 90, 180, 270
          
          Wait, if we have 3 rings, we have 3 z's and 3 r's. Plus pole z.
          Total params: 7.
          
          Let's refine the parameter vector:
          p[0]: z_pole
          p[1]: z_ring1
          p[2]: z_ring2
          p[3]: z_ring3
          p[4]: r_ring1
          p[5]: r_ring2
          p[6]: r_ring3
          
          We can assume the sphere is unit radius, so r_i^2 + z_i^2 = 1.
          This reduces params to:
          p[0]: z_pole
          p[1]: z_ring1
          p[2]: z_ring2
          p[3]: z_ring3
          
          Then r_i = sqrt(1 - z_i^2).
          If z_i^2 > 1, r is invalid. We can use a soft constraint or just clip.
          Better: parameterize by angle theta.
          p[0]: theta_pole (0 for north, pi for south? No, poles are fixed at 0 and pi usually for max spread? 
                 Actually, maybe poles are not exactly at poles? 
                 In D4d symmetry, the poles are on the axis. 
                 Let's allow z_pole to vary.
                 
          Let's use Cartesian z-coordinates for simplicity, but constrain to sphere.
          Actually, just optimizing 7 parameters (3 z's for rings, 1 z for pole, 3 r's) and normalizing each point to unit length is robust.
          """
    z_pole = params[0]
    z1 = params[1]
    z2 = params[2]
    z3 = params[3]
    r1 = params[4]
    r2 = params[5]
    r3 = params[6]
    
    points = []
    
    # Poles
    points.append([0.0, 0.0, z_pole])
    points.append([0.0, 0.0, -z_pole])
    
    # Ring 1: angles 0, 90, 180, 270
    for a in [0, np.pi/2, np.pi, 3*np.pi/2]:
        points.append([r1 * np.cos(a), r1 * np.sin(a), z1])
        
    # Ring 2: angles 45, 135, 225, 315
    for a in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        points.append([r2 * np.cos(a), r2 * np.sin(a), z2])
        
    # Ring 3: angles 0, 90, 180, 270
    for a in [0, np.pi/2, np.pi, 3*np.pi/2]:
        points.append([r3 * np.cos(a), r3 * np.sin(a), z3])
        
    pts = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1)
    norms[norms == 0] = 1.0
    pts = pts / norms[:, np.newaxis]
    
    return pts

def _objective_symmetric(params):
    """
    Minimize sum of inverse distances to favor packing.
    """
    points = _generate_symmetric_points(params)
    dists = pdist(points, metric='euclidean')
    if np.any(dists < 1e-9):
        return 1e10
    # Use a high power to emphasize minimum distance
    return np.sum(1.0 / (dists ** 8))

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = -np.inf
    best_points = None
    
    # Strategy: Grid search over the 7 parameters, then local refinement.
    # Parameters:
    # z_pole in [0, 1]
    # z1, z2, z3 in [-1, 1]
    # r1, r2, r3 in [0, 1]
    
    # Initial coarse grid
    # We expect the optimal configuration to have some symmetry, e.g., z1 = -z3, r1 = r3.
    # Let's try to enforce this symmetry to reduce dimensions?
    # If z1 = -z3 and r1 = r3, then we have:
    # z_pole, z1, z2, r1, r2. (5 params)
    # Let's assume this symmetry holds initially, as it's a common packing motif.
    
    # Grid for 5 params:
    # z_pole: 0.8, 0.9, 1.0
    # z1: -0.8, -0.6, -0.4, -0.2, 0.0
    # z2: -0.4, -0.2, 0.0, 0.2, 0.4
    # r1: 0.6, 0.7, 0.8, 0.9
    # r2: 0.6, 0.7, 0.8, 0.9
    
    # Actually, let's just do a random restart + local optimization in the 7D space.
    # It's safer than a fixed grid.
    
    np.random.seed(42)
    
    # Define bounds for the 7 parameters
    # z_pole: [0, 1]
    # z1, z2, z3: [-1, 1]
    # r1, r2, r3: [0, 1]
    
    bounds = [
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0)
    ]
    
    # Multi-start L-BFGS-B
    n_starts = 100
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial guess
        # Bias towards symmetric configurations
        if i < 50:
            # Symmetric guess: z1 = -z3, r1 = r3
            z_pole = np.random.uniform(0.8, 1.0)
            z1 = np.random.uniform(-0.8, -0.2)
            z2 = np.random.uniform(-0.4, 0.4)
            z3 = -z1
            r1 = np.random.uniform(0.6, 0.9)
            r2 = np.random.uniform(0.6, 0.9)
            r3 = r1
            x0 = np.array([z_pole, z1, z2, z3, r1, r2, r3])
        else:
            # Random guess within bounds
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1]),
                np.random.uniform(bounds[3][0], bounds[3][1]),
                np.random.uniform(bounds[4][0], bounds[4][1]),
                np.random.uniform(bounds[5][0], bounds[5][1]),
                np.random.uniform(bounds[6][0], bounds[6][1])
            ])
            
        try:
            res = minimize(
                _objective_symmetric,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            pts = _generate_symmetric_points(res.x)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception:
            pass

    # Fallback: Icosahedron + Poles (standard seed)
    if best_points is None or best_score < 0.01:
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_xyz = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden], [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_xyz = ico_xyz / np.linalg.norm(ico_xyz, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1], [0, 0, -1]])
        best_points = np.vstack([ico_xyz, poles])
        best_score = _score(best_points)

    # Final safety check
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.0, neginf=0.0)
    if best_points.shape != (14, 3):
        best_points = np.eye(14, 3)
        
    return best_points
