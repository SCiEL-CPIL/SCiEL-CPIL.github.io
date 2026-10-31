"""Optimized 16-point placement: minimize (max_dist/min_dist)^2.

Strategy:
1. Parameterize 16 points as a deformed hexagonal lattice cluster.
   - 3 global params: shear, aspect ratio, global scale.
   - 16 local params: radial perturbations for each point.
   Total 19 dimensions.
2. Use Differential Evolution for global search.
3. Use L-BFGS-B for local refinement.
4. Optionally, a final unconstrained 32D polish if time permits.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2
DEADLINE = 140.0  # safety margin before 150s kill

# Define base hexagonal lattice points for 16 points
# Center + 6 neighbors + 9 from next ring
def _get_base_hex_points():
    pts = []
    # Center
    pts.append((0.0, 0.0))
    # Ring 1: 6 points
    for k in range(6):
        angle = np.pi / 3 * k
        pts.append((np.cos(angle), np.sin(angle)))
    # Ring 2: We need 9 more. The full ring has 12.
    # Hex coords (u, v) -> (u + v/2, v*sqrt(3)/2)
    # Ring 2 coords:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,0 is ring 1?), 
    # Let's list Ring 2 explicitly in Cartesian for a unit hex lattice.
    # Actually, easier to generate in axial coords and convert.
    # Axial (q, r). Cart: x = q + r/2, y = r * sqrt(3)/2.
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-2 is duplicate? No)
    # Let's just take the 12 points of ring 2 and pick 9.
    # To maintain symmetry as much as possible, we might pick a symmetric subset?
    # Or just take the first 9? The deformation will handle asymmetry.
    # For better initial symmetry, let's pick 9 that are somewhat balanced.
    # Actually, any 9 will do because we have 16 radial DOFs.
    ring2_coords = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -1), (1, -2), (2, 0) # duplicates in my head, let's be precise
    ]
    # Precise Ring 2 in axial:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-2 is wrong), (2,0 is wrong)
    # Correct Ring 2 axial:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-2) -> wait.
    # Neighbors of (1,0) are (2,0), (1,1), (0,1), (0,-1), (1,-1), (2,-1).
    # Let's just generate all axial points with distance 2 and take 9.
    import itertools
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # distance in axial is max(|q|, |r|, |q+r|)? No, that's cube coords.
            # In axial, distance is (|q| + |r| + |q+r|) / 2.
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            if dist == 2.0:
                ring2.append((q, r))
    # There should be 12 points.
    # Take first 9.
    selected_ring2 = ring2[:9]
    
    for q, r in selected_ring2:
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        pts.append((x, y))
        
    return np.array(pts)

BASE_PTS = _get_base_hex_points()

def _reconstruct_from_params(params):
    """
    params: (19,)
    0: shear
    1: aspect_ratio (scale y relative to x)
    2: global_scale
    3-18: radial perturbations for each of the 16 points
    """
    shear = params[0]
    aspect = params[1]
    g_scale = params[2]
    deltas = params[3:19]
    
    # Base points
    pts = BASE_PTS.copy()
    
    # Apply linear transformation: 
    # x' = x + shear * y
    # y' = aspect * y
    # Then scale
    # This allows shearing and stretching.
    # We also want to allow rotation? The base hex is aligned. 
    # Shear + Aspect covers affine transformations that preserve the "hexagonal" character mostly.
    # A general affine transform has 4 params (2x2 matrix). We have 2. 
    # Adding rotation and uniform scale might be good.
    # Let's use:
    # 1. rotation
    # 2. aspect ratio (stretch y)
    # 3. shear
    # 4. global scale
    # That's 4 global params. Total 20.
    # Let's stick to 3 global + 16 local = 19 for now, but maybe adjust.
    # Let's use:
    # p0: rotation
    # p1: aspect (scale y)
    # p2: shear
    # p3: global scale
    # p4-19: radial deltas
    # Total 20 params.
    
    # Re-define param unpacking for 20 params
    # But the function signature expects 19 in the current draft? 
    # I will change the code to use 20 params.
    
    # Wait, the prompt says "rewrite from scratch". I can define the parameterization freely.
    # Let's use 20 params: 4 global, 16 local.
    
    # This function is called with 20 params.
    rot = params[0]
    aspect = params[1]
    shear = params[2]
    g_scale = params[3]
    deltas = params[4:20]
    
    # 1. Rotate base points
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pts_rot = pts @ R.T
    
    # 2. Shear and Aspect
    # x' = x + shear * y
    # y' = aspect * y
    pts_shear = pts_rot.copy()
    pts_shear[:, 0] = pts_shear[:, 0] + shear * pts_shear[:, 1]
    pts_shear[:, 1] = aspect * pts_shear[:, 1]
    
    # 3. Radial Perturbations
    # For each point, add a small vector in its radial direction (from origin)
    # If point is at origin, perturb in x-direction.
    for i in range(N):
        p = pts_shear[i]
        r = np.linalg.norm(p)
        if r > 1e-6:
            direction = p / r
        else:
            direction = np.array([1.0, 0.0])
        pts_shear[i] = p + deltas[i] * direction
        
    # 4. Global Scale
    pts_final = pts_shear * g_scale
    
    return pts_final

def _pairwise_dists_sq(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _objective(params):
    pts = _reconstruct_from_params(params)
    d2 = _pairwise_dists_sq(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def _exact_score(pts):
    d2 = _pairwise_dists_sq(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def construct():
    t0 = time.time()
    
    # Phase 1: Differential Evolution on 20D parameter space
    # Bounds:
    # rot: [0, 2*pi]
    # aspect: [0.5, 2.0]
    # shear: [-1.0, 1.0]
    # g_scale: [0.5, 2.0]
    # deltas: [-0.5, 0.5] (small perturbations)
    
    bounds = [
        (0.0, 2*np.pi),
        (0.5, 2.0),
        (-1.0, 1.0),
        (0.5, 2.0)
    ] + [(-0.5, 0.5)] * 16
    
    best_params = None
    best_score = np.inf
    best_pts = None
    
    try:
        # DE is stochastic, use multiple seeds or just a good one
        # time_limit to ensure we don't run over
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            popsize=15,
            polish=True,
            init='sobol',
            time_limit=60
        )
        if time.time() - t0 < DEADLINE:
            best_params = de_result.x
            score = _objective(best_params)
            if score < best_score:
                best_score = score
                best_pts = _reconstruct_from_params(best_params)
                
    except Exception:
        pass
    
    # Phase 2: Local Polish with L-BFGS-B on 20D params
    if best_params is not None and time.time() - t0 < DEADLINE - 10:
        try:
            res = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if time.time() - t0 < DEADLINE - 5:
                new_score = _objective(res.x)
                if new_score < best_score:
                    best_score = new_score
                    best_params = res.x
                    best_pts = _reconstruct_from_params(best_params)
        except Exception:
            pass
    
    # Phase 3: Unconstrained 32D Polish
    # Sometimes the lattice constraint is too tight. Allow free movement.
    if best_pts is not None and time.time() - t0 < DEADLINE - 10:
        init_32 = best_pts.flatten()
        
        def obj_32(p):
            pts = p.reshape(N, D)
            return _exact_score(pts)
        
        try:
            res_32 = minimize(
                obj_32,
                init_32,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if time.time() - t0 < DEADLINE - 5:
                score_32 = _exact_score(res_32.x.reshape(N, D))
                if score_32 < best_score:
                    best_score = score_32
                    best_pts = res_32.x.reshape(N, D)
        except Exception:
            pass
            
    # Fallback if all else fails
    if best_pts is None:
        # Simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
