# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Assume the optimal configuration has D3h or D6h symmetry.
    2. Parameterize 14 points as:
       - 2 poles on the z-axis: (0, 0, +/- h)
       - 6 points in a ring at z=0: radius r1, angle offset 0
       - 6 points in a ring at z=z2: radius r2, angle offset phi
    3. Optimize the parameters (h, r1, z2, r2, phi) to maximize d_min 
       subject to d_max = 1 (scale invariance).
    4. Use a smooth objective: maximize d_min - penalty * (d_max - 1)^2 
       or simply normalize and maximize d_min.
       Actually, simpler: Fix scale by setting d_max=1 is hard in optimization.
       Better: Maximize d_min / d_max. This is smooth almost everywhere.
       Even better: Since we want to maximize ratio, we can just maximize d_min 
       after normalizing the point cloud to have d_max=1.
       
       Let's use a penalty approach on the parameters directly:
       Objective: -d_min(pts(params)) + lambda * (max(0, d_max(pts(params)) - 1.0))^2
       But scale invariance makes this tricky. 
       
       Alternative: 
       Let params = [h, r1, z2, r2, phi].
       Generate points.
       Center points.
       Compute d_min, d_max.
       Ratio = d_min / d_max.
       Maximize Ratio.
       
       This is a 5D optimization problem. We can use Differential Evolution 
       for robust global search in 5D, then polish with L-BFGS-B.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

    def generate_points_from_params(params):
        """
        Generate 14 points from 5 parameters: h, r1, z2, r2, phi.
        Symmetry: D6h subset (2 poles, 2 hex rings).
        """
        h, r1, z2, r2, phi = params
        
        # Poles
        p1 = np.array([0.0, 0.0, h])
        p2 = np.array([0.0, 0.0, -h])
        
        # Ring 1 (6 points at z=0)
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        x1 = r1 * np.cos(angles1)
        y1 = r1 * np.sin(angles1)
        z1 = np.zeros(6)
        ring1 = np.vstack([x1, y1, z1]).T
        
        # Ring 2 (6 points at z=z2)
        angles2 = angles1 + phi
        x2 = r2 * np.cos(angles2)
        y2 = r2 * np.sin(angles2)
        z2_arr = np.full(6, z2)
        ring2 = np.vstack([x2, y2, z2_arr]).T
        
        points = np.vstack([p1, p2, ring1, ring2])
        
        # Center the points to remove translational invariance
        points -= np.mean(points, axis=0)
        
        return points

    def objective(params):
        """
        Minimize negative ratio (d_min / d_max).
        """
        try:
            if time.time() > deadline - 5:
                return 0.0
            
            points = generate_points_from_params(params)
            
            # Check for valid points
            if not np.all(np.isfinite(points)):
                return 0.0
                
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                return 0.0
                
            ratio = d_min / d_max
            return -ratio
        except Exception:
            return 0.0

    # Define bounds for the 5 parameters
    # h: height of poles
    # r1: radius of ring 1
    # z2: height of ring 2
    # r2: radius of ring 2
    # phi: angle offset of ring 2
    
    # We can assume h > 0, r1 > 0, r2 > 0, z2 can be positive or negative 
    # (but by symmetry we can fix relative positions).
    # Let's search in a reasonable box.
    # Since we center and normalize, scale doesn't matter, but relative proportions do.
    # Let's fix the scale of the search space to [-2, 2] for all params to let DE explore.
    
    bounds = [
        (-2.0, 2.0),  # h
        (0.1, 2.0),   # r1 (positive radius)
        (-2.0, 2.0),  # z2
        (0.1, 2.0),   # r2 (positive radius)
        (0.0, np.pi/3) # phi (symmetry of hexagon makes 0 to pi/3 sufficient)
    ]

    best_params = None
    best_score = -np.inf

    # 1. Global Search with Differential Evolution in 5D
    # 5D is small, so DE is very fast and robust.
    try:
        res_de = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            polish=False,
            init='latinhypercube',
            workers=-1
        )
        
        if res_de.fun < -best_score:
            best_params = res_de.x
            best_score = -res_de.fun
    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B
    # L-BFGS-B is efficient for low dimensions and can handle bounds.
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_lbfgs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-10
                }
            )
            if res_lbfgs.fun < -best_score:
                best_params = res_lbfgs.x
                best_score = -res_lbfgs.fun
        except Exception:
            pass

    # 3. Try a few other symmetry configurations if time permits?
    # Configuration B: 1 pole, 13 others? Unlikely.
    # Configuration C: 2 poles, 1 ring of 12?
    # Let's stick to the primary high-symmetry candidate but do a quick 
    # random restart of DE with different seeds if time is abundant.
    
    if time.time() < deadline - 20:
        # Try a second DE run with different seed to ensure we didn't miss a local minimum
        try:
            res_de2 = differential_evolution(
                objective,
                bounds,
                seed=123,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                polish=False,
                init='latinhypercube',
                workers=-1
            )
            if res_de2.fun < -best_score:
                best_params = res_de2.x
                best_score = -res_de2.fun
                
                # Polish again
                res_lbfgs2 = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res_lbfgs2.fun < -best_score:
                    best_params = res_lbfgs2.x
                    best_score = -res_lbfgs2.fun
        except Exception:
            pass

    # Generate final points
    if best_params is None:
        # Fallback: Random points
        points = np.random.uniform(-1, 1, (14, 3))
    else:
        points = generate_points_from_params(best_params)
        
        # Final normalization to ensure d_max is reasonable (though metric is scale invariant)
        # The task asks for coordinates. Any scale is valid as long as ratio is max.
        # Let's normalize so that d_max = 1 for cleanliness.
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            points = points / d_max

    # Fallback if points are invalid
    if not np.all(np.isfinite(points)) or len(points) != 14:
        # Simple fallback: Icosahedron + 2 poles
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        points = np.vstack([ico_verts, pole1, pole2])
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
