"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Define the base hexagonal lattice coordinates for 16 points
    # Structure: 4-3-4-3-2-2 arrangement is complex. 
    # A simpler symmetric structure often optimal for such small N is a subset of a hex grid.
    # Let's use a 4x4 grid perturbed towards hexagonal symmetry, or a specific hex cluster.
    # Common optimal structures for N=16 in 2D are often related to the hexagonal lattice.
    # Let's define a "hexagonal" set of 16 points.
    # Center at (0,0).
    # Ring 1 (6 pts): hexagon
    # Ring 2 (10 pts): hexagon
    # Total 16? No, 1+6+10 = 17.
    # Let's try 1 + 6 + 9? Or just a 4x4 grid which is 16.
    # The 4x4 grid has ratio ~ 4.24.
    # A hexagonal packing of 16 points:
    # Rows: 2, 3, 4, 3, 2, 2? No, that's 16.
    # Let's use a set of integer coordinates from a hex lattice.
    # Axial coordinates (q, r) where x = q + r/2, y = r * sqrt(3)/2
    # We want 16 distinct points.
    # Let's pick a cluster around the origin.
    # q, r in {-2, -1, 0, 1, 2}
    # We can just use a 4x4 square grid as the base topology, but allow shear and scaling
    # to morph it into a better shape. The topology of connections (which points are neighbors)
    # in a 4x4 grid is fixed. We optimize the geometry.
    
    # Base 4x4 grid indices
    i = np.arange(4)
    j = np.arange(4)
    jj, ii = np.meshgrid(j, i)
    base_x = ii.ravel() - 1.5
    base_y = jj.ravel() - 1.5
    
    # To allow morphing into hexagonal-like structures, we apply a linear transformation (shear/aspect)
    # and a non-linear radial correction.
    
    def generate_points(params):
        """
        params: 
        0: aspect_ratio (scale y relative to x)
        1: shear (x += shear * y)
        2: radial_scale (applied to outer ring to break square symmetry)
        3: radial_power (how much the radial correction affects distance)
        """
        aspect = params[0]
        shear = params[1]
        r_scale = params[2]
        r_power = params[3]
        
        x = base_x.copy()
        y = base_y.copy()
        
        # Apply shear
        x = x + shear * y
        
        # Apply aspect ratio
        y = y * aspect
        
        # Compute radial distance from center
        r = np.sqrt(x**2 + y**2)
        
        # Apply radial scaling
        # We want to push outer points out or pull them in relative to inner points.
        # r_scale > 1 pushes outer points further out (if r is normalized).
        # Let's normalize r so max r is 1.
        r_max = np.max(r)
        if r_max > 1e-9:
            r_norm = r / r_max
        else:
            r_norm = r
            
        # Apply scaling factor: 1 + (r_scale - 1) * r_norm^r_power
        # If r_scale = 1, no change.
        # If r_scale > 1, outer points move out.
        factor = 1.0 + (r_scale - 1.0) * (r_norm ** r_power)
        
        x = x * factor
        y = y * factor
        
        return np.column_stack([x, y])

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def objective(params):
        pts = generate_points(params)
        return score_raw(pts)

    # Bounds for parameters
    # aspect: 0.5 to 1.5 (square is 1.0, hex is ~1.15 or 0.86 depending on orientation)
    # shear: -0.5 to 0.5
    # r_scale: 0.5 to 1.5
    # r_power: 1.0 to 5.0
    bounds = [
        (0.5, 1.5),
        (-0.5, 0.5),
        (0.5, 1.5),
        (1.0, 5.0)
    ]
    
    # Initial guesses
    # 1. Square grid (aspect=1, shear=0, r_scale=1, r_power=1)
    x0_square = np.array([1.0, 0.0, 1.0, 1.0])
    
    # 2. Hexagonal-like (aspect ~ 1.15, shear ~ 0.2, r_scale ~ 1.1)
    x0_hex = np.array([1.15, 0.2, 1.1, 2.0])
    
    # 3. Stretched (aspect ~ 0.8, shear ~ -0.2)
    x0_stretch = np.array([0.8, -0.2, 0.9, 3.0])
    
    best_score = 1e10
    best_params = x0_square
    
    # Strategy: Use Differential Evolution for global search in this low-dim space,
    # then polish with L-BFGS-B.
    
    # Time check
    if time.time() < deadline - 10:
        try:
            # DE is robust for small dims
            de_res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=10,
                tol=1e-6,
                polish=False # We will polish manually
            )
            best_params = de_res.x
            best_score = de_res.fun
            
            if time.time() < deadline - 5:
                # Polish with L-BFGS-B
                lb = [b[0] for b in bounds]
                ub = [b[1] for b in bounds]
                res = minimize(
                    objective, 
                    best_params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass

    # If DE didn't run or failed, try a few random starts with L-BFGS-B
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                continue

    # Generate final points
    pts = generate_points(best_params)
    
    # Optional: Final fine-tuning of individual points?
    # The structural constraint might be too tight. 
    # If we have time, we can release the structural constraint and do a few steps of 
    # unconstrained optimization on the 32 coordinates, starting from the structural solution.
    
    if time.time() < deadline - 10:
        try:
            # Unconstrained polish on coordinates
            def obj_coords(x):
                p = x.reshape(N, 2)
                return score_raw(p)
            
            # Initial guess from structural solution
            x_init = pts.ravel()
            
            res = minimize(
                obj_coords, 
                x_init, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                pts = res.x.reshape(N, 2)
        except Exception:
            pass

    return pts

# EVOLVE-BLOCK-END
