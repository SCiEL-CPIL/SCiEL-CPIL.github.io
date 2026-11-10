# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    # Objective function: Minimize negative score
    # Score R = (d_min / d_max)^2
    # We fix P0=(0,0) and P1=(1,0) to break translation and rotation symmetry.
    # This leaves 14 points * 2 coords = 28 variables.
    
    def get_score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_reduced(x_flat):
        # x_flat has length 28 (for points 2 to 15)
        pts = np.zeros((n, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x_flat.reshape((n - 2, 2))
        
        # Add a small penalty if points are too far apart to keep them bounded?
        # The score naturally penalizes large d_max.
        # But if d_max -> inf, score -> 0.
        # If d_min -> 0, score -> 0.
        # So the function is bounded and well-behaved.
        
        return -get_score(pts)

    # Bounds for the 28 variables
    # Points should be roughly within [-2, 2] relative to the fixed points
    bounds = [(-2.0, 2.0)] * (n * d - 4)

    best_x = None
    best_score = -np.inf
    best_pts = None

    # Strategy 1: Differential Evolution (Global Search)
    # Use a relatively small population to keep it fast, but enough for 28 vars
    # popsize=15 is default, let's use 10 to save time, maxiter=200
    try:
        res_de = differential_evolution(
            objective_reduced,
            bounds,
            maxiter=200,
            popsize=10,
            tol=1e-10,
            seed=42,
            polish=False
        )
        
        if time.time() > deadline - 60:
            # Not enough time for polish, return DE result
            best_x = res_de.x
            best_pts = np.zeros((n, 2))
            best_pts[0] = [0.0, 0.0]
            best_pts[1] = [1.0, 0.0]
            best_pts[2:] = best_x.reshape((n - 2, 2))
            return best_pts

        best_x = res_de.x
        current_score = -res_de.fun
        best_score = current_score
        best_pts = np.zeros((n, 2))
        best_pts[0] = [0.0, 0.0]
        best_pts[1] = [1.0, 0.0]
        best_pts[2:] = best_x.reshape((n - 2, 2))

    except Exception:
        # Fallback to random if DE fails
        np.random.seed(42)
        best_pts = np.random.randn(n, 2)
        best_x = best_pts[2:].flatten()
        best_score = get_score(best_pts)

    # Strategy 2: Local Polish with Nelder-Mead
    # Nelder-Mead is good for non-smooth functions
    try:
        res_nm = minimize(
            objective_reduced,
            best_x,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        
        if res_de.fun > res_nm.fun: # Remember we are minimizing negative score
            best_x = res_nm.x
            best_pts = np.zeros((n, 2))
            best_pts[0] = [0.0, 0.0]
            best_pts[1] = [1.0, 0.0]
            best_pts[2:] = best_x.reshape((n - 2, 2))
            best_score = -res_nm.fun

    except Exception:
        pass

    # Strategy 3: Try a Hexagonal Lattice Init and Polish
    # This might find a better local optimum if DE got stuck
    # Generate hex lattice points
    def hex_lattice_init():
        # Generate points in hex lattice
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1 (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2 (12 points)
        # In hex lattice, ring 2 has 12 points.
        # We need 16 total. We have 1 + 6 = 7. We need 9 more from ring 2.
        # The 12 points in ring 2 are at distance 2 from center.
        # Angles are shifted.
        # Let's just generate all 19 and pick 16.
        # Actually, let's just use a random subset of the 12 outer points?
        # Better: Use the 16 points with smallest distance to center?
        # All ring 2 points are distance 2.
        # To maximize min distance, we should space them evenly.
        # 9 points on a circle of radius 2?
        # Let's just take 9 evenly spaced points from the 12 hex lattice points.
        
        ring2_points = []
        for i in range(12):
            # Hex lattice ring 2 angles are not uniform in Cartesian, but let's approximate
            # Or just generate axial coords
            pass
            
        # Simpler: Just use the DE result as the primary driver.
        # The hex lattice is a good init, but DE should find it or something better.
        # If we have time, we can try optimizing a specific hex-like structure.
        
        # Let's try one more restart with a different seed for DE if time permits
        if time.time() < deadline - 120:
             try:
                res_de2 = differential_evolution(
                    objective_reduced,
                    bounds,
                    maxiter=100,
                    popsize=10,
                    tol=1e-10,
                    seed=123,
                    polish=False
                )
                if res_de2.fun < best_score * -1: # best_score is positive R, fun is negative
                     # Wait, best_score is R. res_de2.fun is -R.
                     # So we want res_de2.fun < -best_score
                     if res_de2.fun < -best_score:
                         best_x = res_de2.x
                         best_pts = np.zeros((n, 2))
                         best_pts[0] = [0.0, 0.0]
                         best_pts[1] = [1.0, 0.0]
                         best_pts[2:] = best_x.reshape((n - 2, 2))
                         best_score = -res_de2.fun
                         # Polish this new best
                         if time.time() < deadline - 30:
                             res_nm2 = minimize(
                                 objective_reduced,
                                 best_x,
                                 method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                             )
                             if res_nm2.fun < -best_score:
                                 best_x = res_nm2.x
                                 best_pts = np.zeros((n, 2))
                                 best_pts[0] = [0.0, 0.0]
                                 best_pts[1] = [1.0, 0.0]
                                 best_pts[2:] = best_x.reshape((n - 2, 2))
                                 best_score = -res_nm2.fun
             except Exception:
                 pass

    return best_pts

# EVOLVE-BLOCK-END
