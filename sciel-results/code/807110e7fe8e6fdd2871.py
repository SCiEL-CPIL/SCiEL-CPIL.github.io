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
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Hexagonal lattice parameterization:
    # Use a hexagonal lattice with 19 points (radius 2), then remove 3 points.
    # The 19-point hex cluster is known to be near-optimal for dense 2D configurations.
    # Parameters: scale (s), aspect ratio (a), angle (theta), and 32 perturbations.
    # We fix the lattice structure and optimize the perturbations + global params.

    # Generate base hexagonal lattice points (19 points: center + 6 ring1 + 12 ring2)
    def make_hex_points(scale, aspect, angle):
        """Generate 19 hexagonal lattice points."""
        # Hexagonal lattice vectors
        v1 = np.array([1.0, 0.0]) * scale
        v2 = np.array([0.5, np.sqrt(3)/2]) * scale * aspect
        
        # Rotate by angle
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        v1 = R @ v1
        v2 = R @ v2
        
        # Generate 19 points: (i,j) for i,j in [-2,2] with i+j even (hexagonal)
        # Actually, standard hex lattice: points at i*v1 + j*v2 for i,j in range
        # For a hex cluster of radius 2, we need |i| <= 2, |j| <= 2, |i+j| <= 2
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) <= 2 and abs(j) <= 2 and abs(i + j) <= 2:
                    pt = i * v1 + j * v2
                    pts.append(pt)
        return np.array(pts)  # Should be 19 points

    # Verify: hex cluster of radius 2 has 1 + 6 + 12 = 19 points
    # We need to remove 3 to get 16.

    # Strategy: Optimize 16 points directly but initialize from hex lattice
    # with perturbations, and use the hex structure as a strong prior.

    def hex_init_with_perturbations(scale, aspect, angle, perturbations):
        """Generate 16 points from hex lattice with small perturbations."""
        base_pts = make_hex_points(scale, aspect, angle)
        # base_pts should have 19 points. Remove 3 to get 16.
        # Which 3 to remove? We'll optimize this too, but for now,
        # remove the 3 farthest from center (the "corners" of the hex)
        # Actually, let's just take the first 16 and optimize all 32 coords.
        
        # Better: use all 19 points as a starting template, pick 16
        # For simplicity, let's just optimize 16 free coordinates
        # but initialize from a hex-like pattern.
        
        # Create 16 points from the 19-point hex by removing 3
        # The 19 points are ordered; let's remove indices that are "outermost"
        # For a hex of radius 2, the outer ring has 12 points.
        # Let's just use a simpler approach: optimize all 32 coordinates
        # but initialize from a scaled hex pattern.
        
        # Generate 16 points by taking a hex pattern
        # Use a 4x4 grid rotated 45 degrees as an alternative, or
        # use the hex lattice directly with 16 selected points.
        
        # Simplest: create 16 points in a hex-like arrangement
        # Ring 0: 1 point (center)
        # Ring 1: 6 points
        # Ring 2: 9 points (out of 12, remove 3)
        # Total: 16
        
        pts = []
        # Center
        pts.append(np.array([0.0, 0.0]))
        
        # Ring 1: 6 points
        for k in range(6):
            ang = k * np.pi / 3
            pts.append(scale * np.array([np.cos(ang), np.sin(ang)] * aspect))
        
        # Ring 2: 12 points, remove 3
        for k in range(12):
            ang = k * np.pi / 6
            # Skip every 4th point to remove 3 from 12
            if k % 4 != 3:  # This removes 3 points
                pts.append(2 * scale * np.array([np.cos(ang), np.sin(ang)] * aspect))
        
        # Apply rotation
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        pts = np.array(pts) @ R.T
        
        # Apply perturbations
        pts += perturbations.reshape(16, 2)
        
        return pts

    def objective_hex(params):
        """Objective for hexagonal lattice parameterization."""
        # params: [scale, aspect, angle, 32 perturbations]
        scale = params[0]
        aspect = params[1]
        angle = params[2]
        perturbations = params[3:35]
        
        pts = hex_init_with_perturbations(scale, aspect, angle, perturbations)
        return -get_score(pts)

    def soft_objective_free(x, t=5.0):
        """Smooth surrogate for free coordinate optimization."""
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-12)
        
        # Soft min using log-sum-exp
        a_min = -dists / t
        a_min_max = np.max(a_min)
        soft_min = -t * (a_min_max + np.log(np.sum(np.exp(a_min - a_min_max)) / len(dists)))
        
        # Soft max using log-sum-exp
        a_max = dists / t
        a_max_max = np.max(a_max)
        soft_max = t * (a_max_max + np.log(np.sum(np.exp(a_max - a_max_max)) / len(dists)))
        
        return -(soft_min - soft_max)

    best_points = None
    best_score = -np.inf

    # Strategy 1: Hexagonal lattice parameterization with DE
    if time.time() < deadline - 60:
        try:
            # Bounds: scale in [0.5, 5], aspect in [0.5, 2], angle in [0, pi]
            # Perturbations in [-0.5, 0.5]
            bounds_hex = [(0.5, 5.0), (0.5, 2.0), (0.0, np.pi)]
            bounds_hex += [(-0.5, 0.5)] * 32
            
            result = differential_evolution(
                objective_hex,
                bounds_hex,
                seed=42,
                maxiter=500,
                popsize=25,
                tol=1e-8,
                polish=True,
                init='sobol',
                workers=1
            )
            
            params = result.x
            pts = hex_init_with_perturbations(params[0], params[1], params[2], params[3:35])
            score = get_score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass

    # Strategy 2: Free coordinate optimization with DE from hex init
    if time.time() < deadline - 30:
        try:
            # Initialize from best hex result or default hex
            if best_points is not None:
                init_pts = best_points
            else:
                init_pts = hex_init_with_perturbations(1.0, 1.0, 0.0, np.zeros(32))
            
            # Normalize to unit diameter for better conditioning
            d_max_init = np.max(pdist(init_pts))
            if d_max_init > 0:
                init_pts = init_pts / d_max_init
            
            # DE on free coordinates
            bounds_free = [(-2.0, 2.0)] * 32
            
            result = differential_evolution(
                lambda x: -get_score(x.reshape(16, 2)),
                bounds_free,
                seed=123,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol',
                x0=init_pts.flatten(),
                workers=1
            )
            
            pts = result.x.reshape(16, 2)
            score = get_score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass

    # Strategy 3: Local refinement with smooth surrogate
    if best_points is not None:
        current_best = best_points.copy()
        current_score = best_score
        
        # Try multiple temperature values for soft-min/max
        for t in [5.0, 2.0, 1.0, 0.5]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    soft_objective_free,
                    current_best.flatten(),
                    args=(t,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                refined_points = res.x.reshape(n, 2)
                refined_score = get_score(refined_points)
                
                if refined_score > current_score:
                    current_best = refined_points
                    current_score = refined_score
            except Exception:
                pass
        
        best_points = current_best
        best_score = current_score

    # Fallback
    if best_points is None:
        # Simple 4x4 grid
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        best_score = get_score(best_points)

    # Final validation
    if best_points is None or best_points.shape != (16, 2) or not np.all(np.isfinite(best_points)):
        # Emergency fallback
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))

    return best_points


# EVOLVE-BLOCK-END
