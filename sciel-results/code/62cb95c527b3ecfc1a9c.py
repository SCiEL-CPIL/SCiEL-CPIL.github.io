"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _compute_radii(centers):
    """
    Given centers, compute radii based on Delaunay triangulation.
    For each circle, radius is half the minimum distance to any other circle
    or to the boundary.
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Distance to boundary
    for i in range(n):
        d_boundary = min(centers[i][0], 1.0 - centers[i][0],
                         centers[i][1], 1.0 - centers[i][1])
        radii[i] = d_boundary
    
    # If n >= 3, use Delaunay triangulation to find nearest neighbors
    if n >= 3:
        try:
            tri = Delaunay(centers)
            # For each point, find nearest neighbor distance
            for i in range(n):
                # Get neighbors from Delaunay simplices
                neighbors = set()
                for simplex in tri.simplices:
                    if i in simplex:
                        for j in simplex:
                            if j != i:
                                neighbors.add(j)
                for j in neighbors:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = 0.5 * d
                    if r < radii[i]:
                        radii[i] = r
        except Exception:
            pass
    elif n == 2:
        d = np.linalg.norm(centers[0] - centers[1])
        r = 0.5 * d
        radii[0] = min(radii[0], r)
        radii[1] = min(radii[1], r)
    elif n == 1:
        pass
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 1e-9)
    return radii


def _objective(centers_flat):
    """
    Objective function to minimize (negative of sum of radii).
    Also penalize violations of constraints.
    """
    n = N
    centers = centers_flat.reshape(n, 2)
    
    # Check boundary constraints
    for i in range(n):
        if centers[i][0] < 0 or centers[i][0] > 1 or centers[i][1] < 0 or centers[i][1] > 1:
            return 1e10
    
    radii = _compute_radii(centers)
    
    # Check if circles are disjoint (they should be by construction, but verify)
    # Actually, by construction, radii are set so that circles just touch or are separated
    # But we need to ensure no overlap. Let's add a penalty if there's any overlap.
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - 1e-9:
                penalty += (radii[i] + radii[j] - d) ** 2
    
    return -(np.sum(radii)) + penalty


def _repulsion_force(centers, radii):
    """
    Compute repulsion forces between circles to help push them apart
    when they are too close.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = centers[i] - centers[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                # Random small perturbation
                d_vec = np.random.randn(2) * 1e-6
                d = np.linalg.norm(d_vec)
            min_dist = radii[i] + radii[j]
            if d < min_dist * 1.1:
                # Repulsive force
                f = (min_dist - d) / d
                d_unit = d_vec / d
                forces[i] += f * d_unit
                forces[j] -= f * d_unit
    
    return forces


def construct():
    """
    Pack 26 circles in the unit square to maximize sum of radii.
    Returns array of shape (26, 3) with rows (x, y, radius).
    """
    n = N
    
    # Initial configuration: grid-like
    side = int(np.ceil(np.sqrt(n)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                centers.append([x, y])
    centers = np.array(centers[:n])
    
    best_centers = centers.copy()
    best_sum_radii = -1e10
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if restart == 0:
            c = centers.copy()
        else:
            c = centers + np.random.randn(n, 2) * 0.05
            c = np.clip(c, 0.01, 0.99)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                c.flatten(),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * n),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_opt = result.x.reshape(n, 2)
            c_opt = np.clip(c_opt, 0, 1)
            
            radii = _compute_radii(c_opt)
            s = np.sum(radii)
            
            if s > best_sum_radii:
                best_sum_radii = s
                best_centers = c_opt.copy()
        except Exception:
            pass
    
    # Final radii
    radii = _compute_radii(best_centers)
    
    # Construct result
    result = np.zeros((n, 3))
    result[:, 0] = best_centers[:, 0]
    result[:, 1] = best_centers[:, 1]
    result[:, 2] = radii
    
    # Verify validity and fix if needed
    for i in range(n):
        if result[i, 0] < 0:
            result[i, 0] = 0
        if result[i, 0] > 1:
            result[i, 0] = 1
        if result[i, 1] < 0:
            result[i, 1] = 0
        if result[i, 1] > 1:
            result[i, 1] = 1
        result[i, 2] = min(result[i, 2],
                           min(result[i, 0], 1 - result[i, 0],
                               result[i, 1], 1 - result[i, 1]))
    
    # Final check for overlaps and shrink if necessary
    for _ in range(10):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(result[i, :2] - result[j, :2])
                if d < result[i, 2] + result[j, 2] - 1e-10:
                    overlap_found = True
                    # Shrink both radii slightly
                    shrink = (result[i, 2] + result[j, 2] - d) / 2 + 1e-10
                    result[i, 2] = max(result[i, 2] - shrink, 1e-9)
                    result[j, 2] = max(result[j, 2] - shrink, 1e-9)
        if not overlap_found:
            break
    
    return result
# EVOLVE-BLOCK-END
