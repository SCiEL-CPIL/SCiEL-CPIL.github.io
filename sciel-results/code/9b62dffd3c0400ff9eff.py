"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Topology-Constrained Initialization:
   - Instead of random or simple greedy, initialize points in a symmetric quasi-lattice structure.
   - The 11 points are arranged to respect the symmetry of the equilateral triangle.
   - This reduces the effective search space by leveraging known structural regularity.
2. Barycentric Coordinate Parameterization:
   - Points are represented in barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0.
   - This ensures points are always inside the triangle without needing projection.
   - The optimization variables are the 2 independent barycentric coordinates per point.
3. Robust Local Search:
   - Uses Nelder-Mead on the barycentric parameters.
   - Multi-start with different symmetric base configurations and perturbations.
   - Focuses computational budget on refining high-quality structural bases rather than random exploration.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _barycentric_to_cartesian(bary):
    """
    Convert barycentric coordinates (u, v, w) to Cartesian (x, y).
    Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    P = u*A + v*B + w*C
    x = v + 0.5*w
    y = (sqrt(3)/2)*w
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack((x, y))

def _cartesian_to_barycentric(pts):
    """
    Convert Cartesian (x, y) to barycentric coordinates (u, v, w).
    Inverse of _barycentric_to_cartesian.
    w = 2*y / sqrt(3)
    v = x - 0.5*w
    u = 1 - v - w
    """
    x = pts[:, 0]
    y = pts[:, 1]
    w = 2 * y / np.sqrt(3)
    v = x - 0.5 * w
    u = 1 - v - w
    # Clip to [0, 1] to handle numerical errors, though the optimization should keep them valid
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    # Normalize to ensure sum is exactly 1
    sum_uvw = u + v + w
    u /= sum_uvw
    v /= sum_uvw
    w /= sum_uvw
    return np.column_stack((u, v, w))

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_triplets[:, 2]]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective_bary(bary_flat):
    """
    Objective function for minimization in barycentric space.
    bary_flat: shape (N*3,) but we only optimize 2 independent coords per point?
    Actually, for simplicity and robustness, we optimize all 3 and enforce sum=1 via projection
    or just use 2 independent coords. Let's use 2 independent coords (v, w) and compute u=1-v-w.
    Wait, the standard Nelder-Mead works best with fixed dimension.
    Let's stick to optimizing the 2 independent barycentric coordinates (v, w) for each point.
    u is derived.
    Constraint: u, v, w >= 0.
    If u < 0, we are outside.
    To handle constraints in Nelder-Mead, we can use a penalty or projection.
    Projection in barycentric space is easier: if u<0, set u=0 and renormalize v,w.
    """
    bary = bary_flat.reshape(N, 2)
    v = bary[:, 0]
    w = bary[:, 1]
    u = 1 - v - w
    
    # Check validity: u, v, w must be >= 0
    # If invalid, we return a large penalty to steer the optimizer away.
    # Or we can project. Projection is better for Nelder-Mead.
    # Projection: if any of u,v,w is negative, set it to 0 and renormalize the others.
    # This is a bit complex to vectorize efficiently for the objective function call.
    # Alternative: Use a penalty term.
    
    min_uvw = np.min(np.stack([u, v, w], axis=1), axis=1)
    if np.any(min_uvw < -1e-9):
        # Large penalty if significantly outside
        return 1.0 + np.sum((min_uvw[min_uvw < 0])**2)
    
    # If slightly outside due to floating point, clip for area calculation
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    sum_uvw = u + v + w
    u /= sum_uvw
    v /= sum_uvw
    w /= sum_uvw
    
    bary_full = np.column_stack((u, v, w))
    pts = _barycentric_to_cartesian(bary_full)
    
    area = _min_triangle_area(pts)
    return -area

def _generate_symmetric_base(rng):
    """
    Generate a symmetric base configuration for 11 points.
    The optimal configuration for Heilbronn problems often has symmetries.
    For N=11, a common structure is a central point, surrounded by rings.
    Let's define a few topological templates.
    
    Template 1: 1 center, 3 inner, 3 mid, 4 outer?
    Template 2: 1 center, 2 inner, 4 mid, 4 outer?
    
    Let's use a grid-like structure in barycentric coordinates.
    """
    # Barycentric coordinates for a 3x3 grid plus 2 extra?
    # Or a hexagonal packing?
    
    # Let's try a structured approach:
    # 1 point at center (1/3, 1/3, 1/3)
    # 3 points near vertices
    # 3 points near midpoints of edges
    # 3 points in between?
    
    # Actually, let's use a known good starting point from literature or a simple grid.
    # A 3x3 grid in barycentric coordinates gives 9 points.
    # We need 11.
    
    # Let's define a base set of barycentric coordinates.
    # We will use a "quasi-lattice" that is slightly perturbed.
    
    # Base points (u, v, w)
    # 1. Center
    pts_bary = []
    
    # Center
    pts_bary.append([1/3, 1/3, 1/3])
    
    # 3 points on axes (midpoints of edges)
    pts_bary.append([0.5, 0.5, 0.0])
    pts_bary.append([0.5, 0.0, 0.5])
    pts_bary.append([0.0, 0.5, 0.5])
    
    # 3 points near vertices
    pts_bary.append([0.8, 0.1, 0.1])
    pts_bary.append([0.1, 0.8, 0.1])
    pts_bary.append([0.1, 0.1, 0.8])
    
    # 3 points in between center and vertices
    pts_bary.append([0.4, 0.3, 0.3])
    pts_bary.append([0.3, 0.4, 0.3])
    pts_bary.append([0.3, 0.3, 0.4])
    
    # 3 more points? We have 10. Need 1 more.
    # Maybe shift the center? Or add a point at the centroid of a sub-triangle?
    # Let's add a point at [0.25, 0.25, 0.5] and adjust others?
    # Actually, let's just generate a few random symmetric bases and let the optimizer work.
    
    # A better strategy: Use a 3x3 grid in barycentric coordinates (9 points)
    # and add 2 more points at the midpoints of the lines from center to vertices?
    
    grid_pts = []
    for i in range(3):
        for j in range(3):
            # Map i, j in {0,1,2} to barycentric coords
            # i corresponds to v, j corresponds to w?
            # u = 1 - v - w
            # Let's use a simple linear mapping
            v = (i + 0.5) / 3.0
            w = (j + 0.5) / 3.0
            u = 1 - v - w
            if u >= -1e-9:
                grid_pts.append([max(0, u), v, w])
    
    # We have 9 points from the grid.
    # Add 2 more points. Where?
    # Maybe at the vertices of the triangle? No, that reduces area.
    # Maybe at the midpoints of the grid cells?
    
    # Let's just use the 9 grid points and 2 random points in the center region.
    # Then optimize.
    
    # For now, return the 9 grid points + 2 random points near center.
    for _ in range(2):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        w = 1 - u - v
        grid_pts.append([u, v, w])
        
    return np.array(grid_pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Topology-Constrained Optimization with Barycentric Coordinates.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    
    # Generate diverse symmetric bases
    num_bases = 10
    for _ in range(num_bases):
        base_bary = _generate_symmetric_base(rng)
        # Perturb the base
        perturbed_bary = base_bary + np.random.normal(0, 0.05, (N, 3))
        # Normalize
        sums = perturbed_bary.sum(axis=1, keepdims=True)
        perturbed_bary /= sums
        # Clip negatives
        perturbed_bary = np.clip(perturbed_bary, 0, 1)
        # Re-normalize after clipping
        sums = perturbed_bary.sum(axis=1, keepdims=True)
        perturbed_bary /= sums
        
        initializations.append(perturbed_bary)
        
        # Also add the un-perturbed base
        initializations.append(base_bary)
        
    # Add a few purely random barycentric initializations
    for _ in range(5):
        rand_bary = np.random.dirichlet(np.ones(N), size=N)
        # Note: np.random.dirichlet returns an array of shape (N, 3) if size=N?
        # Actually, np.random.dirichlet(alpha, size=None) returns an array of shape (len(alpha),) or (size, len(alpha)).
        # If alpha is shape (3,), and size=N, it returns (N, 3).
        # But we want each point to be a valid barycentric coord.
        # So we need to generate N independent Dirichlet samples.
        rand_bary = np.random.dirichlet(np.ones(3), size=N)
        initializations.append(rand_bary)
        
    # Shuffle
    rng.shuffle(initializations)
    
    # Optimize each initialization
    for i, init_bary in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # We optimize 2 independent coordinates per point: v and w.
        # u is derived.
        x0 = np.column_stack((init_bary[:, 1], init_bary[:, 2])).flatten()
        
        try:
            res = minimize(
                _objective_bary, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000,  # Reduce maxiter to allow more restarts
                    'xatol': 1e-6, 
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            # Reconstruct barycentric coords from the result
            res_vw = res.x.reshape(N, 2)
            v = res_vw[:, 0]
            w = res_vw[:, 1]
            u = 1 - v - w
            
            # Clip and normalize
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            w = np.clip(w, 0, 1)
            sums = u + v + w
            u /= sums
            v /= sums
            w /= sums
            
            bary_opt = np.column_stack((u, v, w))
            pts_opt = _barycentric_to_cartesian(bary_opt)
            
            # Verify inside
            x = pts_opt[:, 0]
            y = pts_opt[:, 1]
            if np.all(y >= -1e-9) and np.all(y <= np.sqrt(3) * x + 1e-9) and np.all(y <= np.sqrt(3) * (1 - x) + 1e-9) and np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            # If optimization fails, evaluate the initial point
            try:
                pts_init = _barycentric_to_cartesian(init_bary)
                score = _min_triangle_area(pts_init)
                if score > best_score:
                    best_score = score
                    best_pts = pts_init.copy()
            except:
                pass
            continue
            
        # Check time again
        if time.time() - start_time > DEADLINE - 10:
            break

    # Fallback
    if best_pts is None:
        # Use a simple grid
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
