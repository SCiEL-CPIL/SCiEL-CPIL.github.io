"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses a two-stage approach:
    1. Start from a good heuristic initial configuration (hexagonal-like packing
       with boundary adjustments).
    2. Use gradient-based optimization to push circles apart, maximizing the
       minimum distance (which directly gives radius = min_dist / 2, bounded by
       distance to boundary).
    
    The objective is to maximize the minimum of:
      - pairwise distances / 2
      - distance from each center to the nearest boundary (0.5 - |x|, 0.5 - |y|)
    
    We reformulate as: maximize r subject to:
      ||c_i - c_j|| >= 2r  for all i < j
      0.5 - |c_ik| >= r    for all i, k
    
    We use a smooth approximation and iterative refinement.
    """
    
    # Stage 1: Generate a good initial packing using a hexagonal lattice
    # with some randomization and boundary handling.
    
    def hexagonal_packing(n):
        """Generate n points in a hexagonal arrangement within [-0.5, 0.5]^2."""
        # Try different hexagonal spacings and offsets
        best_points = None
        best_score = -1
        
        for attempt in range(20):
            # Random jitter and rotation
            angle = np.random.uniform(0, np.pi / 6)  # small rotation
            dx = np.random.uniform(-0.02, 0.02)
            dy = np.random.uniform(-0.02, 0.02)
            
            # Hexagonal lattice parameters
            # For a hexagonal lattice, the spacing between rows is s * sqrt(3)/2
            # and within a row is s
            # We want to fit ~7 columns and ~8 rows for 50 points
            # Let's try various spacings
            
            # Estimate spacing: area per point ~ 1/50, so spacing ~ sqrt(1/50) ~ 0.14
            s = np.sqrt(1.0 / n) * np.random.uniform(0.95, 1.15)
            
            # Generate hexagonal lattice points
            points = []
            rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3)))) + 2
            for row in range(rows):
                row_y = -0.5 + s * np.sqrt(3) / 2 * (row + 0.5) + dy
                if row_y > 0.5:
                    break
                offset = (s / 2) * (row % 2) + dx
                cols = int(np.ceil((1.0 - 2 * s) / s)) + 2
                for col in range(cols):
                    row_x = -0.5 + s * (col + 0.5) + offset
                    if row_x > 0.5:
                        continue
                    points.append([row_x, row_y])
            
            points = np.array(points)
            
            # Rotate around center
            if abs(angle) > 1e-10:
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                points = points @ rot.T
            
            # Clip to square (keep only points inside)
            mask = (np.abs(points[:, 0]) <= 0.5) & (np.abs(points[:, 1]) <= 0.5)
            points = points[mask]
            
            if len(points) < n:
                continue
            
            # Take the first n points (or best n by some criterion)
            points = points[:n]
            
            # Compute score: minimum distance
            from scipy.spatial.distance import pdist
            min_dist = pdist(points)
            min_dist = min_dist.min()
            
            # Also check boundary distance
            boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
            min_boundary = boundary_dists.min()
            
            score = min(min_dist / 2, min_boundary)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        if best_points is None:
            # Fallback: grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        return best_points
    
    # Generate initial configuration
    np.random.seed(42)
    points = hexagonal_packing(N)
    
    # Stage 2: Optimize using a repulsive force model
    # We want to maximize the minimum distance, which is equivalent to
    # minimizing the maximum "violation" or using a smooth penalty.
    
    def objective_and_grad(x):
        """
        Objective: minimize a smooth approximation of -min_distance.
        We use: f(x) = -min_i j (||x_i - x_j||) for i<j, also considering boundaries.
        
        Better: minimize sum of soft constraints that penalize when distances < 2r.
        Actually, let's maximize r by minimizing a barrier function.
        
        Let's define: for each pair (i,j), d_ij = ||x_i - x_j||
        For each point i, b_i = min(0.5 - |x_ik|) for k in {0,1}
        
        We want to maximize r = min(min_ij d_ij/2, min_i b_i)
        
        Equivalently, minimize -r.
        
        Use a smooth min: soft_min(z) = -log(sum(exp(-z_i / t))) * t
        """
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(x)
        
        # Boundary distances
        b = np.minimum(0.5 - np.abs(x[:, 0]), 0.5 - np.abs(x[:, 1]))
        
        # All "constrained" distances
        all_dists = np.concatenate([d, b])
        
        # Smooth min with temperature
        t = 0.01
        z = all_dists / t
        # Handle numerical stability
        z_max = z.max()
        log_sum = z_max + np.log(np.sum(np.exp(z - z_max)))
        soft_min_val = -t * log_sum
        
        # Gradient of soft_min w.r.t. all_dists
        # d(soft_min)/d(all_dists_i) = exp(z_i) / sum(exp(z_j))
        exp_z = np.exp(z - z_max)
        probs = exp_z / exp_z.sum()
        
        # Now backpropagate to x
        n_pairs = len(d)
        n_points = len(x)
        
        grad = np.zeros_like(x)
        
        # Gradient from pairwise distances
        for idx in range(n_pairs):
            # Find which pair this is
            # pdist indexing: for n points, pair (i,j) with i<j
            # We need to map linear index to (i,j)
            # This is expensive; let's compute gradients differently.
            pass
        
        # Better approach: compute gradient analytically
        # For each pair (i, j) with i < j:
        #   d/dx_i d_ij = -(x_i - x_j) / d_ij
        #   d/dx_j d_ij = (x_i - x_j) / d_ij
        
        # Let's recompute
        grad = np.zeros_like(x)
        n = len(x)
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = x[i] - x[j]
                d_ij = np.linalg.norm(diff)
                if d_ij < 1e-10:
                    d_ij = 1e-10
                w = probs[pair_idx] / t
                grad[i] += w * (-diff / d_ij)
                grad[j] += w * (diff / d_ij)
                pair_idx += 1
        
        # Boundary distances
        for i in range(n):
            # b_i = min(0.5 - |x_i0|, 0.5 - |x_i1|)
            # Smooth min of two values
            bx = 0.5 - np.abs(x[i, 0])
            by = 0.5 - np.abs(x[i, 1])
            
            # Gradient of soft min of bx, by
            t_b = 0.01
            zb = np.array([bx, by]) / t_b
            zb_max = zb.max()
            exp_zb = np.exp(zb - zb_max)
            probs_b = exp_zb / exp_zb.sum()
            
            # d(bx)/dx_i0 = -sign(x_i0)
            # d(by)/dx_i1 = -sign(x_i1)
            dx0 = -np.sign(x[i, 0]) if abs(x[i, 0]) > 1e-10 else 0
            dx1 = -np.sign(x[i, 1]) if abs(x[i, 1]) > 1e-10 else 0
            
            w0 = probs_b[0] / t_b
            w1 = probs_b[1] / t_b
            
            grad[i, 0] += w0 * dx0
            grad[i, 1] += w1 * dx1
        
        # We want to maximize soft_min, so minimize -soft_min
        val = -soft_min_val
        grad = -grad
        
        return val, grad
    
    # Use L-BFGS-B for optimization
    x0 = points.flatten()
    
    # Define objective without gradient for safety, but provide gradient
    def obj(x):
        val, _ = objective_and_grad(x)
        return val
    
    # Optimize
    result = minimize(obj, x0, method='L-BFGS-B', jac=None, 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    points = result.x.reshape(-1, 2)
    
    # Ensure points are strictly inside the square
    eps = 1e-9
    points[:, 0] = np.clip(points[:, 0], -0.5 + eps, 0.5 - eps)
    points[:, 1] = np.clip(points[:, 1], -0.5 + eps, 0.5 - eps)
    
    # Final local refinement: pairwise repulsion
    # Do a few iterations of pushing overlapping circles apart
    for it in range(100):
        moved = False
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(points))
        np.fill_diagonal(D, np.inf)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = D[i, j]
                if d < 0.07:  # if too close
                    diff = points[i] - points[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1e-10, 0])
                        norm = 1e-10
                    # Push apart
                    push = 0.001 * diff / norm
                    points[i] += push
                    points[j] -= push
                    moved = True
        
        # Keep inside square
        points[:, 0] = np.clip(points[:, 0], -0.5 + eps, 0.5 - eps)
        points[:, 1] = np.clip(points[:, 1], -0.5 + eps, 0.5 - eps)
        
        if not moved:
            break
    
    return points

# EVOLVE-BLOCK-END
