"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair / 2
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Min distance to boundary
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)

def _init_hex(n_points):
    """Generate initial hexagonal packing."""
    # Estimate spacing
    # Area of unit circle is pi. Area per circle is pi*r^2.
    # r ~ sqrt(pi / (n * sqrt(3)/2)) ?
    # Hex density is pi / (2*sqrt(3)) ~ 0.9069
    # r ~ sqrt(1 / (n * (2/sqrt(3)))) ?
    # Let's just try a few spacings and pick the best one that fits N points.
    
    best_pts = None
    best_r = 0.0
    
    # Spacing s. Number of points approx 2 * (1/s)^2 * sqrt(3)/2
    # s ~ sqrt(3 / n)
    base_s = np.sqrt(3.0 / n_points)
    
    for mult in np.linspace(0.9, 1.2, 20):
        s = base_s * mult
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - s / 2.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) < n_points:
            continue
        
        # Sort by distance from center to pick the most central ones?
        # Or just random sample? Central is usually better for symmetry.
        dists_to_origin = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists_to_origin, kind='stable')[:n_points]
        candidate = points[idx]
        
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
            
    if best_pts is None:
        # Fallback: random points
        best_pts = np.random.uniform(-0.9, 0.9, (n_points, 2))
        
    return best_pts

def _soft_min_objective(x, temp):
    """
    Objective function for L-BFGS-B.
    Minimizes -soft_min_radius.
    x is flattened (N, 2) points.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    # Using cdist is efficient
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Soft min of pairwise distances / 2
    # soft_min(A) = -T * log(sum(exp(-A/T)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # min(A) - T * log(sum(exp(-(A - min(A))/T)))
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    b = 1.0 - radii
    
    # Combine pairwise and boundary constraints into a single soft-min?
    # Or minimize -min(soft_min_pair/2, soft_min_bound)
    # It's better to treat them separately or combine carefully.
    # Let's compute soft_min_pair and soft_min_bound separately.
    
    min_pair_raw = np.min(d)
    if min_pair_raw < 1e-10:
        min_pair_raw = 1e-10 # Prevent log(0)
        
    # Shift by min for stability
    # We want soft_min of d.
    # d_shifted = d - min_pair_raw
    # log_sum = log(sum(exp(-d_shifted / T)))
    # soft_min_pair = min_pair_raw - T * log_sum
    
    # However, d is 300x300. Exp might be slow.
    # For T=0.01, exp(-10/0.01) = exp(-1000) = 0.
    # Only terms close to min matter.
    
    # Optimization: Only consider pairs with d < min_pair_raw + 5*T
    # This creates a sparse matrix effectively.
    
    # For simplicity and speed in this context, let's just use the full vector but be careful.
    # Actually, 300x300 is 90k elements. Exp is fast.
    
    # Pairwise term
    # We want to maximize min(d_ij)/2.
    # Let's define loss_pair = -soft_min(d/2)
    d_half = d / 2.0
    min_d_half = np.min(d_half)
    # Mask diagonal and very large distances to avoid numerical issues? 
    # No, exp(-large) is 0.
    # But exp(-min) is 1.
    # sum(exp(-(d_half - min_d_half)/T))
    # Note: d_half has inf on diagonal. exp(-inf) = 0. Good.
    
    log_sum_pair = np.log(np.sum(np.exp(-(d_half - min_d_half) / temp)))
    soft_min_pair = min_d_half - temp * log_sum_pair
    
    # Boundary term
    min_b = np.min(b)
    # b can be negative if outside. We want to penalize that.
    # If b < 0, soft_min will be negative.
    # We want to maximize min(b).
    # So minimize -soft_min(b).
    
    # Handle case where all b are very negative?
    # Shift by min_b.
    log_sum_bound = np.log(np.sum(np.exp(-(b - min_b) / temp)))
    soft_min_bound = min_b - temp * log_sum_bound
    
    # The radius is min(soft_min_pair, soft_min_bound)
    # We want to maximize this.
    # So we minimize -min(soft_min_pair, soft_min_bound).
    # To make it smooth, we can use soft-min of the two values.
    # r = soft_min(soft_min_pair, soft_min_bound)
    # This is getting complex.
    
    # Simpler: Just minimize the negative of the minimum of the two soft-mins.
    # min_val = min(soft_min_pair, soft_min_bound)
    # return -min_val
    
    # This is non-smooth at the crossover.
    # Better: Use a soft-min of the two.
    # r_soft = -T * log(exp(-soft_min_pair/T) + exp(-soft_min_bound/T))
    # This approximates min(soft_min_pair, soft_min_bound).
    
    # Let's just return -min(soft_min_pair, soft_min_bound) for now, L-BFGS-B can handle mild non-smoothness often, 
    # or we can just optimize one and then the other?
    # No, let's use the soft-min of the two to be safe.
    
    # To avoid underflow in exp(-val/T), shift by min of the two.
    min_two = min(soft_min_pair, soft_min_bound)
    log_sum_two = np.log(np.exp(-(soft_min_pair - min_two) / temp) + np.exp(-(soft_min_bound - min_two) / temp))
    r_final = min_two - temp * log_sum_two
    
    # Penalty for being far outside?
    # If points are far outside, b is very negative, soft_min_bound is very negative, r_final is very negative.
    # Minimizing -r_final means maximizing r_final, so it pushes them in.
    
    return -r_final

def _newton_polish(pts, max_iter=50):
    """
    Active-set Newton method to maximize the minimum distance.
    This solves the system for the binding constraints.
    """
    n = len(pts)
    pts = pts.copy()
    
    for _ in range(max_iter):
        # Compute all pairwise distances and boundary distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        b = 1.0 - radii
        
        # Current radius
        r_pair = np.min(d) / 2.0
        r_bound = np.min(b)
        r = min(r_pair, r_bound)
        
        # Identify active constraints
        # Tolerance
        tol = 1e-8
        active_pairs = []
        active_bounds = []
        
        # Find pairs with distance close to 2*r
        # d_ij <= 2*r + tol
        # Since r is min(d)/2, 2*r is the min distance.
        # So we look for d_ij <= 2*r + tol
        
        # Efficiently find these indices
        # d is n x n
        # We need i < j
        i_idx, j_idx = np.where(d <= 2*r + tol)
        # Filter i < j
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        # Find bounds with b <= r + tol
        b_idx = np.where(b <= r + tol)[0]
        
        if len(i_idx) == 0 and len(b_idx) == 0:
            # No active constraints? Should not happen if r is min.
            break
            
        # Construct Jacobian J (m x 2n) and gradient g (2n)
        # We want to solve J * delta_x = 1 (vector of ones)
        # where delta_x is the step for 2n coordinates.
        # And we want to maximize r, so we move in the direction that increases r.
        # dr = J * delta_x
        # We want dr = 1 (or some positive value) to increase r by 1?
        # Actually, we want to find the direction that increases r the most per unit step?
        # Standard active set method for maximizing min constraints:
        # Maximize r
        # Subject to d_ij >= 2r, b_i >= r
        # Linearize: d_ij + grad_d_ij . delta_x >= 2(r + delta_r)
        # b_i + grad_b_i . delta_x >= r + delta_r
        # grad_d_ij . delta_x - 2 delta_r >= -d_ij + 2r
        # grad_b_i . delta_x - delta_r >= -b_i + r
        
        # Let's set delta_r = 1 (we want to increase r by 1 unit, scaled by step size later?)
        # Or better, solve for delta_x such that the constraints are satisfied with equality for active set,
        # and we maximize the step size?
        
        # Simpler approach:
        # Solve J * delta_x = -1 (where J rows are gradients of constraints w.r.t x)
        # Then move x = x + alpha * delta_x
        # Increase r by alpha?
        
        # Let's construct the system:
        # For each active pair (i, j):
        # u = (x_i - x_j) / ||x_i - x_j||
        # Constraint: (x_i - x_j) . delta_x >= 2 * delta_r
        # Row in J: u at i, -u at j. RHS: 2. (assuming delta_r=1)
        
        # For each active bound i:
        # v = x_i / ||x_i|| (if x_i != 0)
        # Constraint: x_i . delta_x >= delta_r
        # Row in J: v at i. RHS: 1.
        
        m = len(i_idx) + len(b_idx)
        if m == 0:
             break
             
        J = np.zeros((m, 2*n))
        rhs = np.zeros(m)
        
        row = 0
        for i, j in zip(i_idx, j_idx):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Degenerate, skip or handle
                continue
            u = diff / dist
            J[row, 2*i:2*i+2] = u
            J[row, 2*j:2*j+2] = -u
            rhs[row] = 2.0
            row += 1
            
        for i in b_idx:
            norm = np.linalg.norm(pts[i])
            if norm < 1e-10:
                # At center, any direction works? 
                # Constraint: x_i . delta_x >= 1
                # If x_i=0, gradient is undefined.
                # We can pick an arbitrary direction, e.g., (1,0)
                v = np.array([1.0, 0.0])
            else:
                v = pts[i] / norm
            J[row, 2*i:2*i+2] = v
            rhs[row] = 1.0
            row += 1
            
        # We have J (row x 2n) and rhs (row)
        # We want to solve J * delta_x = rhs
        # If row < 2n, underdetermined. We want the minimum norm solution.
        # delta_x = J.T * inv(J * J.T) * rhs
        # Or use lstsq.
        
        try:
            # Solve least squares
            delta_x, residuals, rank, s = np.linalg.lstsq(J, rhs, rcond=None)
        except:
            break
            
        # Check if this step actually increases r
        # Compute new r
        new_pts = pts + delta_x
        new_r = _compute_radius(new_pts)
        
        if new_r > r + 1e-12:
            pts = new_pts
        else:
            # Step was not good, break
            break
            
    return pts

def construct():
    deadline = time.time() + 115.0
    best_pts = _init_hex(N)
    best_r = _compute_radius(best_pts)
    
    # Optimization loop with annealing
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    current_pts = best_pts.copy()
    
    for temp in temps:
        if time.time() > deadline - 5.0:
            break
            
        x0 = current_pts.flatten()
        
        # Define objective
        def objective(x):
            return _soft_min_objective(x, temp)
            
        # Bounds
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
            else:
                # If no improvement, maybe perturb and restart?
                # Or just move to next temp
                pass
                
        except Exception:
            pass
            
        # Newton Polish
        if time.time() > deadline - 2.0:
            break
            
        polished_pts = _newton_polish(current_pts, max_iter=100)
        polished_r = _compute_radius(polished_pts)
        
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            current_pts = polished_pts.copy()
            
    # Final check
    if best_pts.shape != (N, 2):
        best_pts = np.zeros((N, 2))
        
    return best_pts
# EVOLVE-BLOCK-END
