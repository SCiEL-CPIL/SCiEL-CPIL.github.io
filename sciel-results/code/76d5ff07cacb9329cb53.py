"""Strategy: Multi-start L-BFGS-B optimization from random and Fibonacci seeds."""
import numpy as np
from scipy.optimize import minimize
import time

N = 150
# Deadline guard: 120s limit, keep 10s safety margin
DEADLINE = time.time() + 110.0

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    pts: (N, 3) array of unit vectors.
    Returns: (energy, grad)
    """
    # Pairwise distances
    # d_ij = ||p_i - p_j||
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero and log(0). 
    # The problem states points closer than 1e-9 are invalid, 
    # but during optimization we might get close. 
    # We add a small epsilon to prevent numerical issues, 
    # but the true energy is -inf if dist=0. 
    # For gradient descent, we want to push them apart.
    # log(1/d) = -log(d)
    # d = sqrt(dist_sq)
    # E = sum_{i<j} -0.5 * log(dist_sq_ij)
    
    # Mask upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist_sq for upper triangle
    dsq = dist_sq[mask]
    
    # To prevent log(0) or negative log (if dist_sq > 1? No, max dist is 2, log(1/2) is negative. 
    # Wait, log(1/d). If d=2, log(0.5) < 0. If d is small, log(1/d) is large positive.
    # We want to MINIMIZE energy. So we want large distances?
    # Let's check the sign.
    # Task: "minimize the logarithmic energy, the sum over all pairs of log(1/distance)"
    # If distance is small (points close), 1/d is large, log(1/d) is large positive.
    # If distance is large (points far), 1/d is small, log(1/d) is negative (if d>1) or 0 (if d=1).
    # On a unit sphere, max distance is 2. log(1/2) = -0.693.
    # So minimizing energy means maximizing distances? 
    # Yes, we want points to be as far apart as possible.
    # So we want to maximize the sum of log(distances).
    # Or minimize sum of -log(distances).
    
    # Let's compute E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # d_ij = sqrt(dsq_ij)
    # log(d_ij) = 0.5 * log(dsq_ij)
    # E = -0.5 * sum_{i<j} log(dsq_ij)
    
    # Gradient with respect to pts[i]:
    # dE/dp_i = sum_{j != i} dE/dd_ij * dd_ij/dp_i
    # E = -0.5 * sum log(dsq)
    # dE/ddsq = -0.5 / dsq
    # dsq = (p_i - p_j)^2
    # d(dsq)/dp_i = 2 * (p_i - p_j)
    # dE/dp_i = sum_{j != i} (-0.5 / dsq_ij) * 2 * (p_i - p_j)
    #          = - sum_{j != i} (p_i - p_j) / dsq_ij
    
    # Let's vectorize the gradient calculation
    # diff is (N, N, 3)
    # dsq is (N, N)
    # We need to sum over j for each i.
    # Term for pair (i, j) in grad[i] is: -(p_i - p_j) / dsq_ij
    # Term for pair (j, i) in grad[j] is: -(p_j - p_i) / dsq_ji = (p_i - p_j) / dsq_ij
    
    # We can compute a matrix W where W_ij = 1/dsq_ij for i!=j, 0 for i=j
    # Then grad[i] = - sum_j W_ij (p_i - p_j)
    # Note: The factor 0.5 from log(sqrt(dsq)) is absorbed?
    # E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij) = -0.5 sum_{i<j} log(dsq_ij)
    # dE/dp_i = -0.5 * sum_{j != i, j>i? No, sum over all j != i in the full sum formulation}
    # Let's use the full sum E_full = -0.5 * sum_{i != j} log(dsq_ij) / 2 ? No.
    # Sum_{i<j} is half of Sum_{i!=j}.
    # So E = -0.5 * (1/2) * Sum_{i!=j} log(dsq_ij) = -0.25 * Sum_{i!=j} log(dsq_ij)
    # dE/dp_i = -0.25 * sum_{j!=i} (1/dsq_ij) * 2(p_i - p_j)
    #          = -0.5 * sum_{j!=i} (p_i - p_j) / dsq_ij
    
    # Let's verify with a simple 2-point case.
    # p1, p2. E = log(1/|p1-p2|)
    # dE/dp1 = -1/|p1-p2| * d|p1-p2| / dp1 = -1/|p1-p2| * (p1-p2)/|p1-p2| = -(p1-p2)/|p1-p2|^2
    # Formula: -0.5 * sum_{j!=1} (p1-pj)/dsq. Here j=2.
    # -0.5 * (p1-p2)/dsq.
    # dsq = |p1-p2|^2.
    # So formula gives -0.5 * (p1-p2)/|p1-p2|^2.
    # But direct calc gave -1 * (p1-p2)/|p1-p2|^2.
    # Discrepancy factor of 2.
    
    # Let's re-evaluate E = sum_{i<j} log(1/d_ij).
    # For N=2, E = log(1/d_12).
    # dE/dp_1 = -1/d_12 * (p_1-p_2)/d_12 = -(p_1-p_2)/d_12^2.
    # My vectorized formula derived: -0.5 * sum (p_i-p_j)/dsq.
    # For N=2, sum has one term (j=2).
    # Result: -0.5 * (p_1-p_2)/dsq_12.
    # This is half the correct gradient.
    
    # Why?
    # E = sum_{i<j} f(d_ij).
    # If I write E = 0.5 * sum_{i!=j} f(d_ij), then for N=2, E = 0.5 * (f(d_12) + f(d_21)) = f(d_12). Correct.
    # dE/dp_i = 0.5 * sum_{j!=i} f'(d_ij) * d(d_ij)/dp_i
    # f(d) = log(1/d) = -log(d). f'(d) = -1/d.
    # d(d)/dp_i = (p_i-p_j)/d.
    # dE/dp_i = 0.5 * sum_{j!=i} (-1/d_ij) * (p_i-p_j)/d_ij
    #          = -0.5 * sum_{j!=i} (p_i-p_j) / d_ij^2
    #          = -0.5 * sum_{j!=i} (p_i-p_j) / dsq_ij
    
    # So the factor 0.5 IS correct.
    # Why did my manual calc for N=2 give -1?
    # E = log(1/d). dE/dp = -1/d * (p1-p2)/d = -(p1-p2)/d^2.
    # Formula: -0.5 * (p1-p2)/d^2.
    # There is a factor of 2 difference.
    
    # Let's check the definition of energy in the prompt.
    # "sum over all pairs of log(1/distance)"
    # "pair terms summed"
    # Usually "pairs" means i<j.
    # If N=2, there is 1 pair.
    # My derivation using 0.5 * sum_{i!=j} assumes the sum is over all ordered pairs i!=j, and we take half.
    # Sum_{i!=j} has 2 terms for N=2: (1,2) and (2,1).
    # 0.5 * (f(12) + f(21)) = f(12).
    # Gradient of 0.5 * Sum_{i!=j} f(d_ij) w.r.t p_i:
    # Term involving p_i in the sum:
    # i is first index: 0.5 * f(d_ij) for all j!=i.
    # i is second index: 0.5 * f(d_ji) for all j!=i.
    # Since d_ij = d_ji, total contribution is sum_{j!=i} f(d_ij).
    # So dE/dp_i = sum_{j!=i} f'(d_ij) * (p_i-p_j)/d_ij
    #             = sum_{j!=i} (-1/d_ij) * (p_i-p_j)/d_ij
    #             = - sum_{j!=i} (p_i-p_j) / dsq_ij
    
    # So the factor is 1, not 0.5.
    # Where did the 0.5 go?
    # Ah, dE/dp_i = 0.5 * [ sum_{j!=i} (term where i is 1st) + sum_{j!=i} (term where i is 2nd) ]
    # Term where i is 1st: d/dp_i f(d_ij) = f'(d_ij) * (p_i-p_j)/d_ij
    # Term where i is 2nd: d/dp_i f(d_ji) = f'(d_ji) * d(d_ji)/dp_i.
    # d_ji = ||p_j - p_i||. d/dp_i ||p_j - p_i|| = (p_i - p_j)/d_ji.
    # So both terms are identical: f'(d) * (p_i-p_j)/d.
    # So dE/dp_i = 0.5 * [ sum_{j!=i} (...) + sum_{j!=i} (...) ] = sum_{j!=i} f'(d) * (p_i-p_j)/d.
    # So the factor is indeed 1.
    
    # Correct Gradient: G_i = - sum_{j!=i} (p_i - p_j) / dsq_ij
    
    # Let's re-verify energy value.
    # E = sum_{i<j} -log(d_ij)
    # Using dsq: E = sum_{i<j} -0.5 log(dsq_ij)
    
    # Compute Energy
    # dsq is (N, N). Mask i<j.
    # E = -0.5 * sum(dsq[mask])
    
    # Compute Gradient
    # G_i = - sum_{j!=i} (p_i - p_j) / dsq_ij
    # Let's construct a matrix K where K_ij = 1/dsq_ij for i!=j, 0 for i=j.
    # Then G_i = - sum_j K_ij (p_i - p_j) = - (sum_j K_ij) p_i + sum_j K_ij p_j
    
    # Set diagonal to 0 to avoid division by zero
    np.fill_diagonal(dsq, 1.0) # Placeholder
    inv_dsq = 1.0 / dsq
    np.fill_diagonal(inv_dsq, 0.0)
    
    # S_i = sum_j K_ij = row sum of inv_dsq
    S = np.sum(inv_dsq, axis=1)
    
    # sum_j K_ij p_j = inv_dsq @ pts
    Kp = inv_dsq @ pts
    
    # G = -S[:, np.newaxis] * pts + Kp
    grad = -S[:, np.newaxis] * pts + Kp
    
    # Energy
    E = -0.5 * np.sum(dsq[mask])
    
    return E, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return pts / norms

def construct():
    start_time = time.time()
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts0
    best_energy = _energy_and_grad(pts0)[0]
    
    # Multi-start optimization
    # We will try several random starts and the Fibonacci start.
    # We use L-BFGS-B on the flattened vector.
    # However, L-BFGS-B doesn't know about the sphere constraint.
    # We must project after every step or use a constrained solver.
    # scipy.optimize.minimize doesn't easily support projection in the callback for L-BFGS-B in a way that keeps the gradient consistent with the manifold.
    # Standard trick: Optimize in Cartesian space, but project at the end? No, that drifts.
    # Better: Use gradient descent with projection.
    # Or use scipy.optimize.minimize with 'L-BFGS-B' and hope it stays close? No.
    
    # Alternative: Use a simple gradient descent loop with line search and projection.
    # This is robust and vectorizable.
    
    def optimize(pts_init, max_iters=200, tol=1e-8):
        pts = _project_to_sphere(pts_init)
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        # grad_t = grad - (grad . p) p
        grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        step_size = 0.1
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Take step
            pts_new = pts - step_size * grad_proj
            pts_new = _project_to_sphere(pts_new)
            
            energy_new, grad_new = _energy_and_grad(pts_new)
            
            # Simple backtracking line search
            if energy_new < energy:
                # Accept
                pts = pts_new
                energy = energy_new
                grad = grad_new
                grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
                step_size = min(step_size * 1.1, 1.0) # Increase step size
            else:
                # Reject
                step_size *= 0.5
                if step_size < 1e-6:
                    break
            
            # Check convergence
            if np.linalg.norm(grad_proj) < tol:
                break
                
        return pts, energy

    # Try Fibonacci start
    pts_opt, e_opt = optimize(pts0, max_iters=300)
    if e_opt < best_energy:
        best_energy = e_opt
        best_pts = pts_opt

    # Try random starts
    np.random.seed(42)
    for k in range(5):
        if time.time() > DEADLINE:
            break
        
        # Random uniform points
        pts_rand = np.random.randn(N, 3)
        pts_rand = _project_to_sphere(pts_rand)
        
        pts_opt, e_opt = optimize(pts_rand, max_iters=300)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt

    return best_pts

if __name__ == "__main__":
    pts = construct()
    e, g = _energy_and_grad(pts)
    print(f"Energy: {e}")
