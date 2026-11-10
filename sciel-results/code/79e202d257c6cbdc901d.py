import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _solve_packing(initial_centers, deadline):
    """
    Solve the packing problem using SLSQP.
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Objective: minimize -r
    Constraints:
    1. c_i inside boundary: r - (1 - |c_i|) <= 0  =>  |c_i| + r <= 1
    2. c_i, c_j non-overlapping: 2r - |c_i - c_j| <= 0 => |c_i - c_j| - 2r >= 0
    """
    # Initial guess for r
    r_init = _compute_radius(initial_centers) * 0.9
    if r_init <= 0:
        r_init = 0.05
        
    x0 = np.concatenate([initial_centers.flatten(), [r_init]])
    
    # Number of variables
    n_vars = 2 * N + 1
    
    # Objective function
    def objective(x):
        return -x[-1]
    
    # Gradient of objective
    def gradient(x):
        grad = np.zeros(n_vars)
        grad[-1] = -1.0
        return grad
        
    # Constraints
    # We use inequality constraints in the form g(x) >= 0 for scipy? 
    # No, scipy SLSQP uses 'ineq' as g(x) >= 0.
    # So for |c_i| + r <= 1, we need 1 - |c_i| - r >= 0.
    # For |c_i - c_j| - 2r >= 0.
    
    def boundary_constraints(x):
        # x[0:2N] are centers, x[2N] is r
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        dists = np.hypot(centers[:, 0], centers[:, 1])
        # 1 - dist - r >= 0
        return 1.0 - dists - r
        
    def boundary_jacobian(x):
        # Derivative of (1 - sqrt(cx^2+cy^2) - r) wrt x
        # d/dcx = -cx / dist
        # d/dcy = -cy / dist
        # d/dr = -1
        centers = x[:2*N].reshape(N, 2)
        dists = np.hypot(centers[:, 0], centers[:, 1])
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        
        jac = np.zeros((N, n_vars))
        # Columns 0 to 2N-1 are centers
        jac[:, 0::2] = -centers[:, 0] / dists_safe
        jac[:, 1::2] = -centers[:, 1] / dists_safe
        # Last column is r
        jac[:, -1] = -1.0
        return jac

    def pair_constraints(x):
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        # We need all pairs i < j
        # Number of pairs = N*(N-1)/2
        # We can compute this efficiently or just loop for simplicity in definition
        # But SLSQP calls this many times, so vectorization is better.
        # However, constructing the full NxN matrix every time might be slow.
        # Let's try a vectorized approach using triu indices.
        
        # For simplicity and robustness in definition, let's use a loop if N is small, 
        # but N=50 is small enough that even a loop might be okay if optimized, 
        # but vectorized is better.
        
        # Let's precompute indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        ci = centers[i_idx]
        cj = centers[j_idx]
        
        dx = ci[:, 0] - cj[:, 0]
        dy = ci[:, 1] - cj[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # |ci - cj| - 2r >= 0
        return dists - 2.0 * r

    def pair_jacobian(x):
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        ci = centers[i_idx]
        cj = centers[j_idx]
        
        dx = ci[:, 0] - cj[:, 0]
        dy = ci[:, 1] - cj[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        
        num_pairs = len(i_idx)
        jac = np.zeros((num_pairs, n_vars))
        
        # Gradient wrt ci
        # d/dci_x (dist - 2r) = (ci_x - cj_x) / dist
        # d/dci_y (dist - 2r) = (ci_y - cj_y) / dist
        jac[:, i_idx * 2] = dx / dists_safe
        jac[:, i_idx * 2 + 1] = dy / dists_safe
        
        # Gradient wrt cj
        # d/dcj_x (dist - 2r) = -(ci_x - cj_x) / dist
        # d/dcj_y (dist - 2r) = -(ci_y - cj_y) / dist
        jac[:, j_idx * 2] = -dx / dists_safe
        jac[:, j_idx * 2 + 1] = -dy / dists_safe
        
        # Gradient wrt r
        jac[:, -1] = -2.0
        
        return jac

    # Bounds for variables
    # Centers must be within [-1, 1] roughly, but constraint handles it.
    # Let's set loose bounds to help solver.
    bounds = [(-1.0, 1.0)] * (2 * N)
    bounds.append((0.0, 1.0)) # r is positive and < 1

    # Define constraint structures
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints, 'jac': boundary_jacobian},
        {'type': 'ineq', 'fun': pair_constraints, 'jac': pair_jacobian}
    ]

    try:
        # Use SLSQP which handles gradients and constraints well
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success and res.x is not None:
            # Extract centers
            centers = res.x[:2*N].reshape(N, 2)
            # Verify validity
            r_val = _compute_radius(centers)
            if r_val > 0:
                return centers, r_val
    except Exception:
        pass
        
    return initial_centers, _compute_radius(initial_centers)

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Run SLSQP optimization
    best_centers, best_r = _solve_packing(centers, deadline)
    
    # If we have time, try a few random restarts or perturbations?
    # SLSQP is local. A single run from a good start is usually strong.
    # Let's check time.
    if time.time() < deadline - 20:
        # Try a second run from a slightly perturbed start?
        # Or just refine the result?
        # Let's try one more run with a different initial r guess?
        # Actually, let's just return the best found.
        # If we want to improve, we could run SLSQP again on the result 
        # but it's already converged.
        
        # Alternative: try a different initial configuration?
        # For N=50, hexagonal is very good.
        
        return best_centers
