import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    # Get upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow/underflow, use log-sum-exp trick
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init():
    """
    Build a Body-Centered Cubic (BCC) lattice initialization.
    BCC is known to have higher density than simple cubic.
    We need 100 points.
    A 4x4x4 simple cubic grid has 64 points.
    Adding centers to each cube in a 4x4x4 grid would be 64 * 2 = 128 points.
    We can take a subset or adjust.
    Let's try a 4x5x5 grid of unit cells? No, that's 100 cells, 200 points.
    
    Alternative: Start with a 4x5x5 simple cubic grid (100 points) but perturb towards BCC locally?
    Or just use a denser random start?
    
    Actually, a good heuristic is to start with a grid that is slightly "compressed" 
    or use a Halton sequence for better space-filling properties than a grid.
    Let's implement a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Scale to [-0.5, 0.5]
        # Halton is in [0, 1)
        # Add small margin to avoid boundary issues initially, though optimizer handles it
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    return np.array(points)

def _build_grid_init():
    """
    Build a grid initialization.
    5x5x4 grid.
    """
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return centers

def _optimize_from_initial(initial_centers, deadline):
    """
    Run the optimization loop starting from given initial centers.
    Returns best centers and best radius.
    """
    best_centers = initial_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = 10.0
    max_beta = 5000.0
    
    # Schedule of betas
    # We will increase beta gradually
    while time.time() < deadline and beta < max_beta:
        x0 = best_centers.flatten()
        
        try:
            # Use L-BFGS-B
            # Tighter tolerances in later stages?
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < _soft_min_objective(x0, beta):
                new_centers = res.x.reshape(-1, 3)
                # Clip to valid range
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
            else:
                # If no improvement, maybe try a smaller step or break?
                # For now, just increase beta
                pass
        except Exception:
            pass
        
        # Increase beta
        # Geometric progression
        beta *= 1.3
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish pass.
    Use a very high beta and tighter tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few very high beta values
    betas = [10000, 20000, 50000]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < _soft_min_objective(x0, beta):
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Safety margin: 115s out of 120s limit
    deadline = start_time + 115.0
    
    # Try multiple initializations
    # 1. Grid
    # 2. Halton
    # 3. Random (seeded for reproducibility? No, random is fine, but let's use a fixed seed for consistency in testing, or just default)
    # Let's try Grid and Halton.
    
    inits = []
    inits.append(_build_grid_init())
    inits.append(_build_bcc_init())
    
    # Add a random initialization with small noise
    np.random.seed(42)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_init)
    
    best_overall_centers = None
    best_overall_radius = -1.0
    
    # Allocate time to each init
    # We have ~115s.
    # Let's give each init a portion, but allow the best one to use remaining time if it's leading?
    # Simpler: divide time roughly equally, but stop early if time runs out.
    # With 3 inits, ~38s each.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Reserve 10s for final polish on the best result
        # So each init gets (deadline - current_time - 10s) / remaining_inits
        # But simpler: just run until deadline - 10s for the last init?
        # Let's just run each init with a sub-deadline.
        # If we have 3 inits, give each 1/3 of the time, minus a buffer.
        
        # Dynamic time allocation:
        # Time left = deadline - now
        # Inits left = len(inits) - current_index
        # Time for this init = time_left * (1.0 / inits_left) * 0.9 # 10% buffer
        
        # For simplicity, let's just run the first two with ~40s each, and the last one with remaining time minus 10s for polish.
        # Actually, let's just loop and break if time is up.
        
        # Sub-deadline for this init
        # We want to leave 10s for polish.
        # So we can use up to deadline - 10s.
        # But we should distribute among inits.
        
        # Let's use a fixed time per init for the first two, and the rest for the last.
        # Or just let them run until a global sub-deadline.
        
        # Let's try:
        # Init 1: 40s
        # Init 2: 40s
        # Init 3: remaining - 10s
        
        # Since we don't know how many inits we have, let's just calculate:
        # total_time = deadline - start_time
        # time_per_init = total_time / len(inits)
        # But we need to leave 10s for polish.
        # So effective_time = total_time - 10
        # time_per_init = effective_time / len(inits)
        
        # This is getting complicated. Let's just run each init with a generous timeout and rely on the global deadline check inside _optimize_from_initial.
        # But _optimize_from_initial needs a deadline.
        
        # Let's pass a sub-deadline.
        # If we have 3 inits, and 115s total.
        # Leave 15s for polish.
        # 100s for optimization.
        # ~33s per init.
        
        sub_deadline = start_time + 35.0 # for first init
        # This is hardcoded. Let's make it dynamic.
        
        # Let's just use the global deadline for the first init, but that's bad if it takes all time.
        # Let's use a fixed 35s for each of the first 2 inits, and the rest for the 3rd.
        
        # Actually, the grid init is usually good. Halton might be better.
        # Let's just run them sequentially with a shared deadline, but ensure we don't overshoot.
        
        # Simplest robust approach:
        # Run init 1 with deadline = start + 35
        # Run init 2 with deadline = start + 70
        # Run init 3 with deadline = deadline_global - 10
        
        # Let's implement this logic.
        
        pass # Placeholder for logic
    
    # Let's re-implement the loop properly
    
    best_overall_centers = None
    best_overall_radius = -1.0
    
    # We will run each init with a specific time budget.
    # Total budget 115s.
    # Polish budget 15s.
    # Optimization budget 100s.
    # 3 inits -> ~33s each.
    
    opt_deadlines = [
        start_time + 33.0,
        start_time + 66.0,
        deadline - 15.0 # Last init gets remaining time
    ]
    
    for i, init in enumerate(inits):
        if i >= len(opt_deadlines):
            break
            
        current_deadline = opt_deadlines[i]
        
        # If we are already past the deadline, skip
        if time.time() > current_deadline:
            continue
            
        centers, radius = _optimize_from_initial(init, current_deadline)
        
        if radius > best_overall_radius:
            best_overall_radius = radius
            best_overall_centers = centers
    
    # Final polish
    if best_overall_centers is not None:
        best_overall_centers, best_overall_radius = _polish(best_overall_centers, deadline)
    else:
        # Fallback if something went wrong
        best_overall_centers = _build_grid_init()
        best_overall_radius = _compute_radius(best_overall_centers)
        
    return best_overall_centers

# EVOLVE-BLOCK-END
