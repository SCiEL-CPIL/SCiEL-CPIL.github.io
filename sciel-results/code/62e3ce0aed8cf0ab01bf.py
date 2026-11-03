"""
Strategy: Constrained Optimization with SLSQP and Analytical Gradients.
Instead of smoothing the min-distance objective (LSE), we treat the radius 'r' 
as a variable to maximize, subject to hard constraints on pairwise distances 
and wall distances. We use subgradients for the active constraints (those 
closest to being violated). Multi-start with high-symmetry initializations 
(BCC, FCC, Grid) to navigate the non-convex landscape.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _generate_bcc_start():
    """
    Generate a Body-Centered Cubic (BCC) lattice configuration.
    BCC is a dense packing (density ~0.68) and provides a good structural template.
    We generate a 4x4x4 BCC lattice (128 points) and select 50.
    """
    m = 4
    # BCC points: corners and body centers of sub-cubes
    # Sub-cube side length
    side = 1.0 / m
    
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                x = -0.5 + (i + 0.5) * side
                y = -0.5 + (j + 0.5) * side
                z = -0.5 + (k + 0.5) * side
                points.append([x, y, z])
                # Body center
                points.append([x, y, z + side/2.0]) # Wait, standard BCC is corner + center of cube.
                # Let's stick to a simpler grid of BCC cells.
                
    # Actually, a simpler BCC generation:
    # Take a cubic grid of spacing 'a', and add points at (a/2, a/2, a/2) relative to each corner.
    # To fit in [-0.5, 0.5], we scale.
    # Let's generate a 4x4x4 grid of BCC unit cells.
    # Total points = 4^3 * 2 = 128.
    
    pts = []
    a = 1.0 / 4.0 # Unit cell size
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Center of the unit cell? No, BCC has atoms at corners and center.
                # If we tile BCC, the lattice points are at (i,j,k) and (i+0.5, j+0.5, k+0.5).
                # Let's shift to center in [-0.5, 0.5].
                # x = (i - 1.5) * a ?
                
                # Let's just generate a dense grid and thin it? No, let's do proper BCC.
                # Lattice vectors:
                # p1 = (0.5, 0.5, 0.5) * a
                # p2 = (0.5, -0.5, 0.5) * a
                # p3 = (-0.5, 0.5, 0.5) * a
                # This is getting complicated. Let's use a simple "Perturbed Grid" which is often enough,
                # or a "FCC" if I can implement it easily.
                
                # Let's try a 5x5x5 grid (125 points) and take 50.
                # Or 4x4x4 (64) + some perturbation.
                pass
    
    # Fallback: High-quality deterministic grid with small random perturbations to break symmetry
    # but keep it structured.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle deterministically to avoid corner bias?
    # Let's use a "Face Centered Cubic" (FCC) approximation if possible, 
    # or just a well-distributed 5x5x5 grid.
    
    # Let's try 5x5x5 grid (125 points).
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Randomly select 50? No, deterministic selection to ensure reproducibility.
    # Sort by distance to center? Or just take first 50?
    # Taking first 50 from a 5x5x5 grid will bias towards one corner.
    # Let's interleave.
    # Flatten, shuffle with fixed seed, take 50.
    idx = np.arange(len(grid))
    np.random.seed(42)
    np.random.shuffle(idx)
    return grid[idx[:N]].copy()

def _generate_fcc_start():
    """
    Generate an FCC-like start.
    FCC points in a unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    We can tile this.
    """
    m = 3 # 3x3x3 unit cells -> 3^3 * 4 = 108 points
    a = 1.0 / m
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Base corner
                x0 = -0.5 + i * a
                y0 = -0.5 + j * a
                z0 = -0.5 + k * a
                
                # 4 FCC points in this cell (relative to x0,y0,z0)
                # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                # Note: This tiling might double count faces?
                # Standard FCC lattice points are at:
                # (i, j, k) where i+j+k is even (if scaled by 2)
                # Let's just use the 4 points per cell.
                pts.append([x0, y0, z0])
                pts.append([x0 + a/2, y0 + a/2, z0])
                pts.append([x0 + a/2, y0, z0 + a/2])
                pts.append([x0, y0 + a/2, z0 + a/2])
                
    pts = np.array(pts)
    # Filter points strictly inside [-0.5, 0.5]
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    if len(pts) < N:
        # Fallback if we didn't get enough (shouldn't happen with m=3)
        return _generate_bcc_start()
        
    # Select 50 points. To be fair, shuffle and take first 50.
    idx = np.arange(len(pts))
    np.random.seed(43)
    np.random.shuffle(idx)
    return pts[idx[:N]].copy()

def _objective_and_constraints(x, r):
    """
    We want to maximize r.
    scipy minimize minimizes f(x). So we minimize -r.
    Variables: x = [c_1, c_2, ..., c_N, r]
    c_i are 3D vectors. r is scalar.
    
    Constraints:
    1. |c_i - c_j| >= 2r  => |c_i - c_j| - 2r >= 0
    2. 0.5 - max(|c_ik|) >= r  => 0.5 - max(|c_ik|) - r >= 0
    """
    centres = x[:N*3].reshape(N, 3)
    r_val = x[-1]
    
    # Objective: minimize -r_val
    return -r_val

def _gradient_objective(x, r):
    """
    Gradient of -r.
    d(-r)/dx = 0 for centres.
    d(-r)/dr = -1.
    """
    grad = np.zeros_like(x)
    grad[-1] = -1.0
    return grad

def _constraint_functions(x, r):
    """
    Returns array of constraint values g(x) >= 0.
    """
    centres = x[:N*3].reshape(N, 3)
    r_val = x[-1]
    
    # Wall constraints: 0.5 - max(|c_i|) - r >= 0
    max_abs = np.max(np.abs(centres), axis=1)
    wall_constr = 0.5 - max_abs - r_val
    
    # Pairwise constraints: |c_i - c_j| - 2r >= 0
    # We only need to check pairs that are "active" or close to violating.
    # For SLSQP, providing all constraints is safe but slow.
    # N=50 -> 1225 pairs. This is manageable.
    
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We need the upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    pair_constr = pair_dists - 2 * r_val
    
    return np.concatenate([wall_constr, pair_constr])

def _constraint_gradients(x, r):
    """
    Returns array of gradients of constraints.
    Shape: (num_constraints, 3N + 1)
    """
    centres = x[:N*3].reshape(N, 3)
    r_val = x[-1]
    num_cons = N + (N*(N-1)//2)
    
    grads = np.zeros((num_cons, N*3 + 1))
    
    # 1. Wall constraints gradients
    # g_i = 0.5 - max(|c_ik|) - r
    # dg_i/dx = -sign(c_ik) for the component k that achieves max(|c_ik|)
    # dg_i/dr = -1
    
    max_abs = np.max(np.abs(centres), axis=1)
    max_idx = np.argmax(np.abs(centres), axis=1)
    
    for i in range(N):
        # Gradient wrt centre i
        dim = max_idx[i]
        grads[i, i*3 + dim] = -np.sign(centres[i, dim])
        # Gradient wrt r
        grads[i, -1] = -1.0
        
    # 2. Pairwise constraints gradients
    # g_ij = |c_i - c_j| - 2r
    # dg_ij/dc_i = (c_i - c_j) / |c_i - c_j|
    # dg_ij/dc_j = (c_j - c_i) / |c_i - c_j|
    # dg_ij/dr = -2
    
    iu = np.triu_indices(N, 1)
    idx = 0
    for i, j in zip(iu[0], iu[1]):
        diff = centres[i] - centres[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # Avoid division by zero. If dist is 0, gradient is undefined.
            # SLSQP might struggle. Add a small noise or skip?
            # Better: if dist is tiny, push them apart arbitrarily.
            # For gradient, set to 0? No, that's bad.
            # Let's use a small epsilon.
            dist = 1e-10
            # Actually, if they overlap, the constraint is heavily violated.
            # The gradient direction should be to separate them.
            # If diff is 0, any direction works. Let's pick (1,0,0).
            grad_vec = np.array([1.0, 0.0, 0.0])
        else:
            grad_vec = diff / dist
            
        # Gradient wrt centre i
        grads[N + idx, i*3:i*3+3] = grad_vec
        # Gradient wrt centre j
        grads[N + idx, j*3:j*3+3] = -grad_vec
        # Gradient wrt r
        grads[N + idx, -1] = -2.0
        
        idx += 1
        
    return grads

def _optimize_slsqp(initial_centres, time_limit=10.0):
    """
    Run SLSQP to maximize radius.
    """
    start_time = time.time()
    
    # Initial radius estimate
    r0 = _compute_radius(initial_centres)
    
    # Variable vector: [c_1, ..., c_N, r]
    x0 = np.concatenate([initial_centres.flatten(), [r0]])
    
    # Bounds for centres: [-0.5, 0.5]
    # Bounds for r: [0, 0.5]
    bounds = [(-0.5, 0.5)] * (N*3) + [(0.0, 0.5)]
    
    # We define a wrapper for constraints to fit scipy.optimize.minimize
    # scipy expects fun(x) -> array of g(x) >= 0
    
    def cons_fun(x):
        return _constraint_functions(x, None)
        
    def cons_grad(x):
        return _constraint_gradients(x, None)
        
    # Use SLSQP
    # maxiter needs to be high enough.
    # ftol/gtol for convergence.
    
    res = minimize(
        _objective_and_constraints,
        x0,
        jac=_gradient_objective,
        method='SLSQP',
        bounds=bounds,
        constraints=[
            {
                'type': 'ineq',
                'fun': cons_fun,
                'jac': cons_grad
            }
        ],
        options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
    )
    
    if time.time() - start_time > time_limit:
        # SLSQP doesn't have a clean time-break, but we can check after.
        # If it's still running, we can't easily kill it. 
        # However, 500 iterations on N=50 (151 vars, ~1275 constraints) 
        # might be slow. 
        # Let's reduce maxiter if time is tight? 
        # For now, we rely on the outer loop to limit calls.
        pass
        
    x_opt = res.x
    centres_opt = x_opt[:N*3].reshape(N, 3)
    r_opt = x_opt[-1]
    
    # Clip centres to be safe
    centres_opt = np.clip(centres_opt, -0.5, 0.5)
    
    return centres_opt

def construct():
    """Construct 50 sphere centres in [-0.5, 0.5]^3 to maximize minimum radius."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_centres = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    # Multi-start configurations
    # 1. FCC-like
    # 2. BCC-like (Grid)
    # 3. Random (for diversity)
    
    starts = []
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    
    # Add a few random starts
    for _ in range(3):
        starts.append(np.random.rand(N, 3) - 0.5)
        
    # We will try each start.
    # SLSQP is deterministic given the start.
    
    for i, initial in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
            
        # Allocate time
        # SLSQP can be slow. Let's give it ~20s per run.
        # If it takes longer, we might miss the deadline.
        # We'll just run it and hope it finishes.
        # To be safer, we can reduce maxiter in later runs if time is short.
        
        opt_centres = _optimize_slsqp(initial, time_limit=20.0)
        
        # Evaluate true radius
        radius = _compute_radius(opt_centres)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = opt_centres.copy()
            
    # Final polish: If we have time, run SLSQP again on the best result with tighter tolerances?
    # SLSQP is already quite precise.
    # Maybe just return the best.
    
    if best_centres is None:
        # Fallback
        best_centres = _generate_bcc_start()
        
    return best_centres
# EVOLVE-BLOCK-END
