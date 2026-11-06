"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the actual radius for a given set of centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
        
    # Pairwise distances
    # Using squared distances for speed, then sqrt
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    return min(min_wall, min_pair_dist / 2.0)

def _epigraph_objective_and_gradient(x, epsilon=0.01):
    """
    Objective: Minimize the maximum violation of constraints.
    We want to maximize r. Let r be a variable.
    Constraints:
    1. ||c_i - c_j|| >= r  ->  r - ||c_i - c_j|| <= 0
    2. |c_ik| + r <= 0.5   ->  |c_ik| + r - 0.5 <= 0
    
    We use a smooth approximation for the max function (soft-max) or just 
    minimize the sum of squared violations + penalty on r.
    
    A more robust approach for L-BFGS-B:
    Minimize f(x) = -r + penalty * sum( max(0, r - dist_ij)^2 + max(0, |c_ik| + r - 0.5)^2 )
    But L-BFGS-B doesn't handle constraints well directly without SLSQP.
    
    Alternative: Minimize the maximum constraint violation directly using a 
    smooth max approximation (log-sum-exp or similar).
    
    Let's use the following formulation:
    Minimize:  -r  (we want to maximize r)
    Subject to: 
       r - sqrt(||c_i - c_j||^2) <= 0
       |c_ik| + r - 0.5 <= 0
    
    Since L-BFGS-B only handles box constraints, we use a penalty method or 
    reformulate as minimizing the maximum violation.
    
    Let's minimize:  F(x) = max(0, r - min_pair_dist) + max(0, min_wall_dist - r) - r
    Actually, simpler: 
    We want to find r such that all constraints are satisfied.
    Let's minimize the "slack" s = max( r - min_pair_dist, min_wall_dist - r, 0 )
    But we also want to maximize r.
    
    Best approach for L-BFGS-B: 
    Minimize:  (1/r) * sum( max(0, r - d_ij)^2 ) + (1/r) * sum( max(0, wall_dist_i - r)^2 ) - r
    This is getting complex.
    
    Let's stick to a simpler, proven method: 
    Minimize the maximum distance violation.
    Let d_min = min( min_pair_dist, min_wall_dist * 2 )  # Note: wall constraint is dist to wall >= r, pair is dist >= 2r.
    Actually, let's normalize constraints to radius r.
    Constraint 1: ||c_i - c_j|| >= 2r  =>  2r - ||c_i - c_j|| <= 0
    Constraint 2: 0.5 - |c_ik| >= r    =>  r - (0.5 - |c_ik|) <= 0
    
    We minimize:  V(x) = max( max_{i<j} (2r - ||c_i - c_j||), max_{i,k} (r - (0.5 - |c_ik|)), 0 )
    And we want to maximize r.
    
    This is a max-min problem.
    Let's use the following trick:
    Minimize:  -r + lambda * max(0, V(x))^2
    If V(x) > 0, we penalize violation. If V(x) <= 0, we just maximize r.
    
    However, calculating max is non-differentiable.
    We can use a smooth max: m(x) = (1/beta) * log( sum( exp(beta * x_i) ) )
    
    Let's define the objective to minimize:
    J(x) = -r + C * ( smooth_max( violations ) )^2
    
    Violations:
    v_pair_ij = 2r - dist_ij
    v_wall_ik = r - wall_dist_ik
    
    We compute smooth max of all v_pair and v_wall.
    """
    centers = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    # Calculate pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Avoid sqrt for diagonal
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centers)
    
    # Violations
    # Pair violation: 2r - dist
    v_pair = 2*r - dist
    np.fill_diagonal(v_pair, -np.inf)
    
    # Wall violation: r - wall_dist
    v_wall = r - np.ravel(wall_dist)
    
    # Combine all violations into one vector
    # We only care about positive violations (where constraint is broken)
    # To make it smooth, we use log-sum-exp on the violations.
    # If all violations are negative, smooth max is approx min violation (negative).
    # We want to penalize if smooth_max > 0.
    
    beta = 20.0  # Sharpness of max approximation
    
    # To avoid overflow in exp, shift by max
    all_violations = np.concatenate([np.ravel(v_pair), v_wall])
    v_max = np.max(all_violations)
    
    if v_max < -10:
        # All constraints satisfied with large margin
        smooth_max_val = v_max
    else:
        # Stable log-sum-exp
        exp_vals = np.exp(beta * (all_violations - v_max))
        smooth_max_val = v_max + (1.0/beta) * np.log(np.sum(exp_vals))
        
    # Objective: -r (maximize r) + penalty if smooth_max_val > 0
    penalty = 0.0
    if smooth_max_val > 0:
        penalty = 100.0 * smooth_max_val**2  # Strong penalty for violation
    
    objective = -r + penalty
    
    # Gradient calculation is complex for smooth max.
    # For simplicity and robustness in this context, we will use a 
    # numerical gradient or rely on L-BFGS-B's finite difference if we don't provide jac.
    # However, finite difference on 151 variables is slow.
    
    # Let's provide an analytical gradient approximation for the penalty term.
    # grad of smooth_max_val w.r.t x:
    # w_i = exp(beta * (v_i - v_max)) / sum(exp(...))
    # d(smooth_max)/dx = sum( w_i * dv_i/dx )
    
    # This is getting too complex for a single function call in time.
    # Alternative Strategy: Use SLSQP with constraints? 
    # SLSQP is slower but handles constraints natively.
    # Given N=50, 151 variables. SLSQP might be too slow.
    
    # Let's revert to a simpler, effective heuristic:
    # 1. Start with a good initial guess.
    # 2. Use L-BFGS-B to minimize the maximum violation (approximated by sum of squared violations for active sets).
    
    # Actually, let's use the "push apart" idea but vectorized and smoother.
    # Force F_i = sum_{j!=i} (1/dist_ij)^2 * (c_i - c_j) for pairs too close.
    # Force F_i += (1/wall_dist_i)^2 * normal for walls too close.
    # This is a gradient descent on a repulsive potential.
    
    return objective

def _repulsive_forces(centers, r, k=10.0):
    """
    Calculate repulsive forces based on a potential energy function.
    U = sum_{i<j} 1/(dist_ij - 2r)^2  (if dist < 2r, else 0)
    U += sum_{i,k} 1/(wall_dist_ik - r)^2 (if wall_dist < r, else 0)
    
    This creates strong forces when constraints are violated.
    """
    forces = np.zeros_like(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # We only apply force if dist < 2r + margin
    margin = 0.1 * r
    mask = dist < (2*r + margin)
    
    if np.any(mask):
        # Force magnitude: proportional to 1/(dist - 2r)^2 is singular.
        # Use 1/(dist - 2r + epsilon)^2
        eps = 1e-3
        d_eff = dist[mask] - 2*r + eps
        mag = k / (d_eff**2)
        
        # Direction is diff / dist
        # We need to apply this to the matrix
        force_mat = np.zeros_like(dist)
        force_mat[mask] = mag
        
        # Force on i from j is proportional to (c_i - c_j) / dist^3 * k
        # Actually, d/dx (1/(|x|^2 - a^2)) is proportional to x / (|x|^2 - a^2)^2
        # Let's just use a simple spring force for simplicity and stability:
        # F = k * (2r - dist) * direction, if dist < 2r
        
        # Reset and use linear spring
        force_mat = np.zeros_like(dist)
        # Only for violating pairs
        violating = dist < 2*r
        if np.any(violating):
            # Displacement vector
            disp = diff[violating]
            dists = dist[violating]
            # Force magnitude: k * (2r - dist)
            mags = k * (2*r - dists)
            # Unit vector
            unit_vecs = disp / dists[:, np.newaxis]
            # Force on i is +mags * unit_vecs (pushes i away from j)
            # Force on j is -mags * unit_vecs
            
            # Accumulate forces
            indices = np.where(violating)
            for idx in range(len(indices[0])):
                i = indices[0][idx]
                j = indices[1][idx]
                f_vec = mags[idx] * unit_vecs[idx]
                forces[i] += f_vec
                forces[j] -= f_vec
                
    # Wall forces
    wall_dist = 0.5 - np.abs(centers)
    for i in range(N):
        for k_coord in range(3):
            wd = wall_dist[i, k_coord]
            if wd < r:
                # Push away from wall
                # Normal is sign(c_ik)
                sign = np.sign(centers[i, k_coord])
                if sign == 0: sign = 1
                f_mag = k * (r - wd)
                forces[i, k_coord] += sign * f_mag
                
    return forces

def _optimize_repulsive(centers_init, r_init, time_limit=10.0):
    """
    Gradient descent on repulsive potential.
    """
    start = time.time()
    centers = centers_init.copy()
    r = r_init
    k = 1.0
    step_size = 0.01
    
    best_r = r
    best_centers = centers.copy()
    
    # Calculate initial radius
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()
    
    for it in range(500):
        if time.time() - start > time_limit:
            break
            
        forces = _repulsive_forces(centers, r, k=10.0)
        
        # Normalize forces to prevent explosion
        f_norm = np.linalg.norm(forces)
        if f_norm > 1e-6:
            forces = forces / f_norm
            
        centers = centers + step_size * forces
        
        # Clip to cube
        centers = np.clip(centers, -0.499, 0.499)
        
        # Update r estimate (try to increase r if we have space)
        # Actually, we want to find the max r for this configuration.
        # Let's just compute the radius and see if it improved.
        new_r = _compute_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            
        # Adaptive step size: decrease if oscillating
        if it % 10 == 0:
            step_size *= 0.9
            k *= 0.95
            
    return best_centers, best_r

def _make_fcc_initial():
    """Face-Centered Cubic lattice initial guess."""
    # FCC unit cell has 4 atoms.
    # We need 50.
    # Let's generate a grid of FCC points.
    # FCC basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # In a cube of side L.
    # Let's use a 4x4x4 grid of unit cells? No, that's too many.
    # Let's use a 3x3x3 grid of FCC unit cells? 27 * 4 = 108 points. Too many.
    # 2x2x2 grid of FCC unit cells? 8 * 4 = 32 points. Not enough.
    # Let's mix FCC and simple cubic.
    
    # Alternative: Start with a simple cubic grid and jitter.
    # Or, generate points on a grid and optimize.
    
    # Let's try a 5x5x4 grid (100 points) and take the first 50 with best spacing?
    # No, just take the first 50.
    
    # Better: Use a "quasi-random" lattice like Halton sequence?
    # Or just a jittered cubic grid.
    
    # Let's create a 4x4x4 grid (64 points) and take 50.
    # Spacing = 1/4 = 0.25. Radius approx 0.125.
    # We want ~0.136.
    # A 3x4x5 grid? 60 points.
    # Spacing in x: 1/3, y: 1/4, z: 1/5.
    # Min spacing is 0.2. Radius 0.1. Too small.
    
    # The optimal packing is likely close to a distorted FCC or HCP.
    # Let's generate a random set of points and use the repulsive optimizer.
    
    return None

def _make_random_initial(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.49, 0.49, size=(N, 3))

def _make_grid_initial(seed, dims=(4,4,4)):
    rng = np.random.RandomState(seed)
    nx, ny, nz = dims
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    
    grid = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    # Shuffle and take N
    if len(grid) >= N:
        idx = rng.choice(len(grid), N, replace=False)
        centers = grid[idx]
    else:
        # Pad with random
        centers = grid.copy()
        while len(centers) < N:
            centers = np.vstack([centers, rng.uniform(-0.49, 0.49, size=(1,3))])
            
    # Add small jitter
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializations
    inits = []
    
    # 1. Jittered 4x4x4 grid (64 pts -> 50)
    inits.append(_make_grid_initial(1, (4,4,4)))
    inits.append(_make_grid_initial(2, (4,4,4)))
    
    # 2. Jittered 5x5x4 grid (100 pts -> 50)
    inits.append(_make_grid_initial(3, (5,5,4)))
    
    # 3. Random
    inits.append(_make_random_initial(10))
    inits.append(_make_random_initial(20))
    
    # 4. "FCC-like" attempt: 
    # Generate FCC points in a 2x2x2 supercell and scale to fit 50?
    # Actually, let's just use the repulsive optimizer on these inits.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time
        remaining = deadline - time.time()
        time_per_init = min(20.0, remaining / (len(inits) - i + 1))
        
        try:
            # First, get a rough radius
            r0 = _compute_radius(init)
            
            # Run repulsive optimization
            opt_centers, opt_radius = _optimize_repulsive(init, r0, time_limit=time_per_init)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass

    # If we have time, do a final polish on the best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        # Use L-BFGS-B on the smooth penalty objective for final refinement
        # We need a proper objective function for L-BFGS-B
        def _smooth_penalty_obj(x):
            centers = x[:N*3].reshape(N, 3)
            r = x[N*3]
            
            # Check bounds
            if np.any(centers > 0.5) or np.any(centers < -0.5) or r < 0:
                return 1e6
                
            # Pairwise
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            # Wall
            wall_dist = 0.5 - np.abs(centers)
            
            # Violations
            v_pair = 2*r - dist
            np.fill_diagonal(v_pair, -np.inf)
            v_wall = r - np.ravel(wall_dist)
            
            all_v = np.concatenate([np.ravel(v_pair), v_wall])
            
            # Soft max
            beta = 10.0
            v_max = np.max(all_v)
            if v_max < -1:
                soft_max = v_max
            else:
                exp_vals = np.exp(beta * (all_v - v_max))
                soft_max = v_max + (1.0/beta) * np.log(np.sum(exp_vals))
                
            penalty = 0.0
            if soft_max > 0:
                penalty = 10.0 * soft_max**2
                
            return -r + penalty

        x0 = np.concatenate([best_centers.flatten(), [best_radius]])
        
        try:
            res = minimize(_smooth_penalty_obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.499, 0.499)]*(N*3) + [(0, 0.5)],
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < -best_radius + 1e-6:
                new_centers = res.x[:N*3].reshape(N, 3)
                new_r = res.x[N*3]
                
                # Verify with strict radius calculation
                strict_r = _compute_radius(new_centers)
                if strict_r > best_radius:
                    best_radius = strict_r
                    best_centers = new_centers
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        best_centers = _make_grid_initial(42, (4,4,4))
        
    return best_centers

# EVOLVE-BLOCK-END
