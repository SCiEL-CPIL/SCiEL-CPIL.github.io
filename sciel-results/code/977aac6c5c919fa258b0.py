import numpy as np
import time

N = 100
DEADLINE = 145.0  # Leave margin before 150s limit

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_radius(centers):
        """Compute the minimum radius for a set of centers.
        Returns (radius, min_pair_dist, min_wall_dist) for potential reuse."""
        # Pair distances
        # Using broadcasting for vectorization
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        return r, min_pair, min_wall

    def compute_radius_only(centers):
        """Faster version if we don't need the components."""
        r, _, _ = compute_radius(centers)
        return r

    def compute_forces(centers, beta=1.0):
        """Compute repulsion forces between all pairs and wall forces.
        beta controls the 'softness' of the repulsion."""
        # Pair forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        dist_sq_safe = np.where(dist_sq < 1e-10, 1e-10, dist_sq)
        
        # Inverse square law (or higher power for steeper repulsion)
        # Force magnitude is proportional to 1/d^2, direction is along diff
        # F_ij = (c_i - c_j) / |c_i - c_j|^2 * (1/|c_i - c_j|) ? No.
        # Potential V ~ 1/r^k. Force F = -grad V ~ k/r^(k+1).
        # Let's use F ~ 1/r^3 (so V ~ 1/r^2) to make it strong at close range.
        # Actually, standard soft sphere is V ~ (1-r)^2.
        # For packing, we want to maximize min distance.
        # Minimizing sum(1/d^2) is a good proxy.
        # Force on i from j: (c_i - c_j) / d_ij^2 * (1/d_ij) ?
        # Let's use F = (c_i - c_j) / d_ij^4. This corresponds to V = 1/d^3.
        
        inv_dist_sq = 1.0 / dist_sq_safe
        # Gradient of 1/d^2 is -2/d^4 * (c_i - c_j).
        # We want to minimize sum(1/d^2).
        # grad_i sum_{j!=i} 1/|c_i-c_j|^2 = sum_{j!=i} -2 (c_i-c_j) / |c_i-c_j|^4
        # So force (direction of descent) is sum 2 (c_i-c_j) / d^4.
        # We move in direction of negative gradient, so we add this force.
        
        # Vectorized computation:
        # W[i,j] = 2 / dist_sq[i,j]^2
        W = 2.0 / (dist_sq_safe * dist_sq_safe)
        np.fill_diagonal(W, 0.0)
        
        # forces[i] = sum_j W[i,j] * (c_i - c_j)
        # = c_i * sum_j W[i,j] - sum_j W[i,j] * c_j
        W_sum = np.sum(W, axis=1)
        Wc = W.dot(centers)
        pair_forces = centers * W_sum[:, np.newaxis] - Wc
        
        # Wall forces
        # We want to push spheres away from walls.
        # Potential V_wall = sum_k (1/(0.5 - |x_k|)^2) for x_k > 0, etc.
        # Or simply linear/quadratic push near walls.
        # Let's use a strong inverse square push near walls.
        # If x > 0, distance to wall is 0.5 - x. Force should be -1/d^2 (push left).
        # If x < 0, distance to wall is 0.5 + x. Force should be +1/d^2 (push right).
        
        wall_forces = np.zeros_like(centers)
        for k in range(3):
            x = centers[:, k]
            # Distance to positive wall
            d_pos = 0.5 - x
            # Distance to negative wall
            d_neg = 0.5 + x
            
            # Only apply force if close to wall (e.g., d < 0.2)
            # Force magnitude ~ 1/d^2
            f_pos = np.zeros(N)
            mask_pos = d_pos < 0.2
            if np.any(mask_pos):
                # Clip distance to avoid infinity
                d_safe = np.maximum(d_pos[mask_pos], 0.01)
                f_pos[mask_pos] = -1.0 / (d_safe * d_safe) # Push towards center (negative x direction)
                
            f_neg = np.zeros(N)
            mask_neg = d_neg < 0.2
            if np.any(mask_neg):
                d_safe = np.maximum(d_neg[mask_neg], 0.01)
                f_neg[mask_neg] = 1.0 / (d_safe * d_safe) # Push towards center (positive x direction)
            
            wall_forces[:, k] = f_pos + f_neg
        
        return pair_forces + wall_forces

    def optimize_force(centers, max_steps=5000, init_temp=0.05):
        """Simulated annealing / Langevin dynamics with force-directed repulsion."""
        centers = centers.copy().astype(float)
        current_r, _, _ = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        temp = init_temp
        step = 0.01
        no_improve = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
            
            forces = compute_forces(centers)
            
            # Normalize forces to prevent huge jumps
            fmag = np.linalg.norm(forces, axis=1)
            max_f = np.max(fmag)
            if max_f > 1e-10:
                forces = forces / max_f
            
            # Add noise
            noise = np.random.randn(N, 3) * temp
            displacement = step * (forces + noise)
            
            trial = centers + displacement
            # Keep inside cube (with small margin)
            trial = np.clip(trial, -0.499, 0.499)
            
            new_r, _, _ = compute_radius(trial)
            
            # Acceptance criterion
            # Always accept if better.
            # Accept worse with probability exp(-(current_r - new_r) / temp)
            # Note: current_r is the radius. We want to maximize it.
            # So "worse" means new_r < current_r.
            # Delta E = current_r - new_r (positive if worse)
            if new_r >= current_r:
                accept = True
            else:
                delta = current_r - new_r
                if temp > 1e-6:
                    accept = np.random.rand() < np.exp(-delta / temp)
                else:
                    accept = False
            
            if accept:
                centers = trial
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            # Cooling
            temp *= 0.995
            step *= 0.999
            
            if no_improve > 200:
                # Perturb to escape local optima
                n_perturb = max(1, N // 10)
                idx = np.random.choice(N, size=n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.02
                centers = np.clip(centers, -0.499, 0.499)
                no_improve = 0
                # Reheat slightly
                temp = max(temp, 0.01)
                step = 0.01
                
        return best_centers, best_r

    def compute_radius_gradient(centers):
        """Compute radius and gradient for steepest ascent."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        grad = np.zeros_like(centers)
        
        # Determine active constraints
        # Pair constraint active if min_pair/2 <= min_wall
        if min_pair / 2.0 <= min_wall + 1e-9:
            # Find pairs contributing to min_pair
            # Use a small tolerance to include nearly-equal pairs
            threshold = min_pair_sq * (1.0 + 1e-5)
            mask = dist_sq <= threshold
            np.fill_diagonal(mask, False)
            
            # Gradient of d_ij = sqrt(dist_sq_ij) wrt c_i is (c_i - c_j) / d_ij
            # We want to maximize min(d_ij/2).
            # Subgradient: sum over active pairs of (c_i - c_j) / (2 * d_ij)
            
            d = np.sqrt(dist_sq)
            d_safe = np.where(d > 1e-10, d, 1.0)
            
            # W[i,j] = 1 / (2 * d_ij) if active
            W = np.zeros_like(dist_sq)
            W[mask] = 1.0 / (2.0 * d_safe[mask])
            
            W_sum = np.sum(W, axis=1)
            Wc = W.dot(centers)
            grad = centers * W_sum[:, np.newaxis] - Wc
        else:
            # Wall constraint active
            # Find (i,k) where wall[i,k] is near min_wall
            threshold = min_wall + 1e-5
            mask_wall = wall <= threshold
            
            # dr/d(c_i,k) = -sign(c_i,k)
            # If c_i,k > 0, wall = 0.5 - c. d(wall)/dc = -1. We want to increase wall, so move c in direction that increases 0.5-c => decrease c. Grad is -1.
            # If c_i,k < 0, wall = 0.5 + c. d(wall)/dc = 1. We want to increase wall, so move c in direction that increases 0.5+c => increase c. Grad is 1.
            # So grad = -sign(c).
            
            signs = np.sign(centers)
            # Handle zero
            signs[signs == 0] = 1.0 
            
            grad = -signs * mask_wall
            
        # Normalize gradient
        gmag = np.linalg.norm(grad)
        if gmag > 1e-10:
            grad /= gmag
            
        return r, grad

    def optimize_gradient(centers, max_steps=5000, init_step=0.005):
        """Steepest ascent on the radius objective."""
        centers = centers.copy().astype(float)
        current_r, _ = compute_radius_gradient(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        step = init_step
        no_improve = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
                
            r, grad = compute_radius_gradient(centers)
            current_r = r
            
            # Try a step in gradient direction
            trial = centers + step * grad
            trial = np.clip(trial, -0.4999, 0.4999)
            
            new_r = compute_radius_only(trial)
            
            if new_r > current_r + 1e-12:
                centers = trial
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                # Slightly increase step if improving
                step = min(step * 1.01, 0.02)
            else:
                # Try smaller step
                step *= 0.5
                no_improve += 1
                
                if no_improve > 50:
                    # Perturb
                    n_perturb = max(1, N // 20)
                    idx = np.random.choice(N, size=n_perturb, replace=False)
                    centers[idx] += np.random.randn(n_perturb, 3) * 0.01
                    centers = np.clip(centers, -0.4999, 0.4999)
                    step = init_step
                    no_improve = 0
        
        return best_centers, best_r

    best_r = 0.0
    best_centers = None
    
    np.random.seed(42)
    
    # Initializations
    inits = []
    
    # 1. 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.44, 0.44, nx)
    ys = np.linspace(-0.44, 0.44, ny)
    zs = np.linspace(-0.44, 0.44, nz)
    grid_pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                grid_pts.append([x, y, z])
    inits.append(np.array(grid_pts[:N]))
    
    # 2. Random
    inits.append(np.random.uniform(-0.44, 0.44, (N, 3)))
    
    # 3. Jittered Grid
    base = inits[0].copy()
    base += np.random.randn(N, 3) * 0.03
    base = np.clip(base, -0.4999, 0.4999)
    inits.append(base)
    
    # Process each initialization
    for init_idx, init in enumerate(inits):
        if remaining() < 20.0:
            break
            
        # Phase 1: Force-directed relaxation to break symmetries
        c_force, r_force = optimize_force(init, max_steps=2000, init_temp=0.05)
        
        # Phase 2: Gradient refinement
        c_grad, r_grad = optimize_gradient(c_force, max_steps=3000, init_step=0.005)
        
        if r_grad > best_r:
            best_r = r_grad
            best_centers = c_grad
        
        # Optional: Another force pass if time permits?
        # Let's just do one full cycle per init to ensure diversity.
        
    # Final polishing on the best result
    if remaining() > 15.0 and best_centers is not None:
        # A few more gradient passes
        for _ in range(3):
            if remaining() < 10.0:
                break
            c, r = optimize_gradient(best_centers, max_steps=2000, init_step=0.002)
            if r > best_r:
                best_r = r
                best_centers = c
            else:
                break
                
    if best_centers is None:
        return inits[0]
        
    return best_centers
