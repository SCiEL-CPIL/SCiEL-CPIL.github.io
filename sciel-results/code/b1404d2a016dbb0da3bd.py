import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 145.0  # Leave margin before 150s limit

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)

    def compute_radius(centers):
        """Compute the minimum radius for a set of centers."""
        if centers.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check inside cube with a small tolerance
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return 0.0
            
        # Pair distances
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        return max(r, 0.0)

    def compute_radius_and_gradient(centers):
        """Compute radius and gradient of the objective (min of pair-dist/2 and wall-dist)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        
        grad = np.zeros_like(centers)
        
        if min_pair / 2.0 <= min_wall + 1e-12:
            # Pair constraint is binding
            threshold = min_pair_sq * (1.0 + 1e-6)
            mask = dist_sq <= threshold
            np.fill_diagonal(mask, False)
            
            d = np.sqrt(dist_sq)
            d_safe = np.where(d > 1e-10, d, 1.0)
            
            W = np.zeros_like(dist_sq)
            W[mask] = 1.0 / (2.0 * d_safe[mask])
            
            W_sum = np.sum(W, axis=1)
            Wc = W.dot(centers)
            
            grad = centers * W_sum[:, np.newaxis] - Wc
            
        else:
            # Wall constraint is binding
            threshold = min_wall + 1e-6
            mask_wall = wall <= threshold
            
            signs = np.sign(centers)
            signs[signs == 0] = 1.0
            
            grad = -signs * mask_wall
        
        gmag = np.linalg.norm(grad)
        if gmag > 1e-10:
            grad /= gmag
        
        return r, grad

    def optimize_steepest(centers, max_steps=5000, init_step=0.005):
        """Steepest ascent on the radius objective with adaptive step size."""
        centers = centers.copy().astype(float)
        current_r, _ = compute_radius_and_gradient(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        step = init_step
        no_improve = 0
        stall_count = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
            
            r, grad = compute_radius_and_gradient(centers)
            current_r = r
            
            trial = centers + step * grad
            trial = np.clip(trial, -0.4999, 0.4999)
            
            new_r = compute_radius(trial)
            
            if new_r > current_r + 1e-12:
                centers = trial
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                step = min(step * 1.02, 0.05)
            else:
                step *= 0.5
                no_improve += 1
                stall_count += 1
                
                if stall_count > 30:
                    n_perturb = max(1, N // 10)
                    idx = np.random.choice(N, size=n_perturb, replace=False)
                    centers[idx] += np.random.randn(n_perturb, 3) * 0.02
                    centers = np.clip(centers, -0.4999, 0.4999)
                    step = init_step
                    stall_count = 0
                    no_improve = 0
            
            if no_improve > 500:
                n_perturb = max(2, N // 5)
                idx = np.random.choice(N, size=n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.04
                centers = np.clip(centers, -0.4999, 0.4999)
                step = init_step
                no_improve = 0
                stall_count = 0
        
        return best_centers, best_r

    def smooth_objective(x, p):
        """
        Objective function: Sum of inverse powers of constraint values.
        Minimizing this approximates maximizing the minimum constraint value.
        """
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dists = np.sqrt(dist_sq)
        
        # Constraint values for pairs: d_ij / 2
        pair_vals = dists / 2.0
        np.fill_diagonal(pair_vals, 1.0)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        
        # Calculate sum of inverse powers
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        pair_vals_masked = np.where(mask, pair_vals, 1.0)
        term_pairs = np.sum(pair_vals_masked ** (-p))
        
        # Wall term
        wall_dists_clipped = np.maximum(wall_dists, 1e-6)
        term_walls = np.sum(wall_dists_clipped ** (-p))
        
        return term_pairs + term_walls

    def smooth_gradient(x, p):
        """
        Gradient of the smooth objective.
        """
        c = x.reshape(N, 3)
        
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dists = np.sqrt(dist_sq)
        
        pair_vals = dists / 2.0
        np.fill_diagonal(pair_vals, 1.0)
        
        # Weights for pairs
        inv_pair_vals = np.zeros_like(pair_vals)
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        inv_pair_vals[mask] = pair_vals[mask] ** (-p - 1)
        
        weights = p * inv_pair_vals
        np.fill_diagonal(weights, 0.0)
        
        W_sum = np.sum(weights, axis=1)
        Wc = weights.dot(c)
        
        grad_pairs = - (c * W_sum[:, np.newaxis] - Wc)
        
        # Walls
        wall_dists = 0.5 - np.abs(c)
        wall_dists_clipped = np.maximum(wall_dists, 1e-6)
        
        inv_walls = wall_dists_clipped ** (-p - 1)
        
        signs = np.sign(c)
        signs[signs == 0] = 1.0
        
        grad_walls = p * inv_walls * signs
        
        total_grad = grad_pairs + grad_walls
        
        return total_grad.reshape(N * 3)

    best_r = 0.0
    best_centers = None
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    inits = []
    
    # Init 1: Jittered 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid_pts = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_pts.append([x, y, z])
    init1 = np.array(grid_pts[:N]) + rng.normal(0, 0.01, (N, 3))
    inits.append(init1)
    
    # Init 2: Random uniform
    inits.append(rng.uniform(-0.44, 0.44, (N, 3)))
    
    # Init 3: Jittered grid with larger noise
    init3 = np.array(grid_pts[:N]) + rng.normal(0, 0.03, (N, 3))
    inits.append(init3)
    
    # Init 4: Face-centered cubic inspired
    fc_pts = []
    for layer in range(5):
        z = -0.4 + layer * 0.2
        for i in range(4):
            for j in range(5):
                x = -0.4 + i * 0.2 + (0.05 if layer % 2 else 0)
                y = -0.4 + j * 0.2 + (0.05 if (layer + i) % 2 else 0)
                fc_pts.append([x, y, z])
    inits.append(np.array(fc_pts[:N]))
    
    for init_idx, init in enumerate(inits):
        if remaining() < 25.0:
            break
            
        centers = np.clip(init, -0.4999, 0.4999)
        x = centers.flatten()
        
        # Annealing of p for smooth optimization
        ps = [4, 8, 16, 32, 64, 128]
        
        for p in ps:
            if remaining() < 10.0:
                break
                
            try:
                def obj(x_curr, p_val=p):
                    return smooth_objective(x_curr, p_val)
                
                def grad(x_curr, p_val=p):
                    return smooth_gradient(x_curr, p_val)
                
                # L-BFGS-B
                res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                               bounds=[(-0.4999, 0.4999)] * (N * 3),
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
                
                if res.fun < 1e10:
                    x_new = np.clip(res.x, -0.4999, 0.4999)
                    c_new = x_new.reshape(N, 3)
                    r_new = compute_radius(c_new)
                    
                    if r_new > best_r:
                        best_r = r_new
                        best_centers = c_new.copy()
                        x = x_new
                    else:
                        # Keep the smoother result for the next p iteration even if raw radius didn't improve immediately
                        # This helps escape local minima in the smooth landscape
                        x = x_new
                        # But we should probably track the best raw radius separately
                        # For now, just update x to continue annealing from the new point
                        # Actually, if r_new didn't improve best_r, we might still want to continue from x_new
                        # because it might be better for the next higher p
                        pass
                    
            except Exception:
                pass
        
        # Polishing phase with steepest ascent on raw radius
        if best_centers is not None and remaining() > 15.0:
            # Polish the current best found in this start
            c_polished, r_polished = optimize_steepest(best_centers, max_steps=5000, init_step=0.002)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = c_polished

    # Final global polish if time permits
    if remaining() > 10.0 and best_centers is not None:
        c_final, r_final = optimize_steepest(best_centers, max_steps=10000, init_step=0.001)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final

    if best_centers is None:
        return inits[0]
    
    return best_centers
