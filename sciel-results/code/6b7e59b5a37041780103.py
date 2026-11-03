"""Bootstrap spawn: model-and-optimize approach for 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape[0] != N or positions.shape[1] != 3:
            return 0.0
        # Check bounds
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        # Pairwise distances
        try:
            from scipy.spatial.distance import pdist
            dists = pdist(positions, 'euclidean')
        except ImportError:
            dists = []
            for i in range(positions.shape[0]):
                for j in range(i+1, positions.shape[0]):
                    d = np.linalg.norm(positions[i] - positions[j])
                    dists.append(d)
            dists = np.array(dists)
        
        if len(dists) > 0:
            min_dist = np.min(dists)
        else:
            min_dist = 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def _pairwise_and_wall(positions):
        """Return (pairwise_dist_matrix_diag_excluded, wall_distances) for smooth objectives."""
        from scipy.spatial.distance import pdist
        dists = pdist(positions, 'euclidean')
        walls = 0.5 - np.abs(positions)
        return dists, walls
    
    def smooth_surrogate(x_flat, beta):
        """
        Smooth surrogate: -soft_min of (pairwise_dists/2, wall_dists).
        soft_min of values v_i = -1/beta * log(sum(exp(-beta * v_i)))
        We minimize the negative, i.e. maximize soft_min.
        So objective = -soft_min = 1/beta * log(sum(exp(-beta * v_i)))
        """
        positions = x_flat.reshape(-1, 3)
        dists, walls = _pairwise_and_wall(positions)
        # All "radius candidates": pairwise/2 and wall distances
        pair_r = dists / 2.0
        all_vals = np.concatenate([pair_r, walls.ravel()])
        # soft_min = -1/beta * logsumexp(-beta * all_vals)
        # objective = -soft_min = 1/beta * logsumexp(-beta * all_vals)
        # Use stable logsumexp
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
        return logsumexp / beta
    
    def smooth_grad(x_flat, beta):
        """Gradient of the smooth surrogate (vectorized)."""
        positions = x_flat.reshape(-1, 3)
        dists, walls = _pairwise_and_wall(positions)
        pair_r = dists / 2.0
        all_vals = np.concatenate([pair_r, walls.ravel()])
        
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        exp_vals = np.exp(neg_beta_vals - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # softmax weights
        
        # d/dv_i [1/beta * logsumexp(-beta * v)] = -weights_i
        grad_all = -weights
        
        # Now compute gradient w.r.t. positions
        grad_pos = np.zeros_like(positions)
        
        n_pairs = len(pair_r)
        # Vectorized pairwise gradient computation
        # Build index arrays for all pairs
        i_idx = []
        j_idx = []
        for i in range(N):
            for j in range(i + 1, N):
                i_idx.append(i)
                j_idx.append(j)
        i_idx = np.array(i_idx)
        j_idx = np.array(j_idx)
        
        # Compute differences and distances for all pairs
        diffs = positions[i_idx] - positions[j_idx]  # shape (n_pairs, 3)
        d_safe = np.maximum(dists, 1e-12)
        
        # For each pair, contribution to grad: w * diff / (2*d) for pos[i], -w * diff / (2*d) for pos[j]
        # grad_all[:n_pairs] are the weights for pairwise terms
        pair_weights = grad_all[:n_pairs]  # shape (n_pairs,)
        
        # Compute per-pair gradient factor: w / (2*d)
        pair_factors = pair_weights / (2.0 * d_safe)  # shape (n_pairs,)
        
        # For each pair, add factor * diff to pos[i] and subtract from pos[j]
        # Use np.add.at for accumulation
        pair_grads = pair_factors[:, np.newaxis] * diffs  # shape (n_pairs, 3)
        
        np.add.at(grad_pos, i_idx, pair_grads)
        np.add.at(grad_pos, j_idx, -pair_grads)
        
        # Wall part: wall_k = 0.5 - |pos[k//3, k%3]|
        # d(wall_k)/d(pos) = -sign(pos[k//3, k%3]) for the relevant coordinate
        wall_grad = grad_all[n_pairs:]
        n_walls = len(wall_grad)
        if n_walls > 0:
            wall_indices = np.arange(n_walls)
            wall_point_idx = wall_indices // 3
            wall_coord_idx = wall_indices % 3
            pos_vals = positions[wall_point_idx, wall_coord_idx]
            signs = np.sign(pos_vals)  # 0 at exactly 0
            wall_contrib = wall_grad * (-signs)
            np.add.at(grad_pos, (wall_point_idx, wall_coord_idx), wall_contrib)
        
        return grad_pos.ravel()
    
    def make_initial_grid():
        """Create a reasonable initial grid configuration."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def make_bcc_lattice():
        """Create a body-centered cubic inspired arrangement."""
        positions = []
        a = 0.5
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * (2*a/3)
                    y = -a + (iy + 0.5) * (2*a/3)
                    z = -a + (iz + 0.5) * (2*a/3)
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + (2*a/3)/2, y + (2*a/3)/2, z + (2*a/3)/2])
        
        positions = np.array(positions[:N])
        return positions
    
    def make_random(seed=None):
        """Create random initial positions."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    def local_optimize_smooth(positions, beta=50.0, maxiter=500, ftol=1e-14):
        """Run local optimization using smooth surrogate with L-BFGS-B."""
        from scipy.optimize import minimize
        
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return smooth_surrogate(x, beta)
        
        def grad_flat(x):
            return smooth_grad(x, beta)
        
        try:
            res = minimize(obj_flat, positions.flatten(), jac=grad_flat, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions
    
    def local_optimize_direct(positions, maxiter=300, ftol=1e-13):
        """Run local optimization on the raw min-radius objective with L-BFGS-B."""
        from scipy.optimize import minimize
        
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_min_radius(x.reshape(-1, 3))
        
        try:
            res = minimize(obj_flat, positions.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions
    
    def polish_pass(positions, beta_list=(100.0, 200.0, 500.0, 1000.0), maxiter_each=200):
        """
        High-precision polish: run smooth optimization with increasing beta,
        then a final direct min-radius optimization.
        """
        pos = positions.copy()
        
        for beta in beta_list:
            if remaining() < 5:
                break
            pos = local_optimize_smooth(pos, beta=beta, maxiter=maxiter_each)
        
        # Final direct optimization for exact min
        if remaining() > 5:
            pos = local_optimize_direct(pos, maxiter=200)
        
        return pos
    
    best_positions = None
    best_radius = -1.0
    
    # Try multiple initializations
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]
    
    for name, init_func in initializations:
        if remaining() < 15:
            break
        
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            
            # Smooth optimization with moderate beta
            if remaining() > 25:
                pos_opt = local_optimize_smooth(positions, beta=30.0, maxiter=300)
            else:
                pos_opt = positions
            
            # Quick direct refinement
            if remaining() > 15:
                pos_opt = local_optimize_direct(pos_opt, maxiter=100)
            
            r = compute_min_radius(pos_opt)
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt.copy()
        except Exception:
            continue
    
    # Perturbations of the best solution
    if best_positions is not None and remaining() > 20:
        for perturb_seed in range(6):
            if remaining() < 10:
                break
            
            np.random.seed(1000 + perturb_seed)
            perturbed = best_positions + np.random.normal(0, 0.012, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            
            pos_opt = local_optimize_smooth(perturbed, beta=40.0, maxiter=200)
            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=100)
            
            r = compute_min_radius(pos_opt)
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt.copy()
    
    # Final high-precision polish of the best solution
    if best_positions is not None and remaining() > 15:
        polished = polish_pass(best_positions, beta_list=(100.0, 300.0, 800.0, 1500.0), maxiter_each=150)
        r_polished = compute_min_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished.copy()
    
    # Final clip to ensure validity
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    # Ensure we return a valid array
    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions
