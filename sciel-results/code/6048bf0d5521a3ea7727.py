"""
Combined approach: Vectorized Smooth Surrogate Optimization for Sphere Packing.
Uses soft-min via log-sum-exp with vectorized gradients for speed.
Implements beta-annealing for global-to-local convergence and a final direct polish.
Improved initializations and more effective annealing schedule.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)
        
        # 1. Calculate all "radius candidates" g_k
        # Pairwise: d_ij / 2
        tri_i, tri_j = np.triu_indices(N, k=1)
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0
        
        # Walls: 6 distances per point
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        
        # Combine all g_k
        g_all = np.concatenate([d_pairs, wall_dists])
        
        # 2. Soft Min Calculation
        # Use stable logsumexp
        scaled = -beta * g_all
        max_scaled = np.max(scaled)
        if not np.isfinite(max_scaled):
            return 0.0, np.zeros_like(c_flat)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        val = -log_sum_exp / beta
        
        # 3. Gradient Calculation
        weights = np.exp(scaled - log_sum_exp)
        # Normalize weights to avoid numerical issues
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0, np.zeros_like(c_flat)
        weights = weights / w_sum
        
        grad_c = np.zeros_like(c)
        
        # Pairwise gradient - fully vectorized using np.add.at
        n_pairs = len(d_pairs)
        # factor_k = weight_k / (4 * d_pairs_k)
        # grad_c[i] += factor_k * diff_k
        # grad_c[j] -= factor_k * diff_k
        
        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        # For pairs with tiny d, the weight should be negligible or we skip
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        
        weighted_diffs = factors[:, np.newaxis] * diff
        
        # Scatter: add to index i, subtract from index j
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)
        
        # Wall part
        w_start = n_pairs
        grad_c[:, 0] -= weights[w_start : w_start+N]
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # We want to MINIMIZE -soft_min.
        return -val, -grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val
            
        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init
            
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def make_initial_grid():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """Proper BCC lattice: corner + body-center points in a 3x3x3 supercell."""
        positions = []
        # Use a 3x3x3 grid of BCC unit cells
        a = 1.0 / 3.0  # lattice constant
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    # Corner point
                    x = -0.5 + ix * a
                    y = -0.5 + iy * a
                    z = -0.5 + iz * a
                    positions.append([x, y, z])
                    # Body-center point
                    positions.append([x + a/2, y + a/2, z + a/2])
        
        positions = np.array(positions, dtype=np.float64)
        
        # Keep only points within the cube
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]
        
        # If more than N, select best N by minimum distance
        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]
        
        # If fewer than N, pad with random
        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
        
        return positions[:N]

    def make_fcc_lattice():
        """Proper FCC lattice: corner + face-center points."""
        positions = []
        a = 1.0 / 3.0  # lattice constant for 3x3x3 supercell
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -0.5 + ix * a
                    y = -0.5 + iy * a
                    z = -0.5 + iz * a
                    # Corner
                    positions.append([x, y, z])
                    # Face centers
                    if ix < 2:
                        positions.append([x + a/2, y, z])
                    if iy < 2:
                        positions.append([x, y + a/2, z])
                    if iz < 2:
                        positions.append([x, y, z + a/2])
                    if ix < 2 and iy < 2:
                        positions.append([x + a/2, y + a/2, z])
                    if ix < 2 and iz < 2:
                        positions.append([x + a/2, y, z + a/2])
                    if iy < 2 and iz < 2:
                        positions.append([x, y + a/2, z + a/2])
        
        positions = np.array(positions, dtype=np.float64)
        
        # Keep only points within the cube
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]
        
        # If more than N, select best N by minimum distance
        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]
        
        # If fewer than N, pad with random
        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
        
        return positions[:N]

    def make_hcp_like():
        """Hexagonal close-packed inspired: two interleaved triangular layers."""
        positions = []
        # Layer 1: triangular lattice
        a = 0.32
        for row in range(4):
            y = -0.4 + row * a * np.sqrt(3)/2
            for col in range(4):
                x = -0.4 + col * a + (row % 2) * a/2
                z = -0.3
                if abs(x) <= 0.48 and abs(y) <= 0.48 and abs(z) <= 0.48:
                    positions.append([x, y, z])
        
        # Layer 2: shifted triangular lattice
        for row in range(4):
            y = -0.4 + row * a * np.sqrt(3)/2
            for col in range(4):
                x = -0.4 + col * a + (row % 2) * a/2 + a/2
                z = 0.3
                if abs(x) <= 0.48 and abs(y) <= 0.48 and abs(z) <= 0.48:
                    positions.append([x, y, z])
        
        # Layer 3: same as layer 1 but shifted in z
        for row in range(4):
            y = -0.4 + row * a * np.sqrt(3)/2
            for col in range(4):
                x = -0.4 + col * a + (row % 2) * a/2
                z = 0.0
                if abs(x) <= 0.48 and abs(y) <= 0.48 and abs(z) <= 0.48:
                    positions.append([x, y, z])
        
        positions = np.array(positions, dtype=np.float64)
        
        # Keep only points within the cube
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]
        
        # If more than N, select best N
        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]
        
        # If fewer than N, pad with random
        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
        
        return positions[:N]

    def make_random(seed):
        np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0
    
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_lattice),
        ("hcp", make_hcp_like),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
        ("rand5", lambda: make_random(555)),
    ]
    
    # Phase 1: Global Optimization with Smooth Surrogate (Annealed Beta)
    # Use more iterations with vectorized gradients
    for name, init_func in initializations:
        if remaining() < 12:
            break
            
        try:
            c = init_func()
            c = np.clip(c, -0.499999, 0.499999)
            
            # More gradual beta annealing with more stages
            betas = [15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0]
            
            for beta in betas:
                if remaining() < 5:
                    break
                c = local_optimize_smooth(c, beta, maxiter=250, time_budget=2)
            
            # Phase 2: Direct Polish with tighter tolerance
            if remaining() > 4:
                c = local_optimize_direct(c, maxiter=150, time_budget=4)
            
            r = compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_positions = c.copy()
                
        except Exception:
            continue

    # Phase 3: Perturbations of Best Solution with varied scales
    if best_positions is not None and remaining() > 20:
        for perturb_seed in range(15):
            if remaining() < 6:
                break
            np.random.seed(1000 + perturb_seed)
            # Mix of small and large perturbations
            if perturb_seed % 3 == 0:
                scale = 0.003  # small
            elif perturb_seed % 3 == 1:
                scale = 0.015  # medium
            else:
                scale = 0.04    # large
            c_pert = best_positions + np.random.normal(0, scale, size=best_positions.shape)
            c_pert = np.clip(c_pert, -0.499999, 0.499999)
            
            # Refine perturbed solution
            for beta in [80.0, 200.0, 500.0]:
                if remaining() < 3:
                    break
                c_pert = local_optimize_smooth(c_pert, beta, maxiter=120, time_budget=1.5)
                
            if remaining() > 3:
                c_pert = local_optimize_direct(c_pert, maxiter=80, time_budget=2)
                
            r = compute_radius(c_pert)
            if r > best_radius:
                best_radius = r
                best_positions = c_pert.copy()

    # Final high-precision Polish with alternating smooth and direct
    if best_positions is not None and remaining() > 8:
        # Multiple rounds: smooth then direct for better convergence
        for round_num in range(8):
            if remaining() < 3:
                break
            # One more smooth stage with very high beta
            if remaining() > 5:
                best_positions = local_optimize_smooth(best_positions, beta=3000.0, maxiter=400, time_budget=1.5)
            best_positions = local_optimize_direct(best_positions, maxiter=400, time_budget=2)
            best_positions = np.clip(best_positions, -0.499999, 0.499999)

    if best_positions is None:
        best_positions = make_initial_grid()
        
    return best_positions

# EVOLVE-BLOCK-END
