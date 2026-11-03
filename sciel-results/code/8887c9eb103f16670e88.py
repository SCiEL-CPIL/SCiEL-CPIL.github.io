"""
Combined approach: Vectorized Smooth Surrogate Optimization for Sphere Packing.
Uses soft-min via log-sum-exp with vectorized gradients for speed.
Implements beta-annealing for global-to-local convergence and a final direct polish.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 142.0  # Safety margin before 150s kill

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
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta
        
        # 3. Gradient Calculation
        weights = np.exp(-beta * g_all - log_sum_exp)
        
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
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
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
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
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
        """Body-centered cubic inspired."""
        positions = []
        a = 0.5
        spacing = (2*a) / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * spacing
                    y = -a + (iy + 0.5) * spacing
                    z = -a + (iz + 0.5) * spacing
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + spacing/2, y + spacing/2, z + spacing/2])
        return np.array(positions[:N])

    def make_fcc_like():
        """FCC-inspired: generate dense FCC lattice, select best N by local density."""
        # Generate FCC lattice points in a larger region
        a = 0.35  # smaller lattice constant for denser packing
        positions = []
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    x = (ix - 1.5) * a
                    y = (iy - 1.5) * a
                    z = (iz - 1.5) * a
                    positions.append([x, y, z])
                    # Face centers of the cubic cell
                    if ix < 3:
                        positions.append([x + a/2, y, z])
                    if iy < 3:
                        positions.append([x, y + a/2, z])
                    if iz < 3:
                        positions.append([x, y, z + a/2])
                    if ix < 3 and iy < 3:
                        positions.append([x + a/2, y + a/2, z])
                    if ix < 3 and iz < 3:
                        positions.append([x + a/2, y, z + a/2])
                    if iy < 3 and iz < 3:
                        positions.append([x, y + a/2, z + a/2])
        
        positions = np.array(positions, dtype=np.float64)
        
        # Keep only points that fit well within the cube
        mask = np.all(np.abs(positions) <= 0.48, axis=1)
        positions = positions[mask]
        
        # If we have more than N, select the N with best local spacing
        if len(positions) > N:
            # Compute pairwise distances and prefer well-separated points
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            # Select points with largest minimum distance
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]
        
        # If we have fewer than N, pad with random points
        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.4, 0.4, size=(N - len(positions), 3))
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
        ("fcc", make_fcc_like),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]
    
    # Phase 1: Global Optimization with Smooth Surrogate (Annealed Beta)
    # Use more iterations with vectorized gradients
    for name, init_func in initializations:
        if remaining() < 15:
            break
            
        try:
            c = init_func()
            c = np.clip(c, -0.499999, 0.499999)
            
            # Anneal beta: start low (smooth) to high (sharp) with more stages
            # Extended schedule with more intermediate steps for better convergence
            betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
            
            for beta in betas:
                if remaining() < 5:
                    break
                # More iterations at lower beta, fewer at higher beta
                if beta < 100:
                    iter_count = 300
                    time_budget = 4
                elif beta < 500:
                    iter_count = 200
                    time_budget = 3
                else:
                    iter_count = 100
                    time_budget = 2
                c = local_optimize_smooth(c, beta, maxiter=iter_count, time_budget=time_budget)
            
            # Phase 2: Direct Polish with more iterations
            if remaining() > 8:
                c = local_optimize_direct(c, maxiter=300, time_budget=6)
            
            r = compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_positions = c.copy()
                
        except Exception:
            continue

    # Phase 3: Perturbations of Best Solution with coordinate-wise moves
    if best_positions is not None and remaining() > 30:
        # Try multiple perturbation strategies
        perturbation_scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
        for i, scale in enumerate(perturbation_scales):
            if remaining() < 10:
                break
            np.random.seed(2000 + i * 7)
            c_pert = best_positions + np.random.normal(0, scale, size=best_positions.shape)
            c_pert = np.clip(c_pert, -0.499999, 0.499999)
            
            # Refine perturbed solution with appropriate beta stages
            if scale < 0.02:
                # Small perturbation: quick polish
                if remaining() > 8:
                    c_pert = local_optimize_smooth(c_pert, beta=200.0, maxiter=150, time_budget=3)
                if remaining() > 5:
                    c_pert = local_optimize_direct(c_pert, maxiter=200, time_budget=4)
            else:
                # Larger perturbation: more annealing needed
                for beta in [50.0, 150.0, 400.0]:
                    if remaining() < 4:
                        break
                    c_pert = local_optimize_smooth(c_pert, beta=beta, maxiter=100, time_budget=2)
                if remaining() > 4:
                    c_pert = local_optimize_direct(c_pert, maxiter=150, time_budget=3)
                
            r = compute_radius(c_pert)
            if r > best_radius:
                best_radius = r
                best_positions = c_pert.copy()

    # Final high-precision Polish with alternating smooth and direct
    if best_positions is not None and remaining() > 15:
        # Multiple rounds: smooth then direct for better convergence
        # Use very high beta for smooth step to mimic hard min closely
        polish_betas = [3000.0, 5000.0, 8000.0]
        
        for round_num, beta in enumerate(polish_betas):
            if remaining() < 6:
                break
            # Smooth polish with high beta
            if remaining() > 8:
                best_positions = local_optimize_smooth(best_positions, beta=beta, maxiter=200, time_budget=4)
            # Direct polish with tight tolerance and more iterations
            if remaining() > 5:
                best_positions = local_optimize_direct(best_positions, maxiter=400, time_budget=5)
            best_positions = np.clip(best_positions, -0.499999, 0.499999)

    if best_positions is None:
        best_positions = make_initial_grid()
        
    return best_positions

# EVOLVE-BLOCK-END
