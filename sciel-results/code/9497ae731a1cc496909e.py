"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_dist = min(min_dist, np.min(d) if len(d) > 0 else np.inf)
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r

    def generate_hex_lattice(n_points, radius=1.0, perturbation=0.0):
        """Generate a hexagonal lattice of points inside a circle, with optional perturbation."""
        s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0) / 2.0))
        
        k_max = int(1.0 / s) + 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - s * 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > n_points:
            dists = np.linalg.norm(points, axis=1)
            points = points[np.argsort(dists, kind='stable')[:n_points]]
        
        while len(points) < n_points:
            s *= 0.95
            points_list = []
            k_max = int(1.0 / s) + 2
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 1.0 - s * 0.5:
                        points_list.append([x, y])
            points = np.array(points_list)
            if len(points) > n_points:
                dists = np.linalg.norm(points, axis=1)
                points = points[np.argsort(dists, kind='stable')[:n_points]]
        
        if perturbation > 0 and len(points) == n_points:
            rng = np.random.default_rng(42)
            points += rng.normal(0, perturbation, size=points.shape)
            dists = np.linalg.norm(points, axis=1)
            for i in range(len(points)):
                if dists[i] > 1.0:
                    points[i] *= 1.0 / dists[i]
        
        return points

    def soft_min(x, y, beta):
        """
        Smooth approximation of min(x, y) using log-sum-exp.
        x, y are scalars.
        """
        # min(x, y) = -1/beta * log(exp(-beta*x) + exp(-beta*y))
        # To avoid overflow, shift by min(x,y)
        m = min(x, y)
        return m - (1.0 / beta) * np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m)))

    def create_objective(pts_init, beta):
        """
        Create the objective function for L-BFGS-B.
        Returns negative of the soft-min radius.
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                # Penalty for being outside
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            # Pairwise distances
            # We only need the min pairwise distance
            min_pair_dist = np.inf
            for i in range(N):
                d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
                if len(d) > 0:
                    min_pair_dist = min(min_pair_dist, np.min(d))
            
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            val = soft_min(x, y, beta)
            
            return -val
        
        return objective

    def refine_with_lbfgs(pts, betas=None):
        """
        Refine a configuration using L-BFGS-B with increasing beta.
        """
        if betas is None:
            betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        
        current_pts = pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            objective = create_objective(current_pts, beta)
            
            # L-BFGS-B with bounds
            # Bounds: each coordinate in [-1, 1]
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            try:
                res = minimize(
                    objective,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < -compute_score(current_pts):
                    new_pts = res.x.reshape(-1, 2)
                    # Ensure inside container
                    dists = np.linalg.norm(new_pts, axis=1)
                    for i in range(N):
                        if dists[i] > 1.0:
                            new_pts[i] *= 1.0 / dists[i]
                    
                    if compute_score(new_pts) >= compute_score(current_pts) - 1e-12:
                        current_pts = new_pts
                        
            except Exception:
                pass
        
        return current_pts

    def physics_simulate(pts, max_iters=2000, dt=0.005, cooling_rate=0.9995):
        """Physics-based simulation with repulsive forces between circles and container wall."""
        pts = pts.copy()
        n = len(pts)
        
        # Estimate initial radius for force calculations
        min_dist = np.inf
        for i in range(n):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_dist = min(min_dist, np.min(d) if len(d) > 0 else np.inf)
        r_est = min_dist / 2.0
        
        velocity = np.zeros_like(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces (only for close pairs to save time)
            cutoff = 3.0 * r_est
            for i in range(n):
                diffs = pts[i+1:] - pts[i]
                dists = np.linalg.norm(diffs, axis=1)
                
                mask = dists < cutoff
                if not np.any(mask):
                    continue
                
                close_diffs = diffs[mask]
                close_dists = dists[mask]
                
                k = 0.1
                force_magnitudes = k / (close_dists ** 2 + 1e-10)
                force_vectors = close_diffs / (close_dists[:, np.newaxis] + 1e-10) * force_magnitudes[:, np.newaxis]
                
                forces[i] -= np.sum(force_vectors, axis=0)
                
                for j in range(len(force_vectors)):
                    idx = i + 1 + np.where(mask)[0][j]
                    forces[idx] += force_vectors[j]
            
            # Container wall force
            dists_to_origin = np.linalg.norm(pts, axis=1)
            wall_margin = 0.1
            wall_force_strength = 1.0
            for i in range(n):
                if dists_to_origin[i] > 1.0 - wall_margin:
                    overshoot = dists_to_origin[i] - (1.0 - wall_margin)
                    if overshoot > 0:
                        direction = pts[i] / (dists_to_origin[i] + 1e-10)
                        forces[i] -= direction * wall_force_strength * overshoot
            
            damping = 0.9
            velocity = velocity * damping + forces * dt
            
            max_vel = 0.05
            vel_mags = np.linalg.norm(velocity, axis=1)
            for i in range(n):
                if vel_mags[i] > max_vel:
                    velocity[i] *= max_vel / vel_mags[i]
            
            pts += velocity * dt
            
            dists_to_origin = np.linalg.norm(pts, axis=1)
            for i in range(n):
                if dists_to_origin[i] > 1.0:
                    pts[i] *= 1.0 / dists_to_origin[i]
        
        return pts

    # Strategy 1: Generate hexagonal lattice initialization with perturbation
    hex_points = generate_hex_lattice(N, perturbation=0.01)
    best_score = compute_score(hex_points)
    best_pts = hex_points.copy()
    
    # Strategy 2: Run physics simulation to relax
    if time.time() < deadline - 40:
        sim_pts = physics_simulate(hex_points, max_iters=1500, dt=0.005, cooling_rate=0.9995)
        sim_score = compute_score(sim_pts)
        if sim_score > best_score:
            best_score = sim_score
            best_pts = sim_pts
    
    # Strategy 3: Refine with L-BFGS-B using soft-min objective
    if time.time() < deadline - 15 and best_pts is not None:
        refined_pts = refine_with_lbfgs(best_pts)
        refined_score = compute_score(refined_pts)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Strategy 4: Try multiple perturbations and pick the best
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        for trial in range(2):
            if time.time() > deadline - 20:
                break
            
            # Generate new perturbed lattice
            perturbed = generate_hex_lattice(N, perturbation=0.015)
            perturbed += rng.normal(0, 0.01, size=perturbed.shape)
            dists = np.linalg.norm(perturbed, axis=1)
            for i in range(N):
                if dists[i] > 1.0:
                    perturbed[i] *= 1.0 / dists[i]
            
            # Run physics simulation
            if time.time() < deadline - 25:
                sim_pts = physics_simulate(perturbed, max_iters=1000, dt=0.005, cooling_rate=0.9995)
            else:
                sim_pts = perturbed
            
            # Refine with L-BFGS-B
            if time.time() < deadline - 15:
                refined_pts = refine_with_lbfgs(sim_pts, betas=[20.0, 100.0, 500.0])
            else:
                refined_pts = sim_pts
            
            final_score = compute_score(refined_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_pts = refined_pts
    
    # Final validation
    if best_pts is None or compute_score(best_pts) <= 0:
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing
    
    return best_pts
# EVOLVE-BLOCK-END
