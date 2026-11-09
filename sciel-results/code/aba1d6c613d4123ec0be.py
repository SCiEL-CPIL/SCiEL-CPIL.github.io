"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Strategy:
    1. Generate multiple diverse initial guesses.
    2. Run repulsion-based pre-optimization.
    3. Refine using L-BFGS-B with smoothed log-sum-exp objective (increasing beta).
    4. Polish using Nelder-Mead on the exact non-smooth objective.
    5. Local search with coordinate descent and perturbations.
    6. Final high-precision polish pass.
    """
    start_time = time.time()
    time_limit = 148.0
    margin = 3.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)
    
    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius using log-sum-exp.
        """
        centres = centres_flat.reshape(N, 3)
        
        boundary_terms = 0.5 - np.abs(centres)
        
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val
    
    def neg_score(centres_flat):
        """Negative exact score for direct methods."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        return -score_centres(centres)
    
    def repulsion_objective(centres_flat, k=8.0):
        """
        Repulsion-based objective to spread points apart.
        """
        centres = centres_flat.reshape(N, 3)
        
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        repulsion = -np.sum(1.0 / (pair_dists ** k))
        
        boundary_dists = 0.5 - np.abs(centres)
        boundary_penalty = -np.sum(1.0 / (boundary_dists + 0.02) ** 2)
        
        return repulsion + boundary_penalty
    
    def fibonacci_lattice(n_points, seed=0):
        """Generate n_points using Fibonacci lattice in the unit cube."""
        rng = np.random.default_rng(seed)
        golden_ratio = (1 + np.sqrt(5)) / 2
        points = []
        for i in range(n_points):
            t = i / n_points
            z = 1 - 2 * t
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            point = np.array([x, y, z]) * 0.42
            point += rng.normal(0, 0.008, size=(3,))
            point = np.clip(point, -0.5, 0.5)
            points.append(point)
        return np.array(points)
    
    def halton_sequence(n, base):
        """Generate n points in the Halton sequence with given base."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                x += f * (i_val % base)
                i_val //= base
            result.append(x)
        return np.array(result)
    
    def generate_jittered_grid(n_points, seed=0, grid_size=4):
        """Generate a jittered grid of n_points in [-0.5, 0.5]^3."""
        rng = np.random.default_rng(seed)
        coords = np.linspace(-0.44, 0.44, grid_size)
        grid_points = np.array([(x, y, z) for x in coords for y in coords for z in coords])
        
        jitter = rng.uniform(-0.025, 0.025, size=(len(grid_points), 3))
        grid_points = grid_points + jitter
        
        indices = rng.permutation(len(grid_points))[:n_points]
        return grid_points[indices]
    
    def generate_random_points(n_points, seed=0):
        """Generate random points in [-0.44, 0.44]^3."""
        rng = np.random.default_rng(seed)
        return rng.uniform(-0.44, 0.44, size=(n_points, 3))
    
    def coordinate_descent(centres, max_iter=200, step_scale=0.01):
        """
        Coordinate descent: try small moves in each coordinate for each point.
        """
        current = centres.copy()
        current_score = score_centres(current)
        
        rng = np.random.default_rng(123)
        
        for iteration in range(max_iter):
            if check_time():
                break
            
            improved = False
            for i in range(N):
                for dim in range(3):
                    for direction in [-1, 1]:
                        original = current[i, dim]
                        for scale in [step_scale, step_scale * 0.5, step_scale * 0.25]:
                            new_val = original + direction * scale
                            if new_val < -0.5 or new_val > 0.5:
                                continue
                            current[i, dim] = new_val
                            new_score = score_centres(current)
                            if new_score > current_score + 1e-12:
                                current_score = new_score
                                improved = True
                                break
                        if not improved:
                            current[i, dim] = original
                        else:
                            break
            if not improved:
                break
        
        return current
    
    # Generate multiple diverse initial guesses
    rng_main = np.random.default_rng(42)
    
    initial1 = fibonacci_lattice(N, seed=1)
    
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    scale = 0.88
    initial2 = np.column_stack((
        h1 * scale - scale/2,
        h2 * scale - scale/2,
        h3 * scale - scale/2
    ))
    initial2 += rng_main.normal(0, 0.008, size=(N, 3))
    initial2 = np.clip(initial2, -0.5, 0.5)
    
    initial3 = generate_jittered_grid(N, seed=1, grid_size=4)
    
    h1b = halton_sequence(N, 3)
    h2b = halton_sequence(N, 5)
    h3b = halton_sequence(N, 7)
    initial4 = np.column_stack((
        h1b * scale - scale/2,
        h2b * scale - scale/2,
        h3b * scale - scale/2
    ))
    initial4 += rng_main.normal(0, 0.008, size=(N, 3))
    initial4 = np.clip(initial4, -0.5, 0.5)
    
    initial5 = generate_random_points(N, seed=2)
    
    initial6 = generate_jittered_grid(N, seed=3, grid_size=5)
    
    initial7 = fibonacci_lattice(N, seed=7)
    
    # Additional Halton with different bases
    h1c = halton_sequence(N, 2)
    h2c = halton_sequence(N, 5)
    h3c = halton_sequence(N, 11)
    initial8 = np.column_stack((
        h1c * scale - scale/2,
        h2c * scale - scale/2,
        h3c * scale - scale/2
    ))
    initial8 += rng_main.normal(0, 0.008, size=(N, 3))
    initial8 = np.clip(initial8, -0.5, 0.5)
    
    initial_guesses = [initial1, initial2, initial3, initial4, initial5, initial6, initial7, initial8]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_centres = None
    best_score = -np.inf
    
    # Process each initial guess
    for guess_idx, initial_centres in enumerate(initial_guesses):
        if check_time():
            break
        
        current_centres = initial_centres.copy()
        
        # Stage 0: Repulsion pre-optimization
        try:
            result_rep = minimize(
                repulsion_objective,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            if np.all(np.isfinite(result_rep.x)):
                new_c = result_rep.x.reshape(N, 3)
                new_c = np.clip(new_c, -0.5, 0.5)
                if score_centres(new_c) >= score_centres(current_centres) - 1e-10:
                    current_centres = new_c
        except Exception:
            pass
        
        # Stage 1: Smooth L-BFGS-B with increasing beta
        try:
            betas = [5.0, 20.0, 50.0, 150.0, 400.0]
            
            for beta in betas:
                if check_time():
                    break
                
                def smooth_obj(centres_flat, b=beta):
                    return soft_min_score(centres_flat, beta=b)
                
                result = minimize(
                    smooth_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 250, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 20}
                )
                
                if np.all(np.isfinite(result.x)):
                    new_centres = result.x.reshape(N, 3)
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    new_score = score_centres(new_centres)
                    if new_score >= score_centres(current_centres) - 1e-10:
                        current_centres = new_centres
                        
        except Exception:
            pass
        
        # Stage 2: High Beta Polish
        try:
            for beta in [800.0, 2000.0, 5000.0]:
                if check_time():
                    break
                
                def smooth_obj_high(centres_flat, b=beta):
                    return soft_min_score(centres_flat, beta=b)
                
                result = minimize(
                    smooth_obj_high,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
                )
                
                if np.all(np.isfinite(result.x)):
                    new_centres = result.x.reshape(N, 3)
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    new_score = score_centres(new_centres)
                    if new_score > score_centres(current_centres):
                        current_centres = new_centres
                        
        except Exception:
            pass
        
        # Stage 3: Nelder-Mead polish
        try:
            result_nm = minimize(
                neg_score,
                current_centres.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            
            if np.all(np.isfinite(result_nm.x)):
                new_centres = result_nm.x.reshape(N, 3)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_score = score_centres(new_centres)
                
                if new_score > score_centres(current_centres):
                    current_centres = new_centres
                    
        except Exception:
            pass
        
        # Stage 4: Coordinate descent local search
        if not check_time():
            try:
                current_centres = coordinate_descent(current_centres, max_iter=100, step_scale=0.008)
            except Exception:
                pass
        
        # Stage 5: Perturbation-based local search
        if not check_time():
            try:
                local_rng = np.random.default_rng(42 + guess_idx)
                current_score = score_centres(current_centres)
                
                for iteration in range(60):
                    if check_time():
                        break
                    
                    idx = local_rng.integers(N)
                    original = current_centres[idx].copy()
                    
                    perturbation = local_rng.normal(0, 0.004, size=(3,))
                    current_centres[idx] = np.clip(original + perturbation, -0.5, 0.5)
                    
                    new_score = score_centres(current_centres)
                    
                    if new_score > current_score:
                        current_score = new_score
                    else:
                        current_centres[idx] = original
                        
            except Exception:
                pass
        
        # Update best
        final_score = score_centres(current_centres)
        if np.isfinite(final_score) and final_score > best_score:
            best_score = final_score
            best_centres = current_centres.copy()
    
    # If we didn't get a valid result, use the first initial guess
    if best_centres is None:
        best_centres = initial1.copy()
        best_score = score_centres(best_centres)
    
    # Final coordinate descent on best
    if not check_time():
        try:
            best_centres = coordinate_descent(best_centres, max_iter=80, step_scale=0.005)
            best_score = score_centres(best_centres)
        except Exception:
            pass
    
    # Final Nelder-Mead polish on the best result
    if not check_time():
        try:
            result_final = minimize(
                neg_score,
                best_centres.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if np.all(np.isfinite(result_final.x)):
                new_centres = result_final.x.reshape(N, 3)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_score = score_centres(new_centres)
                
                if new_score > best_score:
                    best_centres = new_centres.copy()
                    best_score = new_score
                    
        except Exception:
            pass
    
    # Final validation
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score):
        best_centres = initial1.copy()
    
    return best_centres

# EVOLVE-BLOCK-END
