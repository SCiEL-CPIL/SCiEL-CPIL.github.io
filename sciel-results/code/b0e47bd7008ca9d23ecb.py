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
    1. Generate multiple high-quality initial guesses (Fibonacci lattice, Halton, random).
    2. Refine each using L-BFGS-B with a quadratic sum-of-squared objective (smoother landscape).
    3. Then polish with soft-min log-sum-exp at increasing beta.
    4. Final Nelder-Mead on exact non-smooth objective.
    5. Targeted local search on binding constraints.
    """
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    effective_limit = time_limit - safety_margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        if not np.all(np.isfinite(centres)):
            return 0.0
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 0.0
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)
    
    def get_binding_info(centres):
        """Return indices of binding constraints (pairwise and boundary)."""
        if not np.all(np.isfinite(centres)):
            return [], []
        
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_pair_half = min_pair / 2.0
        
        r = min(min_boundary, min_pair_half)
        tol = r * 0.01 + 1e-8
        
        # Binding boundary constraints: (point_idx, coord_idx)
        binding_boundary = []
        for i in range(N):
            for k in range(3):
                if boundary_dists[i, k] < r + tol:
                    binding_boundary.append((i, k))
        
        # Binding pairwise constraints: (i, j)
        binding_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] / 2.0 < r + tol:
                    binding_pairs.append((i, j))
        
        return binding_pairs, binding_boundary
    
    def quadratic_obj(centres_flat):
        """
        Objective to minimize: negative sum of squared pairwise distances 
        plus negative sum of squared boundary clearances.
        This creates a smoother landscape than log-sum-exp.
        """
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        
        # Sum of squared pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, 0)
        sum_sq_pair = np.sum(sq_dists)
        
        # Sum of squared boundary clearances
        boundary_clearances = 0.5 - np.abs(centres)
        sum_sq_boundary = np.sum(boundary_clearances**2)
        
        # We want to maximize both, so minimize their negative
        return -(sum_sq_pair + 2.0 * sum_sq_boundary)
    
    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius via log-sum-exp.
        """
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        
        # Boundary terms: 0.5 - |c_ik|
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min: -1/beta * log(sum(exp(-beta * x)))
        max_term = np.max(all_terms)
        shifted = -beta * (all_terms - max_term)
        shifted = np.clip(shifted, -500, 500)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0 or not np.isfinite(sum_exp):
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val

    def neg_score(centres_flat):
        """Negative exact score for direct methods."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 1e10
        return -score_centres(centres)
    
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
    
    def generate_halton_init(seed_offset=0):
        """Generate a Halton-based initial configuration."""
        bases = [(2, 3, 5), (3, 5, 7), (5, 7, 11), (2, 5, 7)]
        b = bases[seed_offset % len(bases)]
        h1 = halton_sequence(N, b[0])
        h2 = halton_sequence(N, b[1])
        h3 = halton_sequence(N, b[2])
        
        scale = 0.9
        centres = np.column_stack((
            h1 * scale - scale/2,
            h2 * scale - scale/2,
            h3 * scale - scale/2
        ))
        
        rng = np.random.default_rng(42 + seed_offset)
        centres += rng.normal(0, 0.015, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    def generate_fibonacci_init(seed_offset=0):
        """Generate a Fibonacci lattice-based initial configuration for maximal separation."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi
        
        n = N
        i = np.arange(n)
        
        x = (i + 0.5) / n
        y = np.mod(i * inv_phi, 1.0)
        z = np.mod(i * inv_phi**2, 1.0)
        
        centres = np.column_stack((x, y, z))
        
        scale = 0.95
        centres = centres * scale - scale/2
        
        rng = np.random.default_rng(42 + seed_offset)
        centres += rng.normal(0, 0.01, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    def generate_random_init(seed):
        """Generate a random initial configuration."""
        rng = np.random.default_rng(seed)
        centres = rng.uniform(-0.45, 0.45, size=(N, 3))
        return centres
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_centres = None
    best_score = 0.0
    
    # Try multiple initializations
    initializers = [
        ("fib0", lambda: generate_fibonacci_init(0)),
        ("fib1", lambda: generate_fibonacci_init(1)),
        ("halton0", lambda: generate_halton_init(0)),
        ("halton1", lambda: generate_halton_init(1)),
        ("halton2", lambda: generate_halton_init(2)),
        ("rand1", lambda: generate_random_init(1)),
        ("rand2", lambda: generate_random_init(2)),
    ]
    
    for name, gen_func in initializers:
        if check_time():
            break
        
        try:
            centres = gen_func()
        except Exception:
            continue
        
        current_score = score_centres(centres)
        current_centres = centres.copy()
        
        # Stage 1: Quadratic objective - smooth landscape for global exploration
        if not check_time():
            try:
                result = minimize(
                    quadratic_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                new_centres = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centres)):
                    new_score = score_centres(new_centres)
                    if new_score > current_score:
                        current_centres = new_centres
                        current_score = new_score
            except Exception:
                pass
        
        # Stage 2: Soft-min with increasing beta
        for beta in [10.0, 30.0, 80.0, 200.0]:
            if check_time():
                break
            
            def smooth_obj(cf, b=beta):
                return soft_min_score(cf, beta=b)
            
            try:
                result = minimize(
                    smooth_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
                )
            except Exception:
                continue
            
            new_centres = result.x.reshape(N, 3)
            if not np.all(np.isfinite(new_centres)):
                continue
            new_score = score_centres(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
        
        # Stage 3: High-beta polish
        for beta in [500.0, 1000.0]:
            if check_time():
                break
            
            def smooth_obj_high(cf, b=beta):
                return soft_min_score(cf, beta=b)
            
            try:
                result = minimize(
                    smooth_obj_high,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 15}
                )
            except Exception:
                continue
            
            new_centres = result.x.reshape(N, 3)
            if not np.all(np.isfinite(new_centres)):
                continue
            new_score = score_centres(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
        
        # Stage 4: Nelder-Mead polish on exact objective
        if not check_time():
            try:
                result = minimize(
                    neg_score,
                    current_centres.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                )
                new_centres = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centres)):
                    new_score = score_centres(new_centres)
                    if new_score > current_score:
                        current_centres = new_centres
                        current_score = new_score
            except Exception:
                pass
        
        if current_score > best_score:
            best_centres = current_centres.copy()
            best_score = current_score
    
    # Targeted local search: focus on binding constraints
    if best_centres is not None and not check_time():
        rng = np.random.default_rng(99)
        for iteration in range(500):
            if check_time():
                break
            
            # Identify binding points
            binding_pairs, binding_boundary = get_binding_info(best_centres)
            
            # Build set of points involved in binding constraints
            binding_points = set()
            for i, j in binding_pairs:
                binding_points.add(i)
                binding_points.add(j)
            for i, k in binding_boundary:
                binding_points.add(i)
            
            # Preferentially pick from binding points
            if binding_points and rng.random() < 0.7:
                idx = list(binding_points)[rng.integers(0, len(binding_points))]
            else:
                idx = rng.integers(0, N)
            
            # Try multiple perturbations for this point
            improved = False
            for attempt in range(5):
                perturb_scale = 0.015 * (1.0 + 0.5 * attempt)
                new_point = best_centres[idx] + rng.normal(0, perturb_scale, size=3)
                new_point = np.clip(new_point, -0.495, 0.495)
                
                trial = best_centres.copy()
                trial[idx] = new_point
                
                trial_score = score_centres(trial)
                if trial_score > best_score + 1e-12:
                    best_centres = trial
                    best_score = trial_score
                    improved = True
                    break
            
            # If no improvement, try a coordinated move of a binding pair
            if not improved and binding_pairs and rng.random() < 0.3:
                i, j = binding_pairs[rng.integers(0, len(binding_pairs))]
                direction = best_centres[i] - best_centres[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    step = 0.005
                    trial = best_centres.copy()
                    trial[i] = np.clip(trial[i] + step * direction, -0.495, 0.495)
                    trial[j] = np.clip(trial[j] - step * direction, -0.495, 0.495)
                    trial_score = score_centres(trial)
                    if trial_score > best_score + 1e-12:
                        best_centres = trial
                        best_score = trial_score
    
    # Final high-precision polish
    if best_centres is not None and not check_time():
        try:
            result = minimize(
                neg_score,
                best_centres.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            new_centres = result.x.reshape(N, 3)
            if np.all(np.isfinite(new_centres)):
                new_score = score_centres(new_centres)
                if new_score > best_score:
                    best_centres = new_centres
                    best_score = new_score
        except Exception:
            pass
    
    # Final validation
    if best_centres is None:
        best_centres = generate_fibonacci_init(0)
    
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score) or final_score <= 0:
        best_centres = generate_fibonacci_init(0)
    
    return best_centres

# EVOLVE-BLOCK-END
