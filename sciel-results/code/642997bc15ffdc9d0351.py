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
    1. Generate multiple high-quality initial guesses (Halton, random restarts).
    2. Refine each using L-BFGS-B with a smoothed log-sum-exp objective.
    3. Polish the best result using Nelder-Mead on the exact non-smooth objective.
    4. Targeted local search: push apart the binding (closest) pair.
    """
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    effective_limit = time_limit - safety_margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances). Fast via cdist."""
        if not np.all(np.isfinite(centres)):
            return 0.0
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 0.0
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances using cdist (much faster than manual broadcasting)
        if N > 1:
            dists = cdist(centres, centres, metric='euclidean')
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)

    def find_binding_pair(centres):
        """Find the pair of spheres with minimum distance and the min boundary point."""
        dists = cdist(centres, centres, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_idx]
        
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        return min_idx, min_pair_dist, min_boundary_idx, min_boundary_val

    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius.
        Objective to minimize: -soft_min(r_i)
        where r_i are the constraints (pairwise/2, boundary).
        """
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        
        # Boundary terms: 0.5 - |c_ik|
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2 using cdist
        dists = cdist(centres, centres, metric='euclidean')
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
        ("halton0", lambda: generate_halton_init(0)),
        ("halton1", lambda: generate_halton_init(1)),
        ("halton2", lambda: generate_halton_init(2)),
        ("halton3", lambda: generate_halton_init(3)),
        ("rand1", lambda: generate_random_init(1)),
        ("rand2", lambda: generate_random_init(2)),
        ("rand3", lambda: generate_random_init(3)),
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
        
        # Stage 1: L-BFGS-B with increasing beta
        for beta in [5.0, 15.0, 40.0, 100.0]:
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
        
        # Stage 2: High-beta polish
        for beta in [300.0, 800.0]:
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
        
        # Stage 3: Nelder-Mead polish
        if not check_time():
            try:
                result = minimize(
                    neg_score,
                    current_centres.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
    
    # Targeted local search: push apart the binding pair
    if best_centres is not None and not check_time():
        rng = np.random.default_rng(99)
        for iteration in range(500):
            if check_time():
                break
            
            # Find the binding constraint
            (i, j), pair_dist, (bi, bk), bnd_val = find_binding_pair(best_centres)
            
            if pair_dist / 2.0 < bnd_val:
                # Pairwise constraint is binding: push i and j apart
                direction = best_centres[i] - best_centres[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    direction = rng.normal(0, 1, size=3)
                    norm = np.linalg.norm(direction)
                direction /= norm
                
                # Try several push magnitudes
                improved = False
                for push in [0.002, 0.005, 0.01, 0.015, 0.02]:
                    trial = best_centres.copy()
                    trial[i] = trial[i] + push * direction
                    trial[j] = trial[j] - push * direction
                    trial = np.clip(trial, -0.495, 0.495)
                    
                    trial_score = score_centres(trial)
                    if trial_score > best_score:
                        best_centres = trial
                        best_score = trial_score
                        improved = True
                        break
                
                if not improved:
                    # Try small random perturbation of one of the binding points
                    for idx in [i, j]:
                        for _ in range(5):
                            trial = best_centres.copy()
                            trial[idx] = trial[idx] + rng.normal(0, 0.01, size=3)
                            trial = np.clip(trial, -0.495, 0.495)
                            trial_score = score_centres(trial)
                            if trial_score > best_score:
                                best_centres = trial
                                best_score = trial_score
                                break
            else:
                # Boundary constraint is binding: push point bi away from boundary
                axis = bk
                sign = np.sign(best_centres[bi, axis])
                if sign == 0:
                    sign = 1.0
                
                improved = False
                for push in [0.002, 0.005, 0.01, 0.015, 0.02]:
                    trial = best_centres.copy()
                    trial[bi, axis] = trial[bi, axis] - push * sign
                    trial = np.clip(trial, -0.495, 0.495)
                    
                    trial_score = score_centres(trial)
                    if trial_score > best_score:
                        best_centres = trial
                        best_score = trial_score
                        improved = True
                        break
                
                if not improved:
                    # Random perturbation of the boundary point
                    for _ in range(5):
                        trial = best_centres.copy()
                        trial[bi] = trial[bi] + rng.normal(0, 0.01, size=3)
                        trial = np.clip(trial, -0.495, 0.495)
                        trial_score = score_centres(trial)
                        if trial_score > best_score:
                            best_centres = trial
                            best_score = trial_score
                            break
    
    # Final validation
    if best_centres is None:
        # Fallback
        best_centres = generate_halton_init(0)
    
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score) or final_score <= 0:
        best_centres = generate_halton_init(0)
    
    return best_centres

# EVOLVE-BLOCK-END
