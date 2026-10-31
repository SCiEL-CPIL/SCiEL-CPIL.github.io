"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin

    def time_left():
        return deadline - time.time()

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        """
        Reduces dimensionality by fixing the first two points (closest pair)
        to (0,0) and (1,0).
        """
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
             return None, None

        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        params = transformed[2:].flatten()
        return params, (i_min, j_min)

    def get_dists_sq_from_params(params):
        """Reconstruct points and return squared pairwise distances."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return dists_sq[iu]

    def exact_objective(params):
        """Exact objective using squared distances."""
        d_sq = get_dists_sq_from_params(params)
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def soft_objective(params, beta):
        """
        Smooth surrogate using log-sum-exp on squared distances.
        Minimizes soft_max_d_sq / soft_min_d_sq.
        """
        if time_left() < 2:
            return 1e10
            
        d_sq = get_dists_sq_from_params(params)
        d_sq = np.maximum(d_sq, 1e-12)
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Soft max via log-sum-exp
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min via log-sum-exp
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def generate_hex_seed():
        """Generate a hexagonal lattice subset of 16 points."""
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_other_seeds():
        """Generate alternative initial configurations."""
        seeds = []
        
        # Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        seeds.append(grid)
        
        # Two concentric rings
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        # Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        # Hexagonal with slight rotation
        hex_base = generate_hex_seed()
        angle = np.pi / 12  # 15 degrees
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        seeds.append(hex_base @ rot.T)
        
        return seeds

    def optimize_seed(seed_pts, phase_budgets):
        """
        Run multi-phase optimization on a seed.
        phase_budgets: list of (method, beta_or_None, maxiter, ftol, gtol, time_budget)
        """
        params, info = prepare_params(seed_pts)
        if params is None:
            return None, 1e10
        
        curr_score = exact_objective(params)
        best_params = params.copy()
        best_score = curr_score
        
        for phase_idx, (method, beta, maxiter, ftol, gtol, time_budget) in enumerate(phase_budgets):
            if time_left() < time_budget + 5:
                break
            
            phase_start = time.time()
            phase_deadline = phase_start + time_budget
            
            if method == 'powell':
                def obj_p(params):
                    if time.time() > phase_deadline:
                        return 1e10
                    return exact_objective(params)
                
                try:
                    res = minimize(obj_p, best_params, method='Powell',
                                   options={'maxiter': maxiter, 'maxfev': maxiter * 10, 'disp': False})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
                    
            elif method == 'lbfgs_soft':
                def obj_s(params):
                    if time.time() > phase_deadline:
                        return 1e10
                    return soft_objective(params, beta)
                
                try:
                    res = minimize(obj_s, best_params, method='L-BFGS-B',
                                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': False})
                    if res.fun < 1e9:
                        exact_s = exact_objective(res.x)
                        if exact_s < best_score:
                            best_score = exact_s
                            best_params = res.x.copy()
                except Exception:
                    pass
        
        return best_params, best_score

    def final_polish(params):
        """
        Final high-precision polish using very high beta and tight tolerances.
        """
        if time_left() < 10:
            return params, exact_objective(params)
        
        polish_deadline = time.time() + min(20, time_left() - 5)
        
        def obj_polish(params):
            if time.time() > polish_deadline:
                return 1e10
            return soft_objective(params, 500.0)
        
        try:
            res = minimize(obj_polish, params, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False})
            if res.fun < 1e9:
                exact_s = exact_objective(res.x)
                return res.x, exact_s
        except Exception:
            pass
        
        return params, exact_objective(params)

    # Generate all seeds
    hex_seed = generate_hex_seed()
    other_seeds = generate_other_seeds()
    
    all_seeds = [hex_seed] + other_seeds
    
    # Add perturbed versions of hex seed
    rng = np.random.default_rng(123)
    for scale in [0.03, 0.08, 0.15]:
        perturbed = hex_seed + rng.normal(0, scale, size=(16, 2))
        all_seeds.append(perturbed)
    
    # Phase configurations: (method, beta, maxiter, ftol, gtol, time_budget)
    # Phase 1: Powell coarse search
    # Phase 2: L-BFGS with low beta (smooth, broad)
    # Phase 3: L-BFGS with medium beta (sharper)
    # Phase 4: L-BFGS with high beta (near-exact)
    phase_configs = [
        ('powell', None, 300, None, None, 15),
        ('lbfgs_soft', 10.0, 500, 1e-8, 1e-5, 15),
        ('lbfgs_soft', 50.0, 1000, 1e-10, 1e-7, 15),
        ('lbfgs_soft', 200.0, 2000, 1e-12, 1e-9, 15),
    ]
    
    best_score_overall = 1e10
    best_params_overall = None
    
    # Process seeds in order of expected quality
    for seed_pts in all_seeds:
        if time_left() < 60:
            break
            
        params, score = optimize_seed(seed_pts, phase_configs)
        
        if params is not None and score < best_score_overall:
            best_score_overall = score
            best_params_overall = params.copy()
    
    # Final polish on the best found
    if best_params_overall is not None:
        best_params_overall, best_score_overall = final_polish(best_params_overall)
    
    # Reconstruct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params_overall.reshape(-1, 2)
    
    # Verify
    final_score = score_points(final_pts)
    
    return final_pts

# EVOLVE-BLOCK-END
