"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin for safety

    def get_distances(pts):
        """Compute pairwise squared distances."""
        # Using pdist for efficiency
        d = pdist(pts, 'euclidean')
        return d**2

    def raw_score(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d_sq = get_distances(pts)
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        ratio_sq = max_d_sq / min_d_sq
        return ratio_sq

    def smooth_objective(pts, tau):
        """
        Compute smooth approximation of (max_d / min_d)^2 using log-sum-exp.
        d_sq is squared distances.
        soft_max(d_sq) ~ tau * log(sum(exp(d_sq/tau)))
        soft_min(d_sq) ~ -tau * log(sum(exp(-d_sq/tau)))
        """
        d_sq = get_distances(pts)
        
        # Shift for numerical stability
        max_val = np.max(d_sq)
        min_val = np.min(d_sq)
        
        # Soft max
        # log(sum(exp((d_sq - max_val)/tau))) + max_val/tau
        shifted_max = (d_sq - max_val) / tau
        # Clip to avoid underflow in exp, though sum(exp) handles it usually
        log_sum_max = np.log(np.sum(np.exp(shifted_max))) + max_val / tau
        soft_max = tau * log_sum_max
        
        # Soft min
        # log(sum(exp(-(d_sq - min_val)/tau))) - min_val/tau  -> -tau * log_sum + min_val
        # Actually: -tau * ( log(sum(exp(-d_sq/tau))) )
        # = -tau * ( log(sum(exp(-(d_sq - min_val)/tau) * exp(min_val/tau))) )
        # = -tau * ( log(exp(min_val/tau) * sum(exp(-(d_sq - min_val)/tau))) )
        # = -tau * ( min_val/tau + log(sum(exp(-(d_sq - min_val)/tau))) )
        # = -min_val - tau * log(sum(exp(-(d_sq - min_val)/tau)))
        
        shifted_min = -(d_sq - min_val) / tau
        log_sum_min = np.log(np.sum(np.exp(shifted_min)))
        soft_min = min_val - tau * log_sum_min
        
        if soft_min < 1e-12:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def normalize_and_score(pts):
        """Fix first two points to remove translation/rotation, return normalized points and score."""
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(n, 1)
        d_mat = dists[iu]
        min_idx = np.argmin(d_mat)
        pair = np.unravel_index(min_idx, (n, n))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0

        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return pts, 1e10

        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        transformed = (pts - p0) @ R_mat.T / norm_v

        perm = list(range(n))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(n) if i != i0 and i != i1]
        perm[2:] = remaining

        transformed = transformed[perm]
        return transformed, raw_score(transformed)

    def refine_smooth(pts, tau_start=1.0, tau_end=0.01, max_iter=100):
        """
        Refine points using L-BFGS-B with a smooth surrogate objective.
        Anneals tau from tau_start to tau_end.
        """
        transformed, _ = normalize_and_score(pts)
        if np.any(transformed[:, 0] == transformed[:, 1]): # Check for degeneracy
             return pts, 1e10

        init_params = transformed[2:].flatten()

        # We will run L-BFGS-B multiple times with decreasing tau
        # This is a simple annealing schedule
        taus = np.geomspace(tau_start, tau_end, 5)
        
        current_params = init_params
        best_score = 1e10
        best_params = init_params

        for tau in taus:
            def objective(params):
                pts_local = np.zeros((N, 2))
                pts_local[0] = [0.0, 0.0]
                pts_local[1] = [1.0, 0.0]
                pts_local[2:] = params.reshape(-1, 2)
                
                # Check for distinctness
                d_sq = get_distances(pts_local)
                if np.min(d_sq) < 1e-12:
                    return 1e10
                return smooth_objective(pts_local, tau)

            try:
                res = minimize(objective, current_params, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                current_params = res.x
                score = objective(current_params)
                if score < best_score:
                    best_score = score
                    best_params = current_params.copy()
            except Exception:
                pass

        # Reconstruct points
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Final polish with raw objective to snap to exact local min
        final_score = raw_score(final_pts)
        return final_pts, final_score

    def refine_raw(pts, max_iter=100):
        """Refine points using L-BFGS-B with the raw non-smooth objective."""
        transformed, _ = normalize_and_score(pts)
        init_params = transformed[2:].flatten()

        def objective(params):
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)
            return raw_score(pts_local)

        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            best_params = res.x
        except Exception:
            best_params = init_params

        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        final_score = raw_score(final_pts)
        return final_pts, final_score

    # ===== Generate diverse initial configurations =====
    rng = np.random.RandomState(12345)
    candidates = []

    # 1. Hexagonal lattice - 16 closest points to origin
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    candidates.append(hex_pts[initial_subset].copy())

    # 2. Concentric circles (2 rings of 8 points)
    ring1_r = 1.0
    ring2_r = 1.732 # sqrt(3) for hexagonal packing ratio
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circ_pts = np.vstack([
        np.column_stack([ring1_r * np.cos(angles1), ring1_r * np.sin(angles1)]),
        np.column_stack([ring2_r * np.cos(angles2), ring2_r * np.sin(angles2)])
    ])
    candidates.append(circ_pts.copy())

    # 3. Perturbed hexagonal lattice
    perturbed = hex_pts[initial_subset].copy()
    perturbed += rng.normal(0, 0.1, perturbed.shape)
    candidates.append(perturbed)

    # 4. Grid with perturbation
    grid_pts = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    grid_pts += rng.normal(0, 0.05, grid_pts.shape)
    candidates.append(grid_pts)

    # 5. Another hexagonal configuration
    if len(sorted_indices) > N:
        alt_subset = sorted_indices[1:N+1].tolist()
        candidates.append(hex_pts[alt_subset].copy())

    # 6. Concentric circles with different radii
    ring1_r = 1.0
    ring2_r = 1.9
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circ_pts2 = np.vstack([
        np.column_stack([ring1_r * np.cos(angles1), ring1_r * np.sin(angles1)]),
        np.column_stack([ring2_r * np.cos(angles2), ring2_r * np.sin(angles2)])
    ])
    candidates.append(circ_pts2.copy())

    # 7. Ring of 8 + inner ring of 8 (different ratio)
    ring1_r = 0.8
    ring2_r = 2.2
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circ_pts3 = np.vstack([
        np.column_stack([ring1_r * np.cos(angles1), ring1_r * np.sin(angles1)]),
        np.column_stack([ring2_r * np.cos(angles2), ring2_r * np.sin(angles2)])
    ])
    candidates.append(circ_pts3.copy())

    # 8. Perturbed grid
    grid_pts2 = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    grid_pts2 += rng.normal(0, 0.15, grid_pts2.shape)
    candidates.append(grid_pts2)

    # 9. Random initial points
    random_pts = rng.uniform(-2, 2, (N, 2))
    candidates.append(random_pts)

    # 10. Perturbed hexagonal with larger perturbation
    perturbed2 = hex_pts[initial_subset].copy()
    perturbed2 += rng.normal(0, 0.2, perturbed2.shape)
    candidates.append(perturbed2)
    
    # 11. 4x4 grid but slightly rotated
    grid_rot = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    theta = np.pi / 16
    R_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    grid_rot = grid_rot @ R_rot.T
    candidates.append(grid_rot)

    # ===== Score all candidates =====
    scores = [raw_score(c) for c in candidates]
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_pts = candidates[best_idx].copy()

    # ===== Refine top candidates with continuous optimization =====
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining < 20:
        return best_pts

    # Sort candidates by score
    cand_indices = np.argsort(scores)
    
    # We will refine the top 6 candidates
    num_to_refine = min(6, len(cand_indices))
    
    for rank in range(num_to_refine):
        if time.time() > deadline - 10:
            break
        idx = cand_indices[rank]
        cand = candidates[idx]
        
        # 1. Smooth refinement (Annealed L-BFGS-B on soft objective)
        # This helps escape local minima of the raw objective
        refined_pts, refined_score = refine_smooth(cand, tau_start=0.5, tau_end=0.05, max_iter=50)
        
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts
            
        # 2. Raw refinement (L-BFGS-B on raw objective) starting from smoothed result
        # This snaps to the exact local minimum
        if time.time() < deadline - 5:
            refined_pts2, refined_score2 = refine_raw(refined_pts, max_iter=50)
            if refined_score2 < best_score:
                best_score = refined_score2
                best_pts = refined_pts2

    # ===== Final Polish =====
    if time.time() < deadline - 5:
        # Try a few random perturbations around the best solution and refine
        for attempt in range(3):
            if time.time() > deadline - 2:
                break
            perturbed_best = best_pts + rng.normal(0, 0.05, best_pts.shape)
            # Use smooth then raw
            p1, s1 = refine_smooth(perturbed_best, tau_start=0.2, tau_end=0.02, max_iter=30)
            p2, s2 = refine_raw(p1, max_iter=30)
            if s2 < best_score:
                best_score = s2
                best_pts = p2

    return best_pts

# EVOLVE-BLOCK-END
