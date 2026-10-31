"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin for safety

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

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

        # Rotation matrix
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v

        # Reorder so that i0 is at index 0 and i1 is at index 1
        perm = list(range(n))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(n) if i != i0 and i != i1]
        perm[2:] = remaining

        transformed = transformed[perm]
        return transformed, score_points(transformed)

    def continuous_refine(pts, time_left):
        """Refine points using continuous optimization with smooth surrogate objective."""
        if time_left < 5:
            return pts, score_points(pts)

        transformed, score = normalize_and_score(pts)
        if score >= 1e9:
            return pts, score

        # Now transformed[0] is (0,0) and transformed[1] is (1,0)
        init_params = transformed[2:].flatten()

        def smooth_objective(params, beta=100.0):
            """Smooth surrogate using log-sum-exp for soft-min/soft-max."""
            if time.time() > deadline:
                return 1e10
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)

            d = pdist(pts_local, 'euclidean')
            if np.min(d) < 1e-8:
                return 1e10
            
            # Soft-max: approximates max(d)
            max_d = np.log(np.sum(np.exp(beta * d))) / beta
            # Soft-min: approximates min(d) using negative log-sum-exp
            min_d = -np.log(np.sum(np.exp(-beta * d))) / beta
            
            if min_d < 1e-8:
                return 1e10
            return (max_d / min_d) ** 2

        def exact_objective(params):
            """Exact nonsmooth objective for final verification."""
            if time.time() > deadline:
                return 1e10
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)

            d = pdist(pts_local, 'euclidean')
            min_d = np.min(d)
            max_d = np.max(d)
            if min_d < 1e-8:
                return 1e10
            return (max_d / min_d) ** 2

        bnds = [(-5, 5)] * 28

        best_params = init_params.copy()
        best_score = smooth_objective(init_params)

        # Multi-start with increasing beta for annealing effect
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            try:
                def obj_beta(p):
                    return smooth_objective(p, beta)
                res = minimize(obj_beta, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_score:
                    best_params = res.x
                    best_score = res.fun
            except Exception:
                pass

        # Final polish with exact objective
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_score:
                    best_params = res.x
                    best_score = res.fun
            except Exception:
                pass

        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)

        # Verify with exact score
        actual_score = score_points(final_pts)
        if actual_score < best_score:
            best_score = actual_score

        return final_pts, best_score

    # ===== Phase 1: Generate diverse initial configurations =====
    candidates = []

    # 1. Hexagonal lattice - 16 closest points to origin
    R = 5
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
    candidates.append(hex_pts[initial_subset])

    # 2. Concentric circles (2 rings of 8 points)
    ring1_r = 1.0
    ring2_r = 2.0
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circ_pts = np.vstack([
        np.column_stack([ring1_r * np.cos(angles1), ring1_r * np.sin(angles1)]),
        np.column_stack([ring2_r * np.cos(angles2), ring2_r * np.sin(angles2)])
    ])
    candidates.append(circ_pts)

    # 3. Perturbed hexagonal lattice
    rng = np.random.RandomState(42)
    perturbed = hex_pts[initial_subset].copy()
    perturbed += rng.normal(0, 0.1, perturbed.shape)
    candidates.append(perturbed)

    # 4. Grid with perturbation
    grid_pts = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    grid_pts += rng.normal(0, 0.05, grid_pts.shape)
    candidates.append(grid_pts)

    # 5. Another hexagonal configuration (slightly different subset)
    if len(sorted_indices) > N:
        alt_subset = sorted_indices[1:N+1].tolist()
        candidates.append(hex_pts[alt_subset])

    # ===== Phase 2: Local search on hexagonal lattice for best candidate =====
    best_score = 1e10
    best_pts = None

    # Focus local search on the hexagonal lattice candidate (index 0)
    hex_candidate = candidates[0]
    hex_score = score_points(hex_candidate)

    if hex_score < best_score:
        best_score = hex_score
        best_pts = hex_candidate.copy()

    # Local search: swap points in/out of subset
    candidate_indices = sorted_indices[:100].tolist()
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    current_subset = initial_subset.copy()
    current_score = hex_score

    time_for_local_search = 30  # seconds
    local_deadline = start_time + time_for_local_search

    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        for i in range(N):
            if time.time() > local_deadline:
                break
            for j in outside:
                if time.time() > local_deadline:
                    break
                new_subset = current_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    current_subset = new_subset
                    in_subset = set(current_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True

    if current_score < best_score:
        best_score = current_score
        best_pts = hex_pts[current_subset].copy()

    # ===== Phase 3: Refine all candidates with continuous optimization =====
    # Allocate remaining time to refinement
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining < 10:
        return best_pts

    # Refine the best candidate first
    refined_pts, refined_score = continuous_refine(best_pts, remaining * 0.4)
    if refined_score < best_score:
        best_score = refined_score
        best_pts = refined_pts

    # Try refining other candidates if time permits
    for idx in range(1, len(candidates)):
        if time.time() > deadline - 15:
            break
        cand = candidates[idx]
        cand_score = score_points(cand)
        if cand_score < best_score * 1.5:  # Only refine if promising
            refined_pts, refined_score = continuous_refine(cand, (deadline - time.time()) * 0.3)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts

    # ===== Phase 4: Final polish with multi-start Nelder-Mead =====
    if time.time() < deadline - 20:
        # Use the best configuration as a seed for Nelder-Mead
        transformed, _ = normalize_and_score(best_pts)
        init_params = transformed[2:].flatten()

        def nm_objective(params):
            if time.time() > deadline:
                return 1e10
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)
            d = pdist(pts_local, 'euclidean')
            min_d = np.min(d)
            max_d = np.max(d)
            if min_d < 1e-8:
                return 1e10
            return (max_d / min_d) ** 2

        try:
            res = minimize(nm_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                final_pts = np.zeros((N, 2))
                final_pts[0] = [0.0, 0.0]
                final_pts[1] = [1.0, 0.0]
                final_pts[2:] = res.x.reshape(-1, 2)
                actual = score_points(final_pts)
                if actual < best_score:
                    best_score = actual
                    best_pts = final_pts
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
