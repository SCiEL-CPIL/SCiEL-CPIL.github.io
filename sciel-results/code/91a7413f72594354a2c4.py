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
        return transformed, score_points(transformed)

    def refine_with_lbfgs(pts, max_iter=200, ftol=1e-10, gtol=1e-7):
        """Refine points using L-BFGS-B with symmetry reduction."""
        transformed, score = normalize_and_score(pts)
        if score >= 1e9:
            return pts, score

        init_params = transformed[2:].flatten()

        def objective(params):
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

        try:
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            best_params = res.x
            final_score = res.fun
        except Exception:
            best_params = init_params
            final_score = objective(init_params)

        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)

        actual_score = score_points(final_pts)
        if actual_score < final_score:
            final_score = actual_score

        return final_pts, final_score

    def refine_with_powell(pts, max_iter=100, ftol=1e-10):
        """Refine points using Powell method with symmetry reduction."""
        transformed, score = normalize_and_score(pts)
        if score >= 1e9:
            return pts, score

        init_params = transformed[2:].flatten()

        def objective(params):
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
            res = minimize(objective, init_params, method='Powell',
                           options={'maxiter': max_iter, 'ftol': ftol})
            best_params = res.x
            final_score = res.fun
        except Exception:
            best_params = init_params
            final_score = objective(init_params)

        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)

        actual_score = score_points(final_pts)
        if actual_score < final_score:
            final_score = actual_score

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
    ring2_r = 2.0
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
    ring2_r = 1.8
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

    # ===== Score all candidates =====
    scores = [score_points(c) for c in candidates]
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_pts = candidates[best_idx].copy()

    # ===== Local search on hexagonal lattice for best candidate =====
    hex_candidate = candidates[0]
    hex_score = scores[0]

    if hex_score < best_score:
        best_score = hex_score
        best_pts = hex_candidate.copy()

    # Local search: swap points in/out of subset
    candidate_indices = sorted_indices[:120].tolist()
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    current_subset = initial_subset.copy()
    current_score = hex_score

    local_deadline = start_time + 25  # 25 seconds for local search
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

    # ===== Refine top candidates with continuous optimization =====
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining < 10:
        return best_pts

    # Sort candidates by score and refine the best few
    cand_scores = [(score_points(c), i) for i, c in enumerate(candidates)]
    cand_scores.sort()

    # Also include the locally-searched hex result
    hex_local_pts = hex_pts[current_subset].copy() if current_subset != initial_subset else hex_candidate
    hex_local_score = current_score
    if hex_local_score < best_score:
        best_score = hex_local_score
        best_pts = hex_local_pts

    # Refine the best candidates
    num_to_refine = min(5, len(cand_scores))
    for rank in range(num_to_refine):
        if time.time() > deadline - 20:
            break
        s, idx = cand_scores[rank]
        if s > best_score * 1.2:
            continue
        cand = candidates[idx]
        # Try L-BFGS-B first
        refined_pts, refined_score = refine_with_lbfgs(cand, max_iter=150, ftol=1e-10)
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts

        # Then try Powell on the refined result
        if time.time() < deadline - 15:
            refined_pts2, refined_score2 = refine_with_powell(refined_pts, max_iter=80, ftol=1e-10)
            if refined_score2 < best_score:
                best_score = refined_score2
                best_pts = refined_pts2

    # ===== Final polish with multi-start Nelder-Mead =====
    if time.time() < deadline - 25:
        # Use the best configuration as a seed
        transformed, _ = normalize_and_score(best_pts)
        init_params = transformed[2:].flatten()

        def nm_objective(params):
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
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
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

    # ===== Second round of refinement if time permits =====
    if time.time() < deadline - 15:
        # Try a few more random perturbations around the best solution
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            perturbed_best = best_pts + rng.normal(0, 0.05, best_pts.shape)
            refined_pts, refined_score = refine_with_lbfgs(perturbed_best, max_iter=100, ftol=1e-10)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts

    return best_pts

# EVOLVE-BLOCK-END
