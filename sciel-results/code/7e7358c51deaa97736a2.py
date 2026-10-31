"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # ===== Phase 1: Hexagonal lattice initialization =====
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)

    def score_points(pts):
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)

    # Local search: try swapping one point in the subset with one outside
    candidate_indices = sorted_indices[:150].tolist()

    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]

    improved = True
    while improved and time.time() < deadline - 40:
        improved = False
        for i in range(N):
            if time.time() > deadline - 40:
                break
            for j in outside:
                if time.time() > deadline - 40:
                    break
                new_subset = initial_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    initial_subset = new_subset
                    in_subset = set(initial_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True

    # 2-swap local search
    if time.time() < deadline - 35:
        improved = True
        while improved and time.time() < deadline - 35:
            improved = False
            for i in range(N):
                if time.time() > deadline - 35:
                    break
                for j in range(i+1, N):
                    if time.time() > deadline - 35:
                        break
                    for k in outside:
                        if time.time() > deadline - 35:
                            break
                        for l in outside:
                            if l <= k:
                                continue
                            new_subset = initial_subset.copy()
                            new_subset[i] = k
                            new_subset[j] = l
                            new_pts = hex_pts[new_subset]
                            new_score = score_points(new_pts)
                            if new_score < best_score - 1e-9:
                                best_score = new_score
                                initial_subset = new_subset
                                in_subset = set(initial_subset)
                                outside = [m for m in candidate_indices if m not in in_subset]
                                improved = True

    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)

    # ===== Phase 2: Continuous optimization with symmetry reduction =====
    # Fix first two points to remove translation and rotation invariance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]

    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair[0], pair[1]
    if i0 > i1:
        i0, i1 = i1, i0

    p0 = best_pts[i0]
    p1 = best_pts[i1]

    v = p1 - p0
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-8:
        return best_pts

    # Rotation matrix
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    # Transform all points
    transformed = (best_pts - p0) @ R_mat.T / norm_v

    # Reorder so that i0 is at index 0 and i1 is at index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining

    transformed = transformed[perm]

    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    init_params = transformed[2:].flatten()

    def soft_objective(params, beta=100.0):
        """Smooth surrogate using log-sum-exp for max and -log-sum-exp for min."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]

        # Avoid zero distances
        d_sq = np.maximum(d_sq, 1e-12)

        # Soft max of distances (in squared space)
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))

        # Soft min of distances (in squared space)
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))

        if soft_min_d_sq < 1e-8:
            return 1e10

        return soft_max_d_sq / soft_min_d_sq

    def exact_objective(params):
        """Exact objective for final evaluation."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)

        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)

        if min_d < 1e-8:
            return 1e10

        return (max_d / min_d) ** 2

    bnds = [(-5, 5)] * 28

    best_params = init_params.copy()
    best_exact_score = exact_objective(init_params)

    # Phase 2a: Optimize smooth surrogate with adaptive beta
    try:
        if time.time() < deadline - 30:
            for beta_val in [20.0, 40.0, 80.0, 160.0, 320.0]:
                if time.time() > deadline - 30:
                    break
                res = minimize(soft_objective, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12},
                               args=(beta_val,))

                if res.fun < best_exact_score:
                    candidate = res.x
                    candidate_score = exact_objective(candidate)
                    if candidate_score < best_exact_score:
                        best_params = candidate
                        best_exact_score = candidate_score

    except Exception:
        pass

    # Phase 2b: Polish with exact objective
    try:
        if time.time() < deadline - 15:
            res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})

            if res2.fun < best_exact_score:
                candidate = res2.x
                candidate_score = exact_objective(candidate)
                if candidate_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = candidate_score

    except Exception:
        pass

    # Phase 2c: Multiple restarts with perturbations
    try:
        if time.time() < deadline - 8:
            rng = np.random.default_rng(12345)
            n_restarts = 20
            for restart in range(n_restarts):
                if time.time() > deadline - 8:
                    break

                # Vary perturbation size
                scale = 0.003 + 0.015 * (restart / n_restarts)
                perturbation = rng.normal(0, scale, size=28)
                perturbed_params = best_params + perturbation
                perturbed_params = np.clip(perturbed_params, -5, 5)

                res3 = minimize(exact_objective, perturbed_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13})

                if res3.fun < best_exact_score:
                    candidate = res3.x
                    candidate_score = exact_objective(candidate)
                    if candidate_score < best_exact_score:
                        best_params = candidate
                        best_exact_score = candidate_score

    except Exception:
        pass

    # Phase 2d: Differential evolution for global exploration
    try:
        if time.time() < deadline - 10:
            bounds_de = [(-4, 4)] * 28
            res_de = differential_evolution(exact_objective, bounds_de, maxiter=80,
                                            popsize=20, seed=777, tol=1e-12,
                                            polish=True, init='halton')
            if res_de.fun < best_exact_score:
                candidate = res_de.x
                candidate_score = exact_objective(candidate)
                if candidate_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = candidate_score
    except Exception:
        pass

    # Phase 2e: Final high-precision polish
    try:
        if time.time() < deadline - 2:
            # First, a very tight L-BFGS-B pass
            res4 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15})

            if res4.fun < best_exact_score:
                candidate = res4.x
                candidate_score = exact_objective(candidate)
                if candidate_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = candidate_score

            # Second, try Nelder-Mead for final local refinement
            if time.time() < deadline - 1:
                res_nm = minimize(exact_objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16,
                                           'adaptive': True})
                if res_nm.fun < best_exact_score:
                    candidate = res_nm.x
                    candidate_score = exact_objective(candidate)
                    if candidate_score < best_exact_score:
                        best_params = candidate
                        best_exact_score = candidate_score

    except Exception:
        pass

    # Reconstruct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)

    # Verify final score and return best
    final_score = score_points(final_pts)
    if final_score < best_score:
        return final_pts
    else:
        return best_pts

# EVOLVE-BLOCK-END
