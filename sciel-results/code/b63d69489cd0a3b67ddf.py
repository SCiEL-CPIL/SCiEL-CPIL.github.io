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

    def score_points_fast(pts):
        """Faster scoring using vectorized ops."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
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

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)

    # Local search: try swapping one point in the subset with one outside
    candidate_indices = sorted_indices[:120].tolist()

    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]

    improved = True
    while improved and time.time() < deadline - 45:
        improved = False
        for i in range(N):
            if time.time() > deadline - 45:
                break
            for j in outside:
                if time.time() > deadline - 45:
                    break
                new_subset = initial_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points_fast(new_pts)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    initial_subset = new_subset
                    in_subset = set(initial_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True

    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)

    # ===== Phase 1b: Low-rank sinusoidal deformation of a 4x4 grid =====
    # This allows global warping that equalizes pairwise distances
    def sinusoidal_grid_objective(params):
        """
        params: [a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4]
        12 parameters for low-rank sinusoidal deformation of 4x4 grid
        """
        if time.time() > deadline:
            return 1e10
        
        # Extract parameters
        a = params[0:4]  # x amplitudes
        b = params[4:8]  # x phases
        c = params[8:12]  # y amplitudes (we'll use these for y phases)
        
        # Create 4x4 grid
        grid_x, grid_y = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
        
        # Apply sinusoidal deformation
        # x' = x + a_i * sin(b_i * y)
        # y' = y + c_i * sin(b_i * x)
        deformed_pts = []
        for i in range(4):
            for j in range(4):
                x = grid_x[j, i]
                y = grid_y[j, i]
                x_new = x + a[0] * np.sin(b[0] * y) + a[1] * np.sin(b[1] * x)
                y_new = y + c[0] * np.sin(b[0] * x) + c[1] * np.sin(b[1] * y)
                deformed_pts.append([x_new, y_new])
        
        pts = np.array(deformed_pts)
        return score_points_fast(pts)

    # Try optimizing the sinusoidal grid
    try:
        if time.time() < deadline - 35:
            initial_params = np.zeros(12)
            bounds = [(-0.5, 0.5)] * 12
            
            res = minimize(sinusoidal_grid_objective, initial_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-8})
            
            if res.fun < best_score:
                # Reconstruct the deformed grid
                a = res.x[0:4]
                b = res.x[4:8]
                c = res.x[8:12]
                
                grid_x, grid_y = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
                deformed_pts = []
                for i in range(4):
                    for j in range(4):
                        x = grid_x[j, i]
                        y = grid_y[j, i]
                        x_new = x + a[0] * np.sin(b[0] * y) + a[1] * np.sin(b[1] * x)
                        y_new = y + c[0] * np.sin(b[0] * x) + c[1] * np.sin(b[1] * y)
                        deformed_pts.append([x_new, y_new])
                
                deformed_pts = np.array(deformed_pts)
                deformed_score = score_points(deformed_pts)
                
                if deformed_score < best_score:
                    best_pts = deformed_pts
                    best_score = deformed_score
                    
    except Exception:
        pass

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
        if time.time() < deadline - 35:
            for beta_val in [20.0, 40.0, 80.0, 160.0, 320.0]:
                if time.time() > deadline - 35:
                    break
                res = minimize(soft_objective, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
                               args=(beta_val,))

                candidate_score = exact_objective(res.x)
                if candidate_score < best_exact_score:
                    best_params = res.x
                    best_exact_score = candidate_score

    except Exception:
        pass

    # Phase 2b: Polish with exact objective using Nelder-Mead for nonsmooth handling
    try:
        if time.time() < deadline - 20:
            # First L-BFGS-B polish
            res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            candidate_score = exact_objective(res2.x)
            if candidate_score < best_exact_score:
                best_params = res2.x
                best_exact_score = candidate_score

            # Then Nelder-Mead for final refinement on the nonsmooth objective
            if time.time() < deadline - 15:
                res_nm = minimize(exact_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-10})
                nm_score = exact_objective(res_nm.x)
                if nm_score < best_exact_score:
                    best_params = res_nm.x
                    best_exact_score = nm_score

    except Exception:
        pass

    # Phase 2c: Multiple restarts with perturbations + gradient flow
    try:
        if time.time() < deadline - 12:
            rng = np.random.default_rng(12345)
            n_restarts = 12
            for restart in range(n_restarts):
                if time.time() > deadline - 12:
                    break

                # Vary perturbation size more aggressively
                scale = 0.003 + 0.02 * (restart / n_restarts)
                perturbation = rng.normal(0, scale, size=28)
                perturbed_params = best_params + perturbation
                perturbed_params = np.clip(perturbed_params, -5, 5)

                res3 = minimize(exact_objective, perturbed_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})

                candidate_score = exact_objective(res3.x)
                if candidate_score < best_exact_score:
                    best_params = res3.x
                    best_exact_score = candidate_score
                
                # Also try gradient flow from this perturbed start
                if time.time() < deadline - 12:
                    gf_p, gf_v = _grad_flow(res3.x, n_steps=100, lr=0.002, beta=50.0)
                    gf_s = exact_objective(gf_p)
                    if gf_s < best_exact_score:
                        best_params = gf_p
                        best_exact_score = gf_s

    except Exception:
        pass

    # Phase 2d: Differential evolution for global exploration (reduced budget)
    try:
        if time.time() < deadline - 8:
            bounds_de = [(-4, 4)] * 28
            res_de = differential_evolution(exact_objective, bounds_de, maxiter=30,
                                            popsize=12, seed=777, tol=1e-10,
                                            polish=True, init='halton')
            if res_de.fun < best_exact_score:
                candidate = res_de.x
                candidate_score = exact_objective(candidate)
                if candidate_score < best_exact_score:
                    best_params = candidate
                    best_exact_score = candidate_score
    except Exception:
        pass

    # Phase 2e: Final high-precision polish with gradient flow
    try:
        if time.time() < deadline - 4:
            # L-BFGS-B with very tight tolerance
            res4 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12})
            lbfgs_score = exact_objective(res4.x)
            if lbfgs_score < best_exact_score:
                best_params = res4.x
                best_exact_score = lbfgs_score

            # Gradient flow final polish
            if time.time() < deadline - 2:
                gf_final, gf_final_val = _grad_flow(best_params, n_steps=150, lr=0.001, beta=100.0)
                gf_final_score = exact_objective(gf_final)
                if gf_final_score < best_exact_score:
                    best_params = gf_final
                    best_exact_score = gf_final_score

            # Final Nelder-Mead refinement
            if time.time() < deadline - 1:
                res_nm2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-12})
                nm2_score = exact_objective(res_nm2.x)
                if nm2_score < best_exact_score:
                    best_params = res_nm2.x
                    best_exact_score = nm2_score

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
