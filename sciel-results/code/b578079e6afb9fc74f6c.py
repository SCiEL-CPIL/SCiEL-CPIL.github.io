"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the max common radius for given centers."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        # Distance to boundary: square is [-0.5, 0.5]^2
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)

        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def soft_min(values, beta=50.0):
        """Soft minimum using log-sum-exp trick for smoothness."""
        v = np.asarray(values)
        if len(v) == 0:
            return 0.0
        m = np.min(v)
        # Soft min = -soft_max(-v)
        # soft_max(-v) = -log(sum(exp(-v_i / beta))) * beta ... 
        # Actually: soft_min(v) = -1/beta * log(sum(exp(-beta * v_i)))
        # To avoid overflow, use the shift:
        exp_vals = np.exp(-beta * (v - m))
        return m - np.log(np.sum(exp_vals)) / beta

    def smooth_negative_radius(centers_flat, beta=30.0):
        """Smooth surrogate: minimize -soft_min of all constraints."""
        centers = centers_flat.reshape(-1, 2)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # We want min pair dist / 2 to be large, so constraint is pair_dist/2 >= r
        # For soft-min, we consider all pairwise half-distances and boundary distances
        half_dists = dists / 2.0
        # Boundary distances
        dist_to_boundary = 0.5 - np.abs(centers)
        boundary_dists = dist_to_boundary.flatten()
        # Collect all constraint values (we want them all >= r, so r <= each)
        # soft_min of all these gives a smooth approximation of the min
        all_vals = np.concatenate([half_dists.flatten(), boundary_dists])
        sm = soft_min(all_vals, beta)
        return -sm

    def negative_radius(centers_flat):
        """Exact (nonsmooth) negative radius for final evaluation."""
        centers = centers_flat.reshape(-1, 2)
        return -compute_radius(centers)

    # --- Initialization strategies ---

    # Strategy 1: 7x7 grid (49 points) + 1 guest at center
    ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
    grid_7x7 = np.array([(x, y) for y in ticks7 for x in ticks7])
    guest_center = np.array([[0.0, 0.0]])
    init_7x7_guest = np.vstack([grid_7x7, guest_center])

    # Strategy 2: 5x10 grid
    ticks_x10 = (np.arange(10) + 0.5) / 10 - 0.5
    ticks_y5 = (np.arange(5) + 0.5) / 5 - 0.5
    grid_5x10 = np.array([(x, y) for y in ticks_y5 for x in ticks_x10])

    # Strategy 3: 7x7 grid + guest at various offsets
    np.random.seed(42)
    inits = [init_7x7_guest.copy(), grid_5x10.copy()]

    # Add several guest positions around center
    for dx in [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]:
        for dy in [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]:
            if dx == 0.0 and dy == 0.0:
                continue
            guest = np.array([[dx, dy]])
            inits.append(np.vstack([grid_7x7, guest]))

    # Also try shifting the 7x7 grid slightly to make room for guest
    for shift in [0.01, -0.01, 0.02, -0.02]:
        shifted = grid_7x7 + np.array([shift, 0.0])
        # Keep within bounds
        if np.max(np.abs(shifted)) < 0.5:
            guest = np.array([[0.0, 0.0]])
            inits.append(np.vstack([shifted, guest]))

    # Deduplicate initializations (approximate)
    unique_inits = [inits[0]]
    for init in inits[1:]:
        is_dup = False
        for u in unique_inits:
            if np.max(np.abs(init - u)) < 1e-8:
                is_dup = True
                break
        if not is_dup:
            unique_inits.append(init)
    inits = unique_inits

    best_r = -np.inf
    best_centers = None

    # Evaluate all initializations first
    for init in inits:
        r = compute_radius(init)
        if r > best_r:
            best_r = r
            best_centers = init.copy()

    # Optimize each initialization with smooth surrogate, then polish
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break

        x0 = init.flatten()

        # Phase 1: Smooth optimization with soft-min surrogate
        try:
            result = minimize(
                smooth_negative_radius,
                x0,
                method='Nelder-Mead',
                args=(20.0,),  # beta for soft-min
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            centers = result.x.reshape(-1, 2)
        except Exception:
            centers = init.copy()

        # Phase 2: Polish with exact (nonsmooth) objective
        if time.time() > deadline - 10:
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
            continue

        try:
            result2 = minimize(
                negative_radius,
                centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            centers2 = result2.x.reshape(-1, 2)
        except Exception:
            centers2 = centers

        r = compute_radius(centers2)
        if r > best_r:
            best_r = r
            best_centers = centers2.copy()

        # Early exit if we're very close to theoretical best
        if best_r > 0.07138:
            break

    # Final polish on the best result with tighter tolerance
    if best_centers is not None and time.time() < deadline - 5:
        try:
            result_final = minimize(
                negative_radius,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            centers_final = result_final.x.reshape(-1, 2)
            r_final = compute_radius(centers_final)
            if r_final > best_r:
                best_r = r_final
                best_centers = centers_final
        except Exception:
            pass

    # Safety: ensure all centers are finite and within bounds
    if best_centers is None:
        return grid_5x10

    # Clip to be strictly inside (tiny margin to avoid boundary issues)
    best_centers = np.clip(best_centers, -0.499999, 0.499999)

    return best_centers
# EVOLVE-BLOCK-END
