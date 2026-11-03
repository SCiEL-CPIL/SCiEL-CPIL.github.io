"""Stronger bootstrap: differential-evolution global search + multi-stage local optimization
for 50 equal spheres in a unit cube. Model-and-optimize approach."""
import numpy as np
import time

N = 50
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def min_radius(positions):
        """Compute the largest common radius for positions of shape (N,3) in [-1/2,1/2]^3."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape != (N, 3):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        # Pairwise distances, vectorized
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        # Mask diagonal
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    def objective_direct(x):
        """Negative radius for minimization."""
        return -min_radius(x.reshape(-1, 3))

    def objective_smooth(x, beta=20.0):
        """Smooth surrogate: log-sum-exp based soft-min of the negative radii.
        Lower (more negative) is better. We minimize the soft-max of the constraints."""
        positions = x.reshape(-1, 3)
        # Pairwise
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        # Wall distances
        walls = (0.5 - np.abs(positions))  # shape (N,3)
        # We want to maximize r s.t. d_ij/2 >= r and wall_i >= r for all
        # Equiv: minimize t s.t. r >= d_ij/2, r >= wall_i -> maximize the soft-min
        # Use log-sum-exp to get a smooth estimate of the min (soft min).
        # soft_min(a) = -logsumexp(-beta * a) / beta
        # We want soft_min over {d_ij/2, walls}
        vals = []
        # Gather pairwise half-distances
        half_d2 = d2 / 4.0  # (d/2)^2 ; we'll sqrt after
        # Instead operate on the constraint gaps directly.
        # Let's just build a flat vector of all constraints: c = min_pair, min_wall... 
        # Simpler: compute soft-min of the set of all (d_ij/2) and (wall_ij).
        pd_half = 0.5 * np.sqrt(d2)  # (N,N)
        walls_mat = walls  # (N,3)
        # Flatten (exclude diagonal of pd_half)
        mask = ~np.eye(N, dtype=bool)
        pd_flat = pd_half[mask]
        walls_flat = walls.ravel()
        all_vals = np.concatenate([pd_flat, walls_flat])
        # soft min
        beta_safe = np.clip(beta, 1.0, 100.0)
        scaled = -beta_safe * all_vals
        # logsumexp numerically stable
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        soft_min_val = -lse / beta_safe
        return soft_min_val  # we will MINIMIZE the negative? No: maximize soft_min -> minimize -soft_min

    def objective_smooth_neg(x, beta=20.0):
        return -objective_smooth(x, beta)

    def make_initial_grid():
        """4x4x4 grid, take first N."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """BCC-inspired arrangement."""
        positions = []
        a = 0.5
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * (2 * a / 3)
                    y = -a + (iy + 0.5) * (2 * a / 3)
                    z = -a + (iz + 0.5) * (2 * a / 3)
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + (2 * a / 3) / 2, y + (2 * a / 3) / 2, z + (2 * a / 3) / 2])
        positions = np.array(positions[:N])
        return positions

    def make_random(seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    def local_optimize_direct(positions, maxiter=400):
        """L-BFGS-B on the direct (nonsmooth) objective."""
        from scipy.optimize import minimize
        if remaining() < 5:
            return positions
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res = minimize(objective_direct, positions.flatten(), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8})
            pos_opt = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def local_optimize_smooth(positions, beta_start=5.0, beta_end=30.0, maxiter=300):
        """L-BFGS-B on a smooth surrogate, annealing beta for a sharper objective."""
        from scipy.optimize import minimize
        if remaining() < 5:
            return positions
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        n_beta = 4
        betas = np.linspace(beta_start, beta_end, n_beta)
        pos = positions.flatten()
        for bi, beta in enumerate(betas):
            if remaining() < 3:
                break
            per_iter = max(20, int(maxiter / n_beta))
            def obj_b(x):
                return objective_smooth_neg(x, beta)
            try:
                res = minimize(obj_b, pos, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': per_iter, 'ftol': 1e-12})
                pos = res.x
            except Exception:
                pass
        pos_opt = np.clip(pos.reshape(-1, 3), -0.499999, 0.499999)
        return pos_opt

    def coordinate_descent(positions, maxiter=80):
        """Refine via coordinate descent on the direct objective."""
        if remaining() < 3:
            return positions
        pos = positions.copy()
        step = 0.008
        for it in range(maxiter):
            if remaining() < 2:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    old = pos[i, dim]
                    pos[i, dim] = min(0.499999, old + step)
                    r_pos = min_radius(pos)
                    pos[i, dim] = max(-0.499999, old - step)
                    r_neg = min_radius(pos)
                    pos[i, dim] = old
                    r_cur = min_radius(pos)
                    if r_pos > r_cur + 1e-13:
                        pos[i, dim] = min(0.499999, old + step)
                        improved = True
                    elif r_neg > r_cur + 1e-13:
                        pos[i, dim] = max(-0.499999, old - step)
                        improved = True
            if not improved:
                step *= 0.85
                if step < 1e-6:
                    break
        return pos

    def differential_evolution_global(bounds, popsize, maxiter, seed):
        """Simple differential evolution on the direct objective."""
        from scipy.optimize import differential_evolution
        if remaining() < 20:
            return None
        try:
            res = differential_evolution(objective_direct, bounds, method='de',
                                         popsize=popsize, maxiter=maxiter, seed=seed,
                                         tol=1e-10, mutation=(0.4, 1.0), recombination=0.7,
                                         polish=False, init='latinhypercube')
            return np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
        except Exception:
            return None

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = min_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    bounds = [(-0.499999, 0.499999)] * (N * 3)

    # Phase 1: Differential evolution global search (the main improvement)
    if remaining() > 60:
        de_pos = differential_evolution_global(bounds, popsize=15, maxiter=30, seed=1)
        if de_pos is not None:
            update_best(de_pos)

    # Phase 2: Multi-stage local optimization from multiple good starts
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]

    for name, init_func in initializations:
        if remaining() < 12:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)

            # Stage A: smooth surrogate (helps escape flat regions)
            if remaining() > 25:
                pos_opt = local_optimize_smooth(positions, beta_start=5.0, beta_end=25.0, maxiter=200)
            else:
                pos_opt = positions

            # Stage B: direct objective for sharpness
            if remaining() > 12:
                pos_opt = local_optimize_direct(pos_opt, maxiter=300)

            # Stage C: coordinate descent polish
            if remaining() > 6:
                pos_opt = coordinate_descent(pos_opt, maxiter=40)

            update_best(pos_opt)
        except Exception:
            continue

    # Phase 3: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 20:
        n_perturb = 6
        for k in range(n_perturb):
            if remaining() < 6:
                break
            rng = np.random.RandomState(1000 + k)
            noise_scale = 0.02 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            if remaining() > 18:
                pos_opt = local_optimize_smooth(perturbed, beta_start=8.0, beta_end=30.0, maxiter=150)
            else:
                pos_opt = perturbed

            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=250)

            if remaining() > 5:
                pos_opt = coordinate_descent(pos_opt, maxiter=30)

            update_best(pos_opt)

    # Phase 4: If time remains, a second DE round from a tighter box around best
    if best_positions is not None and remaining() > 40:
        # Build a tighter box around the current best
        tight_bounds = []
        for i in range(N * 3):
            center = best_positions.flatten()[i]
            lo = max(-0.499999, center - 0.10)
            hi = min(0.499999, center + 0.10)
            tight_bounds.append((lo, hi))
        de_pos2 = differential_evolution_global(tight_bounds, popsize=12, maxiter=25, seed=99)
        if de_pos2 is not None:
            # Local polish
            if remaining() > 15:
                de_pos2 = local_optimize_direct(de_pos2, maxiter=200)
            if remaining() > 8:
                de_pos2 = coordinate_descent(de_pos2, maxiter=30)
            update_best(de_pos2)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    # Final high-precision polish if time remains
    if remaining() > 10:
        try:
            polished = local_optimize_direct(best_positions, maxiter=400)
            polished = coordinate_descent(polished, maxiter=60)
            if min_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass

    return best_positions
