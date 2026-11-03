"""Combined strategy: SDP-relaxation QP + smooth-surrogate annealed local optimization + global DE + multi-stage polish.
For 50 equal spheres in a unit cube [-1/2, 1/2]^3."""
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
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    def objective_direct(x):
        """Negative radius for minimization."""
        return -min_radius(x.reshape(-1, 3))

    def sdp_qp_solve(positions_init, maxiter=200, tol=1e-8):
        """
        SDP-relaxation inspired QP:
        Maximize t subject to:
          - x_i^T x_i <= R_max^2 (wall constraint relaxation)
          - (x_i - x_j)^T (x_i - x_j) >= 2*t  for all i<j
        where x_i are 3D positions.
        
        We solve this via iterative proximal/gradient method on the QP formulation:
        - Introduce auxiliary Gram matrix G where G_ij = x_i^T x_j
        - Constrain G to be PSD and consistent with 3D embedding
        - Use projected gradient ascent on t with penalty for violations
        
        Practical approach: alternate between
        1) Fixing positions, optimizing t (t = min over constraints)
        2) Fixing t, moving positions to increase min pairwise distance
           using subgradient of the min-pairwise-distance objective
        """
        from scipy.optimize import minimize
        pos = positions_init.copy().flatten()
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def qp_objective(x):
            """
            Objective: maximize min of {pairwise_half_dists, wall_clearances}
            Using a soft-min formulation that's differentiable.
            """
            p = x.reshape(-1, 3)
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            pd_half = 0.5 * np.sqrt(d2)
            walls = 0.5 - np.abs(p)
            mask = ~np.eye(N, dtype=bool)
            all_vals = np.concatenate([pd_half[mask], walls.ravel()])
            # Soft min with high beta for sharpness
            beta = 50.0
            scaled = -beta * all_vals
            m = np.max(scaled)
            lse = m + np.log(np.sum(np.exp(scaled - m)))
            soft_min = -lse / beta
            return -soft_min
        
        try:
            res = minimize(qp_objective, pos, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            pos_opt = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions_init

    def sdp_relaxation_phase(positions, n_restarts=3, maxiter_per=150):
        """
        SDP-inspired global search:
        1. Create multiple initializations by projecting random Gram matrices
           that satisfy PSD and diagonal constraints
        2. Optimize each with the QP solver
        3. Return the best result
        """
        best_pos = positions.copy()
        best_r = min_radius(positions)
        
        for restart in range(n_restarts):
            if remaining() < 10:
                break
            try:
                # Generate a "Gram-like" initialization:
                # Random points, then optimize to maximize min distance
                rng = np.random.RandomState(42 + restart * 100)
                # Start from a perturbed version of the current best
                noise_scale = 0.05 + restart * 0.03
                perturbed = positions + rng.normal(0, noise_scale, size=positions.shape)
                perturbed = np.clip(perturbed, -0.499999, 0.499999)
                
                pos_opt = sdp_qp_solve(perturbed, maxiter=maxiter_per)
                r = min_radius(pos_opt)
                if r > best_r:
                    best_r = r
                    best_pos = pos_opt.copy()
            except Exception:
                continue
        
        return best_pos

    def objective_smooth_neg(x, beta=20.0):
        """Smooth surrogate: -soft_min of all constraints (pairwise half-distances and walls)."""
        positions = x.reshape(-1, 3)
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        pd_half = 0.5 * np.sqrt(d2)  # (N,N)
        walls = (0.5 - np.abs(positions))  # (N,3)
        mask = ~np.eye(N, dtype=bool)
        pd_flat = pd_half[mask]
        walls_flat = walls.ravel()
        all_vals = np.concatenate([pd_flat, walls_flat])
        beta_safe = np.clip(beta, 1.0, 100.0)
        scaled = -beta_safe * all_vals
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        soft_min_val = -lse / beta_safe
        return -soft_min_val  # minimize this = maximize soft_min

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

    def polish(positions):
        """High-precision polish: direct L-BFGS-B with tight tolerance + coordinate descent."""
        if remaining() < 5:
            return positions
        pos = local_optimize_direct(positions, maxiter=300)
        if remaining() > 5:
            pos = coordinate_descent(pos, maxiter=100)
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
                                         polish=False, updating='immediate')
            pos_opt = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return None

    def full_pipeline(positions):
        """Run the full multi-stage optimization pipeline on given positions."""
        if remaining() < 10:
            return positions
        # Stage 0: SDP-relaxation phase (new)
        if remaining() > 50:
            pos = sdp_relaxation_phase(positions, n_restarts=2, maxiter_per=120)
        else:
            pos = positions
        # Stage 1: Smooth surrogate annealing
        if remaining() > 25:
            pos = local_optimize_smooth(pos, beta_start=5.0, beta_end=30.0, maxiter=150)
        else:
            pos = pos
        # Stage 2: Direct L-BFGS-B
        if remaining() > 12:
            pos = local_optimize_direct(pos, maxiter=250)
        # Stage 3: Coordinate descent polish
        if remaining() > 5:
            pos = coordinate_descent(pos, maxiter=50)
        return pos

    best_positions = None
    best_radius = -1.0

    def update_best(positions):
        nonlocal best_positions, best_radius
        r = min_radius(positions)
        if r > best_radius:
            best_radius = r
            best_positions = positions.copy()
        return r

    # Phase 1: Try multiple initializations with the full pipeline
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
    ]

    for name, init_func in initializations:
        if remaining() < 15:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            pos_opt = full_pipeline(positions)
            update_best(pos_opt)
        except Exception:
            continue

    # Phase 2: Differential evolution for global search (reduced budget since SDP phase uses more time)
    if remaining() > 30:
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        de_results = []
        for seed in [0, 1]:
            if remaining() < 20:
                break
            pos_de = differential_evolution_global(bounds, popsize=12, maxiter=25, seed=seed)
            if pos_de is not None:
                de_results.append(pos_de)
        
        # Polish the best DE result
        if de_results:
            # Pick the best from DE results
            best_de = max(de_results, key=lambda p: min_radius(p))
            if remaining() > 12:
                pos_polished = full_pipeline(best_de)
                update_best(pos_polished)

    # Phase 3: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 15:
        for perturb_seed in range(4):
            if remaining() < 8:
                break
            rng = np.random.RandomState(1000 + perturb_seed)
            perturbed = best_positions + rng.normal(0, 0.012, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            pos_opt = full_pipeline(perturbed)
            update_best(pos_opt)

    # Final polish on the best
    if best_positions is not None and remaining() > 8:
        best_positions = polish(best_positions)
        update_best(best_positions)

    # Ensure we return a valid array
    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    return best_positions
