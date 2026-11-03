"""Sphere packing: 50 equal spheres in unit cube [-0.5, 0.5]^3.
Strategy: diverse initializations + efficient L-BFGS-B local optimization
with a smooth soft-min objective, then direct refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill


def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    def remaining():
        return DEADLINE - elapsed()

    def compute_radius(pos):
        """Largest common radius for positions (N,3)."""
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape != (N, 3) or not np.all(np.isfinite(pos)):
            return 0.0
        if np.any(np.abs(pos) > 0.5 + 1e-9):
            return 0.0
        # Pairwise half-distances
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) * 0.5
        # Wall distances
        walls = 0.5 - np.abs(pos)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    def soft_min_radius(pos, beta):
        """Smooth surrogate: soft-min of all constraints. Higher is better."""
        pos = pos.reshape(-1, 3)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        half_d = 0.5 * np.sqrt(d2)
        walls = 0.5 - np.abs(pos)  # (N, 3)
        # Collect all constraint values
        mask = ~np.eye(N, dtype=bool)
        all_vals = np.concatenate([half_d[mask], walls.ravel()])
        # Soft min via log-sum-exp
        scaled = -beta * all_vals
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        return -lse / beta

    def objective_smooth(x, beta):
        """Minimize negative soft-min radius."""
        return -soft_min_radius(x, beta)

    def objective_direct(x):
        """Minimize negative exact radius."""
        return -compute_radius(x.reshape(-1, 3))

    def local_optimize(pos, beta_schedule=None, maxiter_total=600):
        """Multi-stage L-BFGS-B: smooth (annealed beta) then direct."""
        if remaining() < 5:
            return pos
        bounds = [(-0.4999999, 0.4999999)] * (N * 3)
        x = pos.flatten().copy()

        if beta_schedule is not None:
            n_stages = len(beta_schedule)
            iters_per = max(30, maxiter_total // n_stages)
            for beta in beta_schedule:
                if remaining() < 3:
                    break
                try:
                    res = minimize(objective_smooth, x, args=(beta,),
                                   method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': iters_per, 'ftol': 1e-12, 'gtol': 1e-6})
                    x = res.x
                except Exception:
                    pass

        # Final direct refinement
        if remaining() > 5:
            try:
                res = minimize(objective_direct, x, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
                x = res.x
            except Exception:
                pass

        return np.clip(x.reshape(-1, 3), -0.4999999, 0.4999999)

    def coordinate_descent(pos, maxiter=50):
        """Fine coordinate descent polish."""
        if remaining() < 3:
            return pos
        pos = pos.copy()
        step = 0.005
        for it in range(maxiter):
            if remaining() < 2:
                break
            improved = False
            r_cur = compute_radius(pos)
            for i in range(N):
                for d in range(3):
                    old = pos[i, d]
                    pos[i, d] = min(0.4999999, old + step)
                    r_p = compute_radius(pos)
                    pos[i, d] = max(-0.4999999, old - step)
                    r_m = compute_radius(pos)
                    pos[i, d] = old
                    if r_p > r_cur + 1e-14:
                        pos[i, d] = min(0.4999999, old + step)
                        r_cur = r_p
                        improved = True
                    elif r_m > r_cur + 1e-14:
                        pos[i, d] = max(-0.4999999, old - step)
                        r_cur = r_m
                        improved = True
            if not improved:
                step *= 0.8
                if step < 1e-7:
                    break
        return pos

    def init_grid():
        """4x4x4 grid, take 50 points."""
        m = 4
        t = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y, z) for z in t for y in t for x in t])
        return pts[:N].copy()

    def init_bcc():
        """BCC-inspired: 3x3x3 corners + centers, take 50."""
        a = 0.5
        pts = []
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * (2 * a / 3)
                    y = -a + (iy + 0.5) * (2 * a / 3)
                    z = -a + (iz + 0.5) * (2 * a / 3)
                    pts.append([x, y, z])
                    if len(pts) < N:
                        pts.append([x + (2*a/3)/2, y + (2*a/3)/2, z + (2*a/3)/2])
        return np.array(pts[:N])

    def init_random_repulsive(seed):
        """Random start with pairwise repulsion to spread points."""
        rng = np.random.RandomState(seed)
        pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        # Simple repulsion iterations
        for _ in range(30):
            for i in range(N):
                # Push away from nearest neighbors
                diff = pos[i] - pos
                d2 = np.einsum('ij,ij->i', diff, diff)
                d2[i] = np.inf
                min_j = np.argmin(d2)
                min_d = np.sqrt(d2[min_j])
                if min_d < 0.15:
                    push = diff[min_j] / max(min_d, 1e-8) * 0.01
                    pos[i] += push
            # Also push from walls
            for i in range(3):
                pos[:, i] = np.clip(pos[:, i], -0.48, 0.48)
        return np.clip(pos, -0.4999999, 0.4999999)

    def init_fcc_like():
        """FCC-inspired layered arrangement."""
        pts = []
        layers = 5
        a = 0.5
        for lz in range(layers):
            z = -a + (lz + 0.5) * (2 * a / layers)
            per_layer = N // layers + (1 if lz < N % layers else 0)
            for k in range(per_layer):
                angle = 2 * np.pi * k / per_layer
                r = 0.3 * np.sqrt(k / max(per_layer - 1, 1))
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                pts.append([x, y, z])
        pts = np.array(pts[:N])
        return np.clip(pts, -0.4999999, 0.4999999)

    # Run optimization from multiple diverse starts
    initializations = [
        ("grid", init_grid()),
        ("bcc", init_bcc()),
        ("fcc", init_fcc_like()),
        ("rep0", init_random_repulsive(0)),
        ("rep1", init_random_repulsive(42)),
        ("rep2", init_random_repulsive(123)),
        ("rep3", init_random_repulsive(777)),
        ("rep4", init_random_repulsive(2024)),
        ("rep5", init_random_repulsive(999)),
    ]

    best_pos = None
    best_r = -1.0

    def update_best(pos):
        nonlocal best_pos, best_r
        r = compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        return r

    # Optimize each initialization
    for name, pos0 in initializations:
        if remaining() < 15:
            break
        try:
            # Use annealed beta schedule for smooth optimization
            betas = [3.0, 8.0, 15.0, 25.0]
            pos_opt = local_optimize(pos0, beta_schedule=betas, maxiter_total=500)
            # Coordinate descent polish
            if remaining() > 8:
                pos_opt = coordinate_descent(pos_opt, maxiter=40)
            update_best(pos_opt)
        except Exception:
            continue

    # Perturbation refinement of best
    if best_pos is not None:
        for k in range(8):
            if remaining() < 12:
                break
            rng = np.random.RandomState(5000 + k)
            scale = 0.015 * (1.0 - k / 9.0)
            pert = best_pos + rng.normal(0, scale, size=best_pos.shape)
            pert = np.clip(pert, -0.4999999, 0.4999999)
            try:
                betas = [8.0, 15.0, 25.0]
                pos_opt = local_optimize(pert, beta_schedule=betas, maxiter_total=400)
                if remaining() > 6:
                    pos_opt = coordinate_descent(pos_opt, maxiter=30)
                update_best(pos_opt)
            except Exception:
                continue

    # Final high-precision polish
    if best_pos is not None and remaining() > 10:
        try:
            polished = local_optimize(best_pos, beta_schedule=[20.0, 30.0], maxiter_total=300)
            polished = coordinate_descent(polished, maxiter=60)
            if compute_radius(polished) > best_r:
                best_pos = polished
        except Exception:
            pass

    # Ensure validity
    if best_pos is None:
        best_pos = init_grid()
    best_pos = np.clip(best_pos, -0.4999999, 0.4999999)

    return best_pos
