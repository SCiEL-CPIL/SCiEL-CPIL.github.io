import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # explicit guard, safety margin before 150s kill

    # Precompute upper-triangle indices once (mask is invariant to point positions)
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def generate_hex_grid(spacing, rotation=0.0):
        """Generate hexagonal lattice points inside unit circle with optional rotation."""
        points = []
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 3
        max_i = int(2.0 / spacing) + 3
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if rotation != 0.0:
                    xr = x * np.cos(rotation) - y * np.sin(rotation)
                    yr = x * np.sin(rotation) + y * np.cos(rotation)
                else:
                    xr, yr = x, y
                if xr * xr + yr * yr <= 1.0:
                    points.append([xr, yr])
        return np.array(points)

    def generate_ring_init(n_rings=15):
        """Generate concentric ring initialization."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = ring * 0.065
            if r > 0.95:
                break
            n_points = int(2 * np.pi * r / 0.11)
            if n_points < 1:
                n_points = 1
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts)
        if len(pts) < N:
            extra = np.random.randn(N - len(pts), 2)
            extra *= 0.5
            pts = np.vstack([pts, extra])
        else:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        return pts

    def compute_score(pts):
        """Compute the packing score (minimum radius) in double precision."""
        diff = pts[tri_i, :] - pts[tri_j, :]
        d_flat = np.sqrt(np.sum(diff ** 2, axis=1))
        min_pair = np.min(d_flat)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = np.min(1.0 - radii)
        return min(min_pair / 2.0, min_bound)

    def soft_min_and_grad(x, beta):
        """Soft-min (log-sum-exp) and its gradient."""
        x_max = np.max(x)
        exp_neg = np.exp(-beta * (x - x_max))
        sum_exp = np.sum(exp_neg)
        s_val = -1.0 / beta * (np.log(sum_exp) - beta * x_max)
        grad = exp_neg / sum_exp
        return s_val, grad

    def objective_and_grad(flat_pts, beta):
        """Objective (negative soft-min radius) and gradient. Only upper triangle."""
        pts = flat_pts.reshape(N, 2)

        diff = pts[tri_i, :] - pts[tri_j, :]
        d_flat = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-12)

        radii = np.hypot(pts[:, 0], pts[:, 1])
        d_bound = 1.0 - radii

        eff_rad = np.concatenate([d_flat / 2.0, d_bound])

        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        obj = -s_val

        grad_pair = grad_eff[:n_pairs]
        grad_bound = grad_eff[n_pairs:]

        # Gradient of d_ij wrt p_i and p_j
        safe_d = np.where(d_flat < 1e-9, 1e-9, d_flat)
        unit = diff / safe_d[:, np.newaxis]

        # Force on i from pair (i,j): grad_pair * (p_i - p_j)/d ; on j: -that
        F = np.zeros((N, 2))
        # contribution to i
        np.add.at(F, tri_i, grad_pair[:, np.newaxis] * unit)
        # contribution to j (negative)
        np.add.at(F, tri_j, -grad_pair[:, np.newaxis] * unit)

        # Boundary: d_bound_i = 1 - r_i, grad wrt p_i = -p_i / r_i
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        F_bound = -grad_bound[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        F += F_bound

        return obj, F.flatten()

    # ---- 1. Initialization: best of hex lattices (rotated) and ring ----
    best_init = None
    best_init_score = -np.inf

    spacings = [0.10, 0.104, 0.105, 0.106, 0.107, 0.108, 0.11, 0.112, 0.115]
    rotations = [0.0, np.pi / 12, np.pi / 8, np.pi / 6, np.pi / 5, np.pi / 4]

    for sp in spacings:
        for rot in rotations:
            pts = generate_hex_grid(sp, rot)
            if len(pts) < N:
                continue
            dists_to_center = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            idx = np.argsort(dists_to_center)[:N]
            init_pts = pts[idx].copy()
            score = compute_score(init_pts)
            if score > best_init_score:
                best_init_score = score
                best_init = init_pts

    ring_pts = generate_ring_init(15)
    ring_score = compute_score(ring_pts)
    if ring_score > best_init_score:
        best_init_score = ring_score
        best_init = ring_pts

    if best_init is None:
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9
        best_init_score = compute_score(best_init)

    current_points = best_init.copy()
    current_best_score = best_init_score
    current_best_points = best_init.copy()

    def project_inside(pts):
        radii = np.hypot(pts[:, 0], pts[:, 1])
        out = radii > 1.0
        if np.any(out):
            pts[out] *= (0.9999 / radii[out])[:, np.newaxis]
        return pts

    # ---- 2. Annealed gradient optimization with jittered multi-start ----
    max_time = deadline - start_time - 8.0
    beta_start = 30.0
    beta_end = 50000.0
    num_rounds = 10
    time_per_round = max_time / num_rounds
    rng = np.random.default_rng(12345)

    for round_idx in range(num_rounds):
        if time.time() > deadline - 6.0:
            break
        t = round_idx / max(1, num_rounds - 1)
        beta = beta_start * (beta_end / beta_start) ** t

        flat_pts = current_points.flatten()
        round_deadline = time.time() + time_per_round * 0.85

        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 800,
                    'maxfun': 1500,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            if result.fun < 0 or result.success:
                new_pts = result.x.reshape(N, 2)
                new_pts = project_inside(new_pts)
                new_score = compute_score(new_pts)
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
                    current_points = new_pts.copy()
        except Exception:
            pass

        # Jittered multi-start: escape local minima before the next (higher-beta) round
        if time.time() < deadline - 8.0 and round_idx < num_rounds - 1:
            jitter_scale = 0.004 * (1.0 - t) + 0.0005
            jittered = current_points + rng.normal(0.0, jitter_scale, size=(N, 2))
            jittered = project_inside(jittered)
            j_score = compute_score(jittered)
            # Only keep jitter if it doesn't hurt; otherwise keep the clean point
            # (the next round will re-optimize from whichever is better)
            if j_score >= current_best_score - 1e-6:
                current_points = jittered.copy()
            else:
                current_points = current_best_points.copy()

    # ---- 3. Final high-precision gradient polish ----
    if time.time() < deadline - 5.0:
        beta_final = 100000.0
        flat_pts = current_best_points.flatten()
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta_final,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            if result.fun < 0 or result.success:
                new_pts = result.x.reshape(N, 2)
                new_pts = project_inside(new_pts)
                new_score = compute_score(new_pts)
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
        except Exception:
            pass

    # ---- 4. Target-radius force-directed polish (from inspiration) ----
    # Drives r above the soft-min plateau by fixing constraint violations.
    if time.time() < deadline - 4.0:
        pts = current_best_points.copy()
        r = current_best_score
        step_size = 0.0008
        decay = 0.9995
        for it in range(3000):
            if time.time() > deadline - 4.0:
                break
            diff = pts[tri_i, :] - pts[tri_j, :]
            d_flat = np.sqrt(np.sum(diff ** 2, axis=1))
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_pair = np.min(d_flat)
            min_bound = np.min(1.0 - radii)
            r = min(min_pair / 2.0, min_bound)

            target_r = r * 1.0005 + 0.00005
            forces = np.zeros((N, 2))

            # Pairwise repulsion for pairs closer than 2*target_r
            overlap = np.where(d_flat < 2.0 * target_r, 2.0 * target_r - d_flat, 0.0)
            safe_d = np.where(d_flat < 1e-9, 1e-9, d_flat)
            unit = diff / safe_d[:, np.newaxis]
            contrib = overlap[:, np.newaxis] * unit
            np.add.at(forces, tri_i, contrib)
            np.add.at(forces, tri_j, -contrib)

            # Boundary pull
            b_overlap = np.where(radii > 1.0 - target_r, radii - (1.0 - target_r), 0.0)
            safe_radii = np.where(radii < 1e-9, 1e-9, radii)
            forces += -pts / safe_radii[:, np.newaxis] * b_overlap[:, np.newaxis]

            fm = np.linalg.norm(forces, axis=1)
            max_fm = np.max(fm)
            if max_fm > 0:
                pts += step_size * forces / max_fm
            pts = project_inside(pts)
            step_size *= decay

            if it % 100 == 0:
                pts += rng.normal(0.0, 0.0001, size=(N, 2))
                pts = project_inside(pts)

        new_score = compute_score(pts)
        if new_score > current_best_score:
            current_best_score = new_score
            current_best_points = pts.copy()

    # ---- 5. Exact active-constraint local solve (final high-precision pass) ----
    # Only move points involved in the active (minimum) constraints, tiny steps.
    if time.time() < deadline - 2.0:
        pts = current_best_points.copy()
        r = current_best_score
        for it in range(4000):
            if time.time() > deadline - 2.0:
                break
            diff = pts[tri_i, :] - pts[tri_j, :]
            d_flat = np.sqrt(np.sum(diff ** 2, axis=1))
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_pair = np.min(d_flat)
            min_bound = np.min(1.0 - radii)
            r = min(min_pair / 2.0, min_bound)
            if r <= current_best_score:
                r = current_best_score

            tol = 1e-7
            improved = False
            # Active pair constraints: distance below 2*r + tol
            active_pairs = d_flat < 2.0 * r + tol
            if np.any(active_pairs):
                for k in np.where(active_pairs)[0]:
                    i, j = tri_i[k], tri_j[k]
                    d = d_flat[k]
                    if d < 1e-10:
                        pts[i] += 0.0005 * rng.normal(size=2)
                        improved = True
                        continue
                    direction = (pts[i] - pts[j]) / d
                    push = (2.0 * r - d) / 2.0
                    if push > 0:
                        pts[i] += push * direction
                        pts[j] -= push * direction
                        improved = True
            # Active boundary constraints
            active_bound = radii > 1.0 - r - tol
            if np.any(active_bound):
                for i in np.where(active_bound)[0]:
                    rad = radii[i]
                    if rad < 1e-10:
                        pts[i] += 0.0005 * rng.normal(size=2)
                        improved = True
                        continue
                    pts[i] *= (1.0 - r) / rad
                    improved = True
            pts = project_inside(pts)
            if not improved:
                break

        new_score = compute_score(pts)
        if new_score > current_best_score:
            current_best_score = new_score
            current_best_points = pts.copy()

    # Final safety: ensure all points strictly inside and finite
    radii = np.hypot(current_best_points[:, 0], current_best_points[:, 1])
    out = radii > (1.0 - 1e-9)
    if np.any(out):
        current_best_points[out] *= ((1.0 - 1e-9) / radii[out])[:, np.newaxis]
    current_best_points = np.nan_to_num(current_best_points, nan=0.0, posinf=0.0, neginf=0.0)

    return current_best_points
