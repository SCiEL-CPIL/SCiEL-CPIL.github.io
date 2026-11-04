import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - elapsed()

    def is_valid(z, tol=1e-7):
        """Check if z (N,3) is a valid packing. Returns True if valid."""
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        if np.any(r <= 0):
            return False
        # Boundary check
        if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
            return False
        # Pairwise check
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(r_sum > d_ij + tol):
            return False
        return True

    def score(z):
        """Return sum of radii if valid, else -inf."""
        if is_valid(z):
            return np.sum(z[:, 2])
        return -np.inf

    def max_radius_for_position(i, x, y, r):
        """Max radius circle i can have at position (x,y) given other circles."""
        r_max = min(x, 1 - x, y, 1 - y)
        for j in range(N):
            if i == j:
                continue
            d = np.sqrt((x - x[j])**2 + (y - y[j])**2)
            r_max = min(r_max, d - r[j])
        return max(0.0, r_max)

    def maximize_radii(z):
        """Iteratively grow each circle's radius to its maximum feasible value."""
        z = z.copy()
        x, y, r = z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy()
        for _ in range(20):
            if elapsed() > deadline - 5:
                break
            changed = False
            for i in range(N):
                r_max = max_radius_for_position(i, x, y, r)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    changed = True
            if not changed:
                break
        z[:, 2] = r
        return z

    def local_search_positions(z, n_iters=200):
        """Local search: try small perturbations to positions, then maximize radii."""
        z = z.copy()
        x, y, r = z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy()
        best_r = np.sum(r)
        rng = np.random.RandomState(42)

        for it in range(n_iters):
            if elapsed() > deadline - 5:
                break
            # Pick a random circle to move
            i = rng.randint(N)
            # Try a small random perturbation
            dx = rng.uniform(-0.02, 0.02)
            dy = rng.uniform(-0.02, 0.02)
            nx = x[i] + dx
            ny = y[i] + dy

            # Check if new position keeps circle i within bounds with current radius
            if nx < r[i] or nx > 1 - r[i] or ny < r[i] or ny > 1 - r[i]:
                continue

            # Check no overlap with other circles at current radii
            ok = True
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((nx - x[j])**2 + (ny - y[j])**2)
                if d < r[i] + r[j] - 1e-10:
                    ok = False
                    break
            if not ok:
                continue

            # Try the move
            x[i], y[i] = nx, ny
            # Maximize radii from new positions
            new_r = r.copy()
            for k in range(N):
                new_r[k] = max_radius_for_position(k, x, y, new_r)
            new_score = np.sum(new_r)

            if new_score > best_r + 1e-10:
                best_r = new_score
                r = new_r
            else:
                # Revert
                x[i], y[i] = nx - dx, ny - dy

        z[:, 0], z[:, 1], z[:, 2] = x, y, r
        return z

    def soft_objective(z_flat):
        """Soft-constraint objective for L-BFGS-B."""
        z = z_flat.reshape(N, 3)
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        sum_r = np.sum(r)

        # Overlap penalty
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i, r_j = r[iu[0]], r[iu[1]]
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        pen_overlap = np.sum(overlaps**2)

        # Boundary penalty
        pen_b = (np.sum(np.maximum(0, r - x)**2) +
                 np.sum(np.maximum(0, r - (1 - x))**2) +
                 np.sum(np.maximum(0, r - y)**2) +
                 np.sum(np.maximum(0, r - (1 - y))**2))

        lam = 50.0
        return -sum_r + lam * (pen_overlap + pen_b)

    def soft_gradient(z_flat):
        z = z_flat.reshape(N, 3)
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        grad = np.zeros_like(z)
        lam = 50.0

        # -sum(r)
        grad[:, 2] -= 1.0

        # Overlap penalty gradient
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    continue
                ov = max(0.0, r[i] + r[j] - d)
                if ov > 0:
                    # d/dx_i of (r_i + r_j - d)^2 = -2*ov * dx/d
                    gx = -2 * lam * ov * dx / d
                    gy = -2 * lam * ov * dy / d
                    gr = -2 * lam * ov
                    grad[i, 0] += gx
                    grad[i, 1] += gy
                    grad[i, 2] += gr
                    grad[j, 0] -= gx
                    grad[j, 1] -= gy
                    grad[j, 2] += gr

        # Boundary penalty gradient
        for i in range(N):
            if r[i] > x[i]:
                d = r[i] - x[i]
                grad[i, 0] += -2 * lam * d
                grad[i, 2] += 2 * lam * d
            if r[i] > 1 - x[i]:
                d = r[i] - (1 - x[i])
                grad[i, 0] += 2 * lam * d
                grad[i, 2] += 2 * lam * d
            if r[i] > y[i]:
                d = r[i] - y[i]
                grad[i, 1] += -2 * lam * d
                grad[i, 2] += 2 * lam * d
            if r[i] > 1 - y[i]:
                d = r[i] - (1 - y[i])
                grad[i, 1] += 2 * lam * d
                grad[i, 2] += 2 * lam * d

        return grad.flatten()

    def hex_init(scale, ox, oy):
        """Generate a hexagonal lattice initialization."""
        pts = []
        row = 0
        while True:
            y = row * scale * np.sqrt(3) / 2 + oy
            if y > 1.0:
                break
            col = 0
            while True:
                x = col * scale + ox + (row % 2) * scale / 2
                if x > 1.0:
                    break
                pts.append((x, y))
                col += 1
            row += 1
        if len(pts) < N:
            return None
        np.random.shuffle(pts)
        sel = pts[:N]
        init = np.zeros((N, 3))
        for i, (px, py) in enumerate(sel):
            init[i, 0] = px
            init[i, 1] = py
            init[i, 2] = 0.05
        return init

    def random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        return np.column_stack((x, y, r))

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))
        init = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                init[count, 0] = (i + 0.5) / side
                init[count, 1] = (j + 0.5) / side
                init[count, 2] = 0.5 / side - 0.02
                count += 1
        return init

    def optimize_from_init(init, max_iter=500):
        """Run L-BFGS-B from an initialization, then polish."""
        if elapsed() > deadline - 10:
            return None

        z_flat = init.flatten()
        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.001, 0.5))

        try:
            result = minimize(
                soft_objective, z_flat,
                jac=soft_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-10}
            )
            z_opt = result.x.reshape(N, 3)
        except Exception:
            return None

        # Project to feasibility: clamp positions and reduce radii
        z_proj = z_opt.copy()
        x, y, r = z_proj[:, 0], z_proj[:, 1], z_proj[:, 2]
        # Clamp positions to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # Reduce radii to fit boundaries
        r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
        r = np.maximum(r, 0.001)

        # Iteratively reduce radii to eliminate overlaps
        for _ in range(30):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-10:
                        r[i] = max(0.001, r[i] * 0.5)
                        r[j] = max(0.001, r[j] * 0.5)
                        changed = True
                        continue
                    if r[i] + r[j] > d:
                        excess = r[i] + r[j] - d
                        # Reduce proportionally
                        total = r[i] + r[j]
                        if total > 0:
                            r[i] = max(0.001, r[i] - excess * r[i] / total)
                            r[j] = max(0.001, r[j] - excess * r[j] / total)
                        changed = True
            if not changed:
                break

        z_proj[:, 0], z_proj[:, 1], z_proj[:, 2] = x, y, r

        # Maximize radii
        z_grown = maximize_radii(z_proj)

        # Local search on positions
        if time_left() > 15:
            z_ls = local_search_positions(z_grown, n_iters=100)
            if score(z_ls) > score(z_grown):
                z_grown = z_ls

        return z_grown

    # Generate initializations
    inits = []

    # Grid init
    inits.append(grid_init())

    # Hex inits
    for scale in [0.12, 0.14, 0.16, 0.18, 0.20]:
        for ox in [0.02, 0.06, 0.10]:
            for oy in [0.02, 0.06]:
                h = hex_init(scale, ox, oy)
                if h is not None:
                    inits.append(h)

    # Random inits
    for s in range(15):
        inits.append(random_init(s))

    # Shuffle inits for diversity
    np.random.seed(0)
    np.random.shuffle(inits)

    best_score = -np.inf
    best_solution = grid_init()

    # Quick maximize radii on grid init as baseline
    best_solution = maximize_radii(best_solution)
    best_score = score(best_solution)

    # Process inits
    for idx, init in enumerate(inits):
        if elapsed() > deadline - 20:
            break

        # Adaptive max_iter: use less time for later inits
        remaining = deadline - elapsed() - 20
        if remaining < 5:
            break
        max_iter = int(min(500, max(100, remaining * 30)))

        z_opt = optimize_from_init(init, max_iter=max_iter)
        if z_opt is not None:
            s = score(z_opt)
            if s > best_score:
                best_score = s
                best_solution = z_opt.copy()

    # Final polish if time remains
    if time_left() > 10 and best_score > -np.inf:
        z_final = maximize_radii(best_solution)
        if score(z_final) > best_score:
            best_score = score(z_final)
            best_solution = z_final

        # One more local search pass
        if time_left() > 15:
            z_ls = local_search_positions(best_solution, n_iters=50)
            if score(z_ls) > best_score:
                best_score = score(z_ls)
                best_solution = z_ls

    # Ensure validity
    if not is_valid(best_solution):
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        best_solution = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    best_solution[count, 0] = (i + 0.5) / side
                    best_solution[count, 1] = (j + 0.5) / side
                    best_solution[count, 2] = 0.5 / side - 0.02
                    count += 1

    return best_solution
