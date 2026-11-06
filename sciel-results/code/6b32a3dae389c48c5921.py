"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s limit

    # Helper: compute score (radius) for a configuration
    def compute_score(c):
        """Return the radius r for configuration c (N,2)."""
        # Boundary clearance
        abs_c = np.abs(c)
        min_boundary = 0.5 - np.max(abs_c, axis=1).min()
        
        # Pairwise distances (vectorized)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        r = min(min_pairwise / 2.0, min_boundary)
        return r

    # Objective: negative radius (to minimize)
    def objective(x):
        c = x.reshape(N, 2)
        # Boundary check
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e10
        
        abs_c = np.abs(c)
        min_boundary = 0.5 - np.max(abs_c, axis=1).min()
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-300)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        r = min(min_pairwise / 2.0, min_boundary)
        return -r

    # Gradient of -r w.r.t. centers
    def gradient(x):
        c = x.reshape(N, 2)
        grad = np.zeros_like(c)
        
        abs_c = np.abs(c)
        min_boundary = 0.5 - np.max(abs_c, axis=1).min()
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-300)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        r = min(min_pairwise / 2.0, min_boundary)
        
        # Determine which constraint is active
        if min_pairwise / 2.0 <= min_boundary:
            # Pairwise distance is the bottleneck
            # Find all pairs achieving min_pairwise
            idx = np.argmin(dists)
            i, j = np.unravel_index(idx, dists.shape)
            if i >= j:
                i, j = j, i
            d = c[j] - c[i]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                d = np.array([1.0, 0.0])
                dist = 1.0
            direction = d / dist
            # Gradient of -min_pairwise/2: push i and j apart
            grad[i] = -direction / 2.0
            grad[j] = direction / 2.0
        else:
            # Boundary is the bottleneck
            # Find which center(s) are at the boundary
            max_abs = np.max(abs_c, axis=1)
            min_idx = np.argmin(max_abs)
            # Find which coordinate is at max
            c_min = c[min_idx]
            abs_c_min = abs_c[min_idx]
            max_coord = np.argmax(abs_c_min)
            sign = np.sign(c_min[max_coord])
            if sign == 0:
                sign = 1.0
            # Gradient of -min_boundary: push center inward
            grad[min_idx, max_coord] = -sign
            # Also push other centers that are at the same boundary value
            for k in range(N):
                if k == min_idx:
                    continue
                if np.max(abs_c[k]) >= max_abs[min_idx] - 1e-12:
                    for coord in range(2):
                        if abs_c[k, coord] >= max_abs[k] - 1e-12:
                            s = np.sign(c[k, coord])
                            if s == 0:
                                s = 1.0
                            grad[k, coord] += -s * 0.1  # Small push
        
        return grad.flatten()

    # Objective with gradient for L-BFGS-B
    def objective_and_grad(x):
        return objective(x), gradient(x)

    # Generate diverse initial configurations
    def make_grid_init(perturb=0.0):
        m = int(np.ceil(np.sqrt(N)))  # 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N].copy()
        if perturb > 0:
            np.random.seed(42)
            c += np.random.normal(0, perturb, size=c.shape)
        c = np.clip(c, -0.499, 0.499)
        return c

    def make_hex_init():
        """Hexagonal packing-like initialization."""
        pts = []
        # Hexagonal arrangement: rows with alternating offsets
        row_height = 1.0 / (8 * np.sqrt(3) / 2)  # Approximate
        for row in range(8):
            y = -0.5 + (row + 0.5) * (1.0 / 8)
            offset = (row % 2) * (1.0 / 16)
            n_cols = 7 if row % 2 == 0 else 6
            for col in range(n_cols):
                x = -0.5 + (col + 0.5) * (1.0 / 7) + offset
                if len(pts) < N:
                    pts.append([x, y])
        # Fill remaining
        while len(pts) < N:
            pts.append(np.random.uniform(-0.45, 0.45, 2))
        c = np.array(pts[:N])
        c = np.clip(c, -0.499, 0.499)
        return c

    def make_random_init(seed=123):
        np.random.seed(seed)
        c = np.random.uniform(-0.45, 0.45, (N, 2))
        return c

    # Best tracking
    best_centers = make_grid_init(0.01)
    best_val = compute_score(best_centers)

    def try_optimize(c_init, maxiter=300, maxfun=50000):
        nonlocal best_centers, best_val
        try:
            result = minimize(
                objective_and_grad,
                c_init.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxfun, 'gtol': 1e-5},
                bounds=[(-0.5, 0.5)] * (2 * N)
            )
            c_curr = result.x.reshape(N, 2)
            c_curr = np.clip(c_curr, -0.5 + 1e-9, 0.5 - 1e-9)
            val = compute_score(c_curr)
            if val > best_val:
                best_val = val
                best_centers = c_curr.copy()
        except Exception:
            pass

    # Multiple restarts with different initializations
    initializations = [
        make_grid_init(0.01),
        make_grid_init(0.02),
        make_hex_init(),
        make_random_init(42),
        make_random_init(99),
    ]

    for i, c_init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        try_optimize(c_init, maxiter=200, maxfun=30000)

    # Second round: perturb best and re-optimize
    if time.time() < deadline - 15:
        c_perturb = best_centers.copy()
        np.random.seed(777)
        c_perturb += np.random.normal(0, 0.01, size=c_perturb.shape)
        c_perturb = np.clip(c_perturb, -0.499, 0.499)
        try_optimize(c_perturb, maxiter=300, maxfun=50000)

    # Third round: tighter optimization on best
    if time.time() < deadline - 10:
        try_optimize(best_centers, maxiter=500, maxfun=100000)

    # Final polish: pairwise repulsion for the closest pair
    if time.time() < deadline - 5:
        c = best_centers.copy()
        for _ in range(5000):
            if time.time() > deadline - 3:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_idx = np.argmin(dists)
            i, j = np.unravel_index(min_idx, dists.shape)
            if i >= j:
                i, j = j, i
            
            d = c[j] - c[i]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1.0, 0.0])
                dist = 1.0
            
            direction = d / dist
            step = 0.0005
            new_i = c[i] - direction * step
            new_j = c[j] + direction * step
            
            # Check boundaries
            if np.max(np.abs(new_i)) <= 0.5 and np.max(np.abs(new_j)) <= 0.5:
                old_r = compute_score(c)
                c[i] = new_i
                c[j] = new_j
                new_r = compute_score(c)
                if new_r > old_r + 1e-12:
                    pass  # Keep
                else:
                    c[i] = new_i + direction * step
                    c[j] = new_j - direction * step
                    break
        
        final_r = compute_score(c)
        if final_r > best_val:
            best_val = final_r
            best_centers = c.copy()

    # Ensure final result is valid
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_centers
# EVOLVE-BLOCK-END
