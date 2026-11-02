"""Circle packing in unit square – CMA-ES based global optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import cma

def construct():
    t0 = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # State vector: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    dim = 3 * N

    def objective(x):
        """Penalty-based objective: minimize (overlap + boundary) - sum_radii."""
        # x is a flat array of length 3*N
        # Reshape
        pts = x.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        rs = pts[:, 2]

        # Boundary penalty: circles must be inside [0,1]^2
        # For each circle: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Violation = max(0, r - x) + max(0, r - (1-x)) + max(0, r - y) + max(0, r - (1-y))
        bnd = (np.maximum(0, rs - xs) + np.maximum(0, rs - (1 - xs)) +
               np.maximum(0, rs - ys) + np.maximum(0, rs - (1 - ys)))
        bnd_penalty = np.sum(bnd)

        # Overlap penalty: for each pair i<j, if dist < ri+rj, penalty = (ri+rj - dist)^2
        # Use pairwise distances
        # To avoid O(N^2) in Python, use vectorized approach
        # For 26 circles, 325 pairs – manageable
        total_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    total_overlap += (min_dist - dist) ** 2

        overlap_penalty = total_overlap

        # Sum of radii (to maximize)
        sum_radii = np.sum(rs)

        # Objective: minimize penalty - sum_radii (so lower is better)
        # Use a large penalty weight to ensure feasibility
        penalty_weight = 100.0
        return penalty_weight * (bnd_penalty + overlap_penalty) - sum_radii

    def decode_and_validate(x):
        """Convert flat vector to (x, y, r) array and ensure validity."""
        pts = x.reshape(N, 3)
        xs = pts[:, 0].copy()
        ys = pts[:, 1].copy()
        rs = pts[:, 2].copy()

        # Clamp radii to positive
        rs = np.maximum(rs, 0.001)

        # Clamp centers to keep circles inside [0,1]^2
        # x must be in [r, 1-r], y must be in [r, 1-r]
        xs = np.clip(xs, rs, 1 - rs)
        ys = np.clip(ys, rs, 1 - rs)

        # Ensure radii don't exceed 0.5 (can't fit in unit square otherwise)
        rs = np.minimum(rs, 0.5)

        return np.column_stack([xs, ys, rs])

    # Initialize with a reasonable starting configuration
    # Use a hexagonal-like arrangement
    def initial_config():
        """Generate a diverse initial configuration."""
        # Start with grid and add some randomness
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                r = 0.5 / side - 0.01
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                pts.append([x, y, r])
        
        # Pad if needed
        while len(pts) < N:
            pts.append([0.5, 0.5, 0.01])
        
        return np.array(pts[:N]).flatten()

    best_x = None
    best_obj = np.inf

    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if restart == 0:
            x0 = initial_config()
        elif restart == 1:
            # Corner-weighted
            pts = []
            corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
            for cx, cy in corners:
                pts.append([cx, cy, 0.15])
            for i in range(N - 4):
                x = np.random.uniform(0.2, 0.8)
                y = np.random.uniform(0.2, 0.8)
                r = np.random.uniform(0.05, 0.12)
                pts.append([x, y, r])
            x0 = np.array(pts[:N]).flatten()
        elif restart == 2:
            # Hexagonal pattern
            pts = []
            r_base = 0.08
            dx = 2 * r_base
            dy = dx * np.sqrt(3) / 2
            for row in range(4):
                for col in range(4):
                    if len(pts) >= N:
                        break
                    x = 0.1 + col * dx + (row % 2) * dx / 2
                    y = 0.1 + row * dy
                    r = r_base
                    if x + r > 1:
                        x = 1 - r
                    if y + r > 1:
                        y = 1 - r
                    pts.append([x, y, r])
            while len(pts) < N:
                pts.append([0.5, 0.5, 0.05])
            x0 = np.array(pts[:N]).flatten()
        else:
            # Random
            x0 = np.random.uniform(0.1, 0.9, dim)
            # Set radii to small values
            for k in range(N):
                x0[3 * k + 2] = np.random.uniform(0.03, 0.1)
            # Clamp
            for k in range(N):
                x0[3 * k] = np.clip(x0[3 * k], x0[3 * k + 2], 1 - x0[3 * k + 2])
                x0[3 * k + 1] = np.clip(x0[3 * k + 1], x0[3 * k + 2], 1 - x0[3 * k + 2])

        # CMA-ES settings
        sigma0 = 0.1
        opts = {
            'maxfevals': 2000,
            'seed': 42 + restart,
            'verbose': -9,
            'verb_log': 0,
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            if time.time() > deadline:
                break
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)

            # Track best
            best_in_es = es.result.xbest
            obj_best = es.result.fbest
            if obj_best < best_obj:
                best_obj = obj_best
                best_x = best_in_es.copy()

        # After CMA-ES, do local refinement with Nelder-Mead on the best
        if time.time() < deadline - 5:
            from scipy.optimize import minimize
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()

    if best_x is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final validation and cleanup
    result = decode_and_validate(best_x)
    return result
# EVOLVE-BLOCK-END
