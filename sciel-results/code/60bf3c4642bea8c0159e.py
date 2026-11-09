# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    deadline = time.time() + 620  # 10 second safety margin

    def objective(x):
        """
        Objective function to minimize. We want to maximize d_min/d_max,
        which is equivalent to minimizing (d_max/d_min)^2 or similar.
        Let's minimize (d_max^2 / d_min^2) to avoid square roots where possible?
        Actually, R = (d_min/d_max)^2. We want to maximize R.
        So we minimize -log(R) = log(d_max^2) - log(d_min^2) = 2*log(d_max) - 2*log(d_min).
        Or simply minimize (d_max/d_min)^2.
        Using log-sum-exp style or just direct ratio.
        Direct ratio: d_max^2 / d_min^2.
        If d_min is 0, this is inf.
        """
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-10:
            return 1e10
        return (d_max / d_min) ** 2

    def objective_neg_log(x):
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-10:
            return 1e10
        return -2 * (np.log(d_min) - np.log(d_max))

    # Initialize best point and score
    best_pts = None
    best_score = -1

    def update_best(pts):
        nonlocal best_pts, best_score
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-10:
            score = 0.0
        else:
            score = (d_min / d_max) ** 2
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Strategy:
    # 1. Start with a regular polygon or grid and refine.
    # 2. Use Nelder-Mead or L-BFGS-B with multiple restarts.
    # 3. Use simulated annealing or differential evolution if time permits.

    # Initial guesses
    inits = []

    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly)

    # 2. Grid 4x4
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Center and scale
    grid -= grid.mean(axis=0)
    grid /= np.max(np.abs(grid))
    inits.append(grid)

    # 3. Random points on unit circle
    np.random.seed(1)
    for _ in range(5):
        r = np.random.rand(n)
        ang = np.random.rand(n) * 2 * np.pi
        pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        inits.append(pts)

    # 4. Hexagonal lattice subset
    # Generate a hex lattice and pick 16 points
    # This is tricky to do perfectly, so let's just use a dense grid and optimize.

    # Optimization loop
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break

        x0 = init.flatten()

        # Use Nelder-Mead
        # Maximize R = (d_min/d_max)^2
        # Minimize f(x) = (d_max/d_min)^2
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < 1e10:
                pts = res.x.reshape(n, 2)
                update_best(pts)
        except Exception:
            pass

        # Also try L-BFGS-B with gradient approximation? No gradient provided, use BFGS?
        # Nelder-Mead is robust for this non-smooth-ish function.
        # Let's also try a local search with small perturbations (Simulated Annealing style)
        # if time allows.

    # Refine the best point found so far with more iterations
    if best_pts is not None:
        if time.time() < deadline - 60:  # Leave 60s buffer
            x0 = best_pts.flatten()
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.success or res.fun < 1e10:
                    pts = res.x.reshape(n, 2)
                    update_best(pts)
            except Exception:
                pass

        # Further refinement with smaller steps
        if time.time() < deadline - 30:
            # Manual local search
            current_pts = best_pts.copy()
            current_score = best_score
            step_size = 0.01
            for _ in range(1000):
                if time.time() > deadline:
                    break
                # Try moving each point in random direction
                improved = False
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    idx = np.random.randint(n)
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    new_pts = current_pts.copy()
                    new_pts[idx] += step_size * direction

                    dists = pdist(new_pts)
                    d_min = np.min(dists)
                    d_max = np.max(dists)
                    if d_min < 1e-10:
                        score = 0.0
                    else:
                        score = (d_min / d_max) ** 2

                    if score > current_score:
                        current_pts = new_pts
                        current_score = score
                        improved = True

                if not improved:
                    step_size *= 0.9

            update_best(current_pts)

    if best_pts is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts


# EVOLVE-BLOCK-END
