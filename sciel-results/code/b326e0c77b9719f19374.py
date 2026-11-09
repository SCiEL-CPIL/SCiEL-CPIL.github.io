# EVOLVE-BLOCK-START
import numpy as np


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses a parametric ansatz P(x,y) ∝ exp(-α x² - β y² - γ xy) and optimizes
    (α, β, γ) via gradient ascent to maximize D/M where D = H_{-1} and M = max(H_0, H_1, H_2).

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    # P[y, x] so rows are y, columns are x
    X_grid = X  # shape (10, 10), X_grid[y, x] = x
    Y_grid = Y  # shape (10, 10), Y_grid[y, x] = y

    def compute_P(alpha, beta, gamma):
        """Compute normalized P given parameters."""
        logP = -alpha * X_grid**2 - beta * Y_grid**2 - gamma * X_grid * Y_grid
        # For numerical stability
        logP -= np.max(logP)
        P = np.exp(logP)
        s = np.sum(P)
        if s <= 0 or not np.isfinite(s):
            return None
        P /= s
        return P

    def entropy_of_projection(P, slope):
        """Compute Shannon entropy in bits of the projection x + slope*y."""
        vals = X_grid + slope * Y_grid
        # Pool by unique values
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        proj_probs = np.zeros(len(unique_vals))
        for i in range(len(unique_vals)):
            mask = (inverse == i)
            proj_probs[i] = np.sum(P[mask])
        # Ignore zero probabilities
        nonzero = proj_probs > 1e-15
        if not np.any(nonzero):
            return 0.0
        p = proj_probs[nonzero]
        p = p / np.sum(p)  # renormalize
        return -np.sum(p * np.log2(p))

    def objective(params):
        """Compute D/M ratio. Higher is better. Returns 0 if invalid."""
        alpha, beta, gamma = params
        P = compute_P(alpha, beta, gamma)
        if P is None:
            return 0.0
        D = entropy_of_projection(P, -1.0)
        H0 = entropy_of_projection(P, 0.0)
        H1 = entropy_of_projection(P, 1.0)
        H2 = entropy_of_projection(P, 2.0)
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 0.0
        return D / M

    # Comprehensive optimization: coarse grid + multi-start Nelder-Mead
    best_score = 0.0
    best_params = (0.0, 0.0, 0.0)
    best_P = None

    def nelder_mead(func, x0, bounds=None, max_iter=500, tol=1e-8):
        """Simple Nelder-Mead optimizer with optional bounds."""
        n = len(x0)
        # Initial simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            point = np.array(x0, dtype=float)
            step = 0.1 if bounds is None else (bounds[i][1] - bounds[i][0]) * 0.1
            point[i] += step
            simplex.append(point)
        
        # Evaluate initial simplex
        values = [func(p) for p in simplex]
        
        for iteration in range(max_iter):
            # Sort by value (ascending)
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            if abs(values[-1] - values[0]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < values[0]:
                # Expansion
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = func(simplex[i])
        
        # Return best point
        best_idx = np.argmin(values)
        return simplex[best_idx], values[best_idx]

    # Coarse grid search
    alpha_range = np.linspace(-3.0, 3.0, 25)
    beta_range = np.linspace(-3.0, 3.0, 25)
    gamma_range = np.linspace(-5.0, 5.0, 35)

    grid_best = []
    for a in alpha_range:
        for b in beta_range:
            for g in gamma_range:
                score = objective((a, b, g))
                if score > best_score:
                    best_score = score
                    best_params = (a, b, g)
                    best_P = compute_P(a, b, g)
                if score > 0.5:
                    grid_best.append((score, (a, b, g)))

    # Sort grid results and refine top candidates
    grid_best.sort(reverse=True)
    
    # Refine top 20 candidates with Nelder-Mead
    candidates = []
    if grid_best:
        candidates = [p for _, p in grid_best[:20]]
    else:
        candidates = [best_params]
    
    # Add some random restarts
    np.random.seed(42)
    for _ in range(10):
        a = np.random.uniform(-3, 3)
        b = np.random.uniform(-3, 3)
        g = np.random.uniform(-5, 5)
        candidates.append((a, b, g))
    
    # Refine each candidate
    bounds = [(-5, 5), (-5, 5), (-8, 8)]
    for init in candidates:
        try:
            result, val = nelder_mead(lambda p: objective(tuple(p)), init, bounds=bounds, max_iter=300)
            if val > best_score:
                best_score = val
                best_params = tuple(result)
                best_P = compute_P(*best_params)
        except Exception:
            pass

    # Final fine-tuning around the best solution
    if best_P is not None:
        for _ in range(3):
            result, val = nelder_mead(lambda p: objective(tuple(p)), best_params, bounds=bounds, max_iter=200, tol=1e-10)
            if val > best_score:
                best_score = val
                best_params = tuple(result)
                best_P = compute_P(*best_params)
            else:
                break

    # Final validation and normalization
    if best_P is None:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        # Clip and normalize
        best_P = np.clip(best_P, 0, None)
        s = np.sum(best_P)
        if s <= 1e-9:
            best_P = np.ones((n, n), dtype=np.float64) / (n * n)
        else:
            best_P /= s

    return best_P


# EVOLVE-BLOCK-END
