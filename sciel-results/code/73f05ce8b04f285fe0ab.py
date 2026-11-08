"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
# Deadline guard: evaluator kills after 150s; keep safety margin
DEADLINE = 140.0  # seconds


def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside [0,1]^2 with disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -eps or x + r > 1 + eps or y - r < -eps or y + r > 1 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if d < ri + rj - eps:
                return False
    return True


def _score(circles):
    """Sum of radii if valid, else 0."""
    if _is_valid(circles):
        return float(np.sum(circles[:, 2]))
    return 0.0


def _soft_min_penalty(circles, beta=50.0):
    """
    Smooth surrogate for constraint violation.
    For each pair (i,j): penalty = soft_min(0, dist_ij - (ri + rj))
    For each circle i and each boundary: penalty = soft_min(0, margin)
    Returns a scalar penalty (>= 0), and we minimize objective + penalty.
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Pairwise circle-circle constraints: dist >= ri + rj
    # violation_ij = ri + rj - dist_ij  (positive means overlap)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            # violation = r[i] + r[j] - d
            # soft penalty: use log-sum-exp style or simple quadratic on positive part
            viol = r[i] + r[j] - d
            if viol > 0:
                # Smooth: use exp-based soft penalty
                penalty += np.exp(beta * viol) - 1.0
            # Also add a small smooth term even when not violating to guide gradient
            # Actually, let's use a different smooth approach:
            # penalty += max(0, viol)^2 is not smooth. Use:
            # soft: (1/beta) * log(1 + exp(beta * viol)) - (1/beta) * log(2)
            # But that's always positive. Let's just use the exp version above.
    
    # Boundary constraints: x_i - r_i >= 0, x_i + r_i <= 1, etc.
    for i in range(n):
        margins = [
            x[i] - r[i],        # left
            1.0 - (x[i] + r[i]),  # right
            y[i] - r[i],        # bottom
            1.0 - (y[i] + r[i])   # top
        ]
        for m in margins:
            viol = -m  # positive if violated
            if viol > 0:
                penalty += np.exp(beta * viol) - 1.0
    
    return penalty


def _objective_and_grad(params, beta=50.0):
    """
    Objective: -sum(r) + lambda * penalty
    params: array of shape (3N,) = [x0,y0,r0, x1,y1,r1, ...]
    Returns (f, grad)
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Objective: -sum(r)
    f = -np.sum(r)
    grad = np.zeros(3 * n)
    
    # Gradient of -sum(r) w.r.t. r_i is -1
    for i in range(n):
        grad[3 * i + 2] = -1.0
    
    # Add penalty gradient
    lam = 1.0  # penalty weight
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                d = 1e-12
            viol = r[i] + r[j] - d
            if viol > 0:
                # d/d(viol) of (exp(beta*viol) - 1) = beta * exp(beta*viol)
                coef = lam * beta * np.exp(beta * viol)
                # d(viol)/d(x_i) = -dx/d = -dx/sqrt(dx^2+dy^2) = -dx/d
                # d(viol)/d(x_j) = dx/d
                # d(viol)/d(r_i) = 1
                # d(viol)/d(r_j) = 1
                grad[3 * i + 0] += coef * (-dx / d)
                grad[3 * i + 1] += coef * (-dy / d)
                grad[3 * j + 0] += coef * (dx / d)
                grad[3 * j + 1] += coef * (dy / d)
                grad[3 * i + 2] += coef
                grad[3 * j + 2] += coef
            f += lam * (np.exp(beta * viol) - 1.0) if viol > 0 else 0.0
    
    # Boundaries
    for i in range(n):
        margins = [
            (x[i] - r[i], 0, -1),       # left: margin = x - r, d/dx=1, d/dr=-1
            (1.0 - x[i] - r[i], -1, -1), # right: margin = 1 - x - r, d/dx=-1, d/dr=-1
            (y[i] - r[i], 1, -1),       # bottom
            (1.0 - y[i] - r[i], -1, -1)  # top
        ]
        for m, dx_coeff, dr_coeff in margins:
            viol = -m
            if viol > 0:
                coef = lam * beta * np.exp(beta * viol)
                # d(viol)/d(x_i) = -dx_coeff (since viol = -margin, and margin has dx_coeff * x)
                # Actually margin = c * x + c' * r, so viol = -c*x - c'*r
                # d(viol)/d(x) = -c, d(viol)/d(r) = -c'
                if dx_coeff != 0:
                    if abs(dx_coeff) == 1:
                        # x or y component
                        if margins.index((m, dx_coeff, dr_coeff)) in [0, 1]:
                            # x component
                            grad[3 * i + 0] += coef * (-dx_coeff)
                        else:
                            # y component
                            grad[3 * i + 1] += coef * (-dx_coeff)
                grad[3 * i + 2] += coef * (-dr_coeff)
                f += lam * (np.exp(beta * viol) - 1.0)
    
    return f, grad


def _optimize_from_start(start_circles, beta=50.0, maxiter=2000, tol=1e-10):
    """Run L-BFGS-B optimization from a given starting configuration."""
    n = N
    params0 = start_circles.flatten()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    result = minimize(
        _objective_and_grad,
        params0,
        args=(beta,),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return result.x.reshape(n, 3)


def _grid_start(spacing=0.15):
    """Generate a grid of starting positions."""
    cells = []
    pos = 0.5 / spacing
    i = 0
    while len(cells) < N:
        j = 0
        while len(cells) < N:
            x = (i + 0.5) * spacing
            y = (j + 0.5) * spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                r = spacing * 0.45  # slightly smaller than half-spacing
                cells.append([x, y, r])
            j += 1
        i += 1
    return np.array(cells[:N])


def _random_start(n=N, seed=None):
    """Random starting configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    circles = np.zeros((n, 3))
    for i in range(n):
        r = rng.uniform(0.02, 0.12)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        circles[i] = [x, y, r]
    return circles


def construct():
    start_time = time.time()
    best_circles = None
    best_score = 0.0
    
    # Try multiple starting configurations
    starts = []
    
    # Grid start
    starts.append(_grid_start(spacing=0.15))
    starts.append(_grid_start(spacing=0.12))
    starts.append(_grid_start(spacing=0.10))
    
    # Random starts with different seeds
    for s in range(10):
        starts.append(_random_start(seed=s * 42 + 7))
    
    # Also try the original seed-like grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.49 / side) for i in range(side) for j in range(side)]
    starts.append(np.array(cells[:N]))
    
    for idx, start in enumerate(starts):
        if time.time() > DEADLINE - 20:
            break
        
        try:
            result = _optimize_from_start(start, beta=50.0, maxiter=1500, tol=1e-10)
            
            # Post-process: clip to valid range
            result[:, 2] = np.clip(result[:, 2], 0, 0.5)
            result[:, 0] = np.clip(result[:, 0], 0, 1)
            result[:, 1] = np.clip(result[:, 1], 0, 1)
            
            # Check validity and score
            if _is_valid(result):
                s = _score(result)
                if s > best_score:
                    best_score = s
                    best_circles = result.copy()
        except Exception:
            continue
    
    # If we have a valid result, try a polish pass with tighter parameters
    if best_circles is not None and time.time() < DEADLINE - 30:
        try:
            polished = _optimize_from_start(best_circles, beta=100.0, maxiter=3000, tol=1e-12)
            polished[:, 2] = np.clip(polished[:, 2], 0, 0.5)
            polished[:, 0] = np.clip(polished[:, 0], 0, 1)
            polished[:, 1] = np.clip(polished[:, 1], 0, 1)
            
            if _is_valid(polished):
                s = _score(polished)
                if s > best_score:
                    best_score = s
                    best_circles = polished.copy()
        except Exception:
            pass
    
    # Final fallback: return the seed grid if nothing better
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.49 / side) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
