"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 120.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _soft_min(x, k=50):
    """Smooth approximation of min(x) using log-sum-exp."""
    # x is a 1D array of constraint values (should be >= 0 for validity)
    # We want to penalize negative values.
    # min(x) approx = -logsumexp(-x) / k
    # Actually, for constraints g(x) >= 0, we want to penalize max(0, -g(x)).
    # A smooth approximation of max(0, -g) is softplus(-g) = log(1 + exp(-g)) / k?
    # Let's use a standard soft penalty: sum of max(0, -g_i)^2.
    # But for L-BFGS-B, we need a smooth objective.
    # Let's use the "soft" constraint violation: sum(softplus(-g_i, k))
    # softplus(t) = log(1+exp(t)). For large negative t, ~0. For large positive t, ~t.
    # Here t = -g_i. If g_i > 0, t < 0, softplus ~ 0. If g_i < 0, t > 0, softplus ~ t = -g_i.
    # So sum(softplus(-g_i, k)) approximates sum(max(0, -g_i)).
    # We can scale by k to make it sharper.
    t = -x
    # Avoid overflow in exp
    t_max = np.max(t)
    if t_max > 50:
        t = t - t_max
    s = np.log(np.sum(np.exp(k * t))) / k
    return s

def _compute_violations(centers, radii, n):
    """Compute constraint violations (negative means violation)."""
    violations = []
    # Boundary
    for i in range(n):
        violations.append(centers[i, 0] - radii[i])
        violations.append(1.0 - centers[i, 0] - radii[i])
        violations.append(centers[i, 1] - radii[i])
        violations.append(1.0 - centers[i, 1] - radii[i])
    
    # Overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            violations.append(d - radii[i] - radii[j])
            
    # Positivity
    for i in range(n):
        violations.append(radii[i])
        
    return np.array(violations)

def _objective(params, n, penalty_weight):
    """Objective: -sum(radii) + penalty_weight * soft_constraint_violation."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Clip radii to non-negative for calculation stability
    radii_clip = np.maximum(radii, 0)
    
    obj = -np.sum(radii)
    
    if penalty_weight > 0:
        violations = _compute_violations(centers, radii_clip, n)
        # Soft penalty: sum of softplus(-violation, k)
        # Use k=20 for smoothness
        k = 20
        t = -violations
        t_max = np.max(t)
        if t_max > 50:
            t = t - t_max
        soft_viol = np.log(np.sum(np.exp(k * t))) / k
        obj += penalty_weight * soft_viol
        
    return obj

def _is_valid(centers, radii, tol=1e-6):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if radii[i] < -tol:
            return False
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_packing(centers, radii):
    """Fix small violations by shrinking radii and clipping positions."""
    centers = centers.copy()
    radii = radii.copy()
    n = len(radii)
    # Clip centers to valid range
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], 0.0, 1.0)
        centers[i, 1] = np.clip(centers[i, 1], 0.0, 1.0)
    # Ensure boundary validity
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        radii[i] = max(0.0, radii[i])
    # Iteratively fix overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                needed = radii[i] + radii[j]
                if d < needed:
                    # Shrink the smaller radius
                    if radii[i] < radii[j]:
                        radii[i] = max(0.0, (d - radii[j]) * 0.999)
                    else:
                        radii[j] = max(0.0, (d - radii[i]) * 0.999)
                    changed = True
        if not changed:
            break
    return centers, radii

def _random_initial(n_circles, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.15
    attempts = 0
    while len(centers) < n_circles and attempts < 20000:
        attempts += 1
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
    # Fill remaining randomly if needed
    while len(centers) < n_circles:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        centers.append((x, y))
    return np.array(centers)

def _hex_initial(n_circles, jitter=0.0):
    """Generate a hexagonal packing initialization."""
    rows = 5
    cols = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx + (0.5 * dx if r % 2 == 1 else 0)
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n_circles]
    return np.array(centers)

def _optimize_single_start(centers_init, radii_init, time_budget):
    """Run L-BFGS-B optimization from a single start."""
    n = len(radii_init)
    params = np.concatenate([centers_init.flatten(), radii_init])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
        
    # Grow and relax strategy
    # Start with high penalty, decrease it
    penalty_weights = [100.0, 50.0, 10.0, 1.0, 0.1, 0.01]
    
    current_params = params.copy()
    best_params = params.copy()
    best_val = np.sum(radii_init)
    
    for pw in penalty_weights:
        if _remaining() < 2.0:
            break
            
        res = minimize(
            _objective,
            current_params,
            args=(n, pw),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
        )
        
        current_params = res.x
        centers = current_params[:2*n].reshape(n, 2)
        radii = current_params[2*n:]
        
        # Check validity and update best
        if _is_valid(centers, radii, tol=1e-5):
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_params = current_params.copy()
                
    # Final polish with very low penalty and more iterations
    if _remaining() > 1.0:
        res = minimize(
            _objective,
            current_params,
            args=(n, 0.01),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )
        current_params = res.x
        centers = current_params[:2*n].reshape(n, 2)
        radii = current_params[2*n:]
        if _is_valid(centers, radii, tol=1e-5):
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_params = current_params.copy()
                
    centers_best = best_params[:2*n].reshape(n, 2)
    radii_best = best_params[2*n:]
    centers_best, radii_best = _fix_packing(centers_best, radii_best)
    
    if _is_valid(centers_best, radii_best, tol=1e-5):
        return centers_best, radii_best, np.sum(radii_best)
    else:
        return None, None, 0.0

def construct():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal
    for jitter in [0.0, 0.01, 0.02, 0.03]:
        c = _hex_initial(n, jitter=jitter)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.8 for i in range(n)])
        inits.append((c, r))
        
    # Random
    for seed in [42, 123, 7, 99, 555, 1234, 4321, 777]:
        c = _random_initial(n, seed=seed)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.7 for i in range(n)])
        inits.append((c, r))
        
    # Optimize from each start
    for idx, (c_init, r_init) in enumerate(inits):
        if _remaining() < 5.0:
            break
            
        time_budget = min(_remaining() - 2.0, 10.0)
        centers, radii, total = _optimize_single_start(c_init, r_init, time_budget)
        
        if centers is not None:
            if total > best_sum:
                best_sum = total
                best_centers = centers
                best_radii = radii
                
    # If no valid result found, fall back to simple grid
    if best_centers is None:
        side = int(np.ceil(np.sqrt(n)))
        dx = 1.0 / side
        dy = 1.0 / side
        centers = []
        radii = []
        for j in range(side):
            for i in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                r = 0.5 * dx - 1e-3
                centers.append((x, y))
                radii.append(r)
        best_centers = np.array(centers[:n])
        best_radii = np.array(radii[:n])
        best_sum = np.sum(best_radii)
        
    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
