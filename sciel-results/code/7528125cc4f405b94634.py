"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 110.0  # safety margin before 120s limit


def _hilbert_d2xy(d, n_bits=12):
    """Convert 1D Hilbert curve coordinate (in [0,1]) to 2D point in [0,1]^2."""
    # Use a simple space-filling curve approximation via interleaving bits
    # For better locality, use a recursive Hilbert-like mapping
    d = np.clip(d, 0, 1)
    # Map to integer grid
    size = 1 << n_bits
    idx = int(d * (size - 1))
    
    # Simple Hilbert curve d->xy conversion
    x = 0
    y = 0
    s = size // 2
    while s > 0:
        rx = 1 & (idx >> 1)
        ry = 1 & (idx ^ rx)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        idx >>= 2
        s //= 2
    
    return np.array([x / (size - 1), y / (size - 1)])


def _hilbert_curve_positions(n, n_bits=12):
    """Get n evenly-spaced points along the Hilbert curve."""
    # Use midpoints of segments for better coverage
    ts = (np.arange(n) + 0.5) / n
    positions = np.array([_hilbert_d2xy(t, n_bits) for t in ts])
    return positions


def _objective(params, n, boundary_penalty=1000.0, overlap_penalty=10000.0):
    """Objective: minimize negative sum of radii + penalties for violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Boundary constraints: circles must be inside [0,1]^2
    # r[i] <= x[i], r[i] <= 1-x[i], r[i] <= y[i], r[i] <= 1-y[i]
    viol_x = np.maximum(0, r - np.minimum(x, 1 - x))
    viol_y = np.maximum(0, r - np.minimum(y, 1 - y))
    boundary_viol = np.sum(viol_x**2) + np.sum(viol_y**2)
    
    # Overlap constraints: dist(i,j) >= r[i] + r[j]
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)**2
    
    total_penalty = boundary_penalty * boundary_viol + overlap_penalty * overlap_viol
    return -np.sum(r) + total_penalty


def _valid(arr):
    """Check if arrangement is valid: all circles inside unit square, no overlaps."""
    n = arr.shape[0]
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Boundary check
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _score(arr):
    """Score: sum of radii if valid, else -inf."""
    if not _valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])


def construct():
    start_time = time.time()
    n = N
    
    # Initial positions from Hilbert curve
    init_positions = _hilbert_curve_positions(n, n_bits=12)
    
    # Initial radii: small
    init_r = np.full(n, 0.02)
    
    x0 = np.concatenate([init_positions[:, 0], init_positions[:, 1], init_r])
    
    best_params = x0.copy()
    best_score = -np.inf
    
    # Try multiple random restarts and local optimizations
    n_restarts = 30
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Perturb initial conditions
        if restart == 0:
            params0 = x0.copy()
        else:
            # Random perturbation of Hilbert positions
            pos = init_positions.copy()
            noise_scale = 0.05 * (restart + 1) / n_restarts
            pos += rng.randn(n, 2) * noise_scale
            pos = np.clip(pos, 0.05, 0.95)
            r0 = np.full(n, 0.015 + 0.02 * rng.random())
            params0 = np.concatenate([pos[:, 0], pos[:, 1], r0])
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(
                _objective,
                params0,
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Extract and validate
            x_opt = res.x[:n]
            y_opt = res.x[n:2*n]
            r_opt = np.maximum(0, res.x[2*n:])  # Ensure non-negative radii
            
            # Clip to boundary
            r_opt = np.minimum(r_opt, np.minimum(x_opt, 1 - x_opt))
            r_opt = np.minimum(r_opt, np.minimum(y_opt, 1 - y_opt))
            
            arr = np.column_stack([x_opt, y_opt, r_opt])
            
            # Check validity
            if _valid(arr):
                s = _score(arr)
                if s > best_score:
                    best_score = s
                    best_params = res.x.copy()
                    
        except Exception:
            continue
    
    # If we have a valid best, use it
    if best_score > -np.inf:
        x_best = best_params[:n]
        y_best = best_params[n:2*n]
        r_best = np.maximum(0, best_params[2*n:])
        r_best = np.minimum(r_best, np.minimum(x_best, 1 - x_best))
        r_best = np.minimum(r_best, np.minimum(y_best, 1 - y_best))
        arr = np.column_stack([x_best, y_best, r_best])
        if _valid(arr):
            return arr
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])


if __name__ == '__main__':
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"Valid: {_valid(result)}")
