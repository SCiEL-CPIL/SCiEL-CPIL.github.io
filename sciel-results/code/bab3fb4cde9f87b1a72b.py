import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid
xs = np.arange(-3, 1.5, 0.01)
ys = np.arange(-1.5, 1.5, 0.01)
GX, GY = np.meshgrid(xs, ys, indexing='ij')
GX = GX.ravel().astype(np.float64)
GY = GY.ravel().astype(np.float64)
N = len(GX)

def corridor_mask(x, y, a_deg):
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx = GX - x
    dy = GY - y
    u = dx*ca + dy*sa
    v = -dx*sa + dy*ca
    in_R1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_R2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_R1 | in_R2

def compute_area(path):
    """path: array of (x,y,a) for a=0..90 (91 rows). Returns sofa area."""
    inside = np.ones(N, dtype=bool)
    for (x, y, a) in path:
        inside &= corridor_mask(x, y, a)
        if not inside.any():
            break
    return inside.sum() * 0.0001

def path_from_params(params, d):
    """params: 2d array [p_1..p_d, q_1..q_d]. Returns path for a=0..90."""
    p = params[:d]
    q = params[d:2*d]
    angles = np.arange(0, 91)  # 0..90
    t = angles / 90.0
    # x(t) = -2t + sum p_k (t^k - t)
    # y(t) = sum q_k (t^k - t)
    x = -2*t
    y = np.zeros_like(t)
    for k in range(1, d+1):
        basis = t**k - t
        x += p[k-1] * basis
        y += q[k-1] * basis
    path = np.column_stack([x, y, angles.astype(float)])
    return path

def objective(params, d):
    path = path_from_params(params, d)
    return -compute_area(path)

# Initial guess: straight line (all params 0)
d = 6
x0 = np.zeros(2*d)

start_time = time.time()
deadline = 500  # seconds

result = minimize(objective, x0, args=(d,), method='Powell',
                  options={'maxiter': 200, 'maxfev': 1500, 'xtol': 1e-8, 'ftol': 1e-10})

best_params = result.x
best_area = -result.fun

# Check time
if time.time() - start_time < deadline:
    # Try a few more restarts
    for restart in range(5):
        if time.time() - start_time > deadline - 100:
            break
        x0_r = np.random.normal(0, 0.5, 2*d)
        res_r = minimize(objective, x0_r, args=(d,), method='Powell',
                         options={'maxiter': 100, 'maxfev': 500, 'xtol': 1e-8, 'ftol': 1e-10})
        if -res_r.fun > best_area:
            best_area = -res_r.fun
            best_params = res_r.x

best_path = path_from_params(best_params, d)

def moving_sofa_2d() -> np.ndarray:
    # Return the path rows for angles 1..89 (the evaluator adds 0 and 90)
    return best_path[1:90, :]

# But wait, the function must be self-contained and return the path. Let me restructure.
