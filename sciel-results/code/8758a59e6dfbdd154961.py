import numpy as np
import time
from scipy.optimize import minimize

N = 32

def _check_validity(circles):
    """Check if circles are valid. Return True if valid, False otherwise."""
    if circles.shape != (N, 3):
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check bounds
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(r <= 0):
        return False
    if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            if d2 < sum_r*sum_r - 1e-9: # Small tolerance
                return False
    return True

def _fix_circles(circles):
    """Shrink radii slightly to ensure validity if there are tiny violations."""
    x, y, r = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
    
    # Ensure within bounds
    for i in range(N):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        
    # Ensure no overlaps by shrinking
    # This is a simple heuristic. If overlap, shrink both.
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if d < sum_r - 1e-9:
                    # Shrink
                    overlap = sum_r - d
                    # Distribute shrinkage
                    shrink = overlap / 2 + 1e-6
                    r[i] -= shrink
                    r[j] -= shrink
                    changed = True
                    # Ensure r > 0
                    r[i] = max(r[i], 1e-6)
                    r[j] = max(r[j], 1e-6)
                    
    return np.column_stack([x, y, r])

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    def hex_init():
        points = []
        k = 0
        while len(points) < N:
            l = 0
            while len(points) < N:
                x = l * np.sqrt(3) * 0.05 + (k % 2) * np.sqrt(3) * 0.5 * 0.05
                y = k * 1.5 * 0.05
                points.append([x, y])
                l += 1
            k += 1
        points = np.array(points[:N])
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = min(1.0 / width, 1.0 / height)
        points[:, 0] = (points[:, 0] - min_x) * scale + (1 - width * scale) / 2
        points[:, 1] = (points[:, 1] - min_y) * scale + (1 - height * scale) / 2
        radii = np.full(N, 0.01)
        return np.column_stack([points, radii])

    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        obj = -np.sum(r)
        penalty = 0.0
        w_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        penalty += np.sum(w_viol ** 2)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                viol = sum_r*sum_r - d2
                if viol > 0:
                    penalty += viol * viol
        return obj + penalty * 50.0

    def gradient(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        for i in range(N):
            if r[i] > x[i]:
                grad_x[i] += -2 * (r[i] - x[i])
                grad_r[i] += 2 * (r[i] - x[i])
            if r[i] > 1 - x[i]:
                grad_x[i] += 2 * (r[i] - (1 - x[i]))
                grad_r[i] += 2 * (r[i] - (1 - x[i]))
            if r[i] > y[i]:
                grad_y[i] += -2 * (r[i] - y[i])
                grad_r[i] += 2 * (r[i] - y[i])
            if r[i] > 1 - y[i]:
                grad_y[i] += 2 * (r[i] - (1 - y[i]))
                grad_r[i] += 2 * (r[i] - (1 - y[i]))
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                viol = sum_r*sum_r - d2
                if viol > 0:
                    factor = 2 * viol
                    grad_x[i] += factor * (-2 * dx)
                    grad_x[j] += factor * (2 * dx)
                    grad_y[i] += factor * (-2 * dy)
                    grad_y[j] += factor * (2 * dy)
                    grad_r[i] += factor * (2 * sum_r)
                    grad_r[j] += factor * (2 * sum_r)
        grad[0:N] = grad_x * 50.0
        grad[N:2*N] = grad_y * 50.0
        grad[2*N:3*N] = grad_r * 50.0 - 1.0
        return grad

    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    best_result = None
    best_score = -np.inf

    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        if attempt == 0:
            init = hex_init()
        else:
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.02)
            init = np.column_stack([x, y, r])
        try:
            result = minimize(objective, init, jac=gradient, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except Exception:
            pass
        if time.time() > deadline - 5:
            break

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    circles = best_result.x.reshape(N, 3)
    # Reorder to x, y, r
    circles = np.column_stack([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Fix any small violations
    if not _check_validity(circles):
        circles = _fix_circles(circles)
        
    return circles
