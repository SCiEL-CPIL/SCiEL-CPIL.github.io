import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    penalty += (r[i] + r[j] - dist)**2
        
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i]))**2
        
        return -np.sum(r) + 1000.0 * penalty

    def gradient(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        grad = np.zeros_like(params)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dist = 1e-12
                if dist < r[i] + r[j]:
                    diff = r[i] + r[j] - dist
                    grad[3*i] += 2000.0 * diff * (dx / dist)
                    grad[3*i+1] += 2000.0 * diff * (dy / dist)
                    grad[3*i+2] += 2000.0 * diff
                    grad[3*j] -= 2000.0 * diff * (dx / dist)
                    grad[3*j+1] -= 2000.0 * diff * (dy / dist)
                    grad[3*j+2] += 2000.0 * diff
        
        for i in range(N):
            if x[i] < r[i]:
                grad[3*i] -= 2000.0 * (r[i] - x[i])
                grad[3*i+2] += 2000.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                grad[3*i] += 2000.0 * (x[i] - (1 - r[i]))
                grad[3*i+2] -= 2000.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                grad[3*i+1] -= 2000.0 * (r[i] - y[i])
                grad[3*i+2] += 2000.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                grad[3*i+1] += 2000.0 * (y[i] - (1 - r[i]))
                grad[3*i+2] -= 2000.0 * (y[i] - (1 - r[i]))
        
        grad[2::3] -= 1
        
        return grad

    def get_initial_params_hex():
        rows = [5, 6, 5, 5, 5]
        centers = []
        for i, count in enumerate(rows):
            for j in range(count):
                x = j * 1.0 + (i % 2) * 0.5
                y = i * 0.866
                centers.append((x, y))
        
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        scale = min(0.9 / width, 0.9 / height)
        offset_x = (1 - width * scale) / 2
        offset_y = (1 - height * scale) / 2
        
        params = np.zeros(3 * N)
        for i, (x, y) in enumerate(centers):
            px = (x - min_x) * scale + offset_x
            py = (y - min_y) * scale + offset_y
            params[3*i] = px
            params[3*i+1] = py
            params[3*i+2] = 0.02
        return params

    def get_initial_params_random():
        params = np.zeros(3 * N)
        params[0::3] = np.random.rand(N)
        params[1::3] = np.random.rand(N)
        params[2::3] = 0.01
        return params

    best_score = -np.inf
    best_params = None

    # Try hex first
    if time.time() < deadline:
        try:
            res = minimize(objective, get_initial_params_hex(), jac=gradient, method='L-BFGS-B', 
                           bounds=[(0, 1)]*N + [(0, 1)]*N + [(1e-6, 0.5)]*N,
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
            score = np.sum(res.x[2::3])
            valid = True
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            for i in range(N):
                if x[i] < r[i] - 1e-8 or x[i] > 1 - r[i] + 1e-8 or y[i] < r[i] - 1e-8 or y[i] > 1 - r[i] + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-8:
                            valid = False
                            break
                    if not valid: break
            if valid and score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    # Random restarts
    while time.time() < deadline:
        try:
            res = minimize(objective, get_initial_params_random(), jac=gradient, method='L-BFGS-B', 
                           bounds=[(0, 1)]*N + [(0, 1)]*N + [(1e-6, 0.5)]*N,
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
            score = np.sum(res.x[2::3])
            valid = True
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            for i in range(N):
                if x[i] < r[i] - 1e-8 or x[i] > 1 - r[i] + 1e-8 or y[i] < r[i] - 1e-8 or y[i] > 1 - r[i] + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-8:
                            valid = False
                            break
                    if not valid: break
            if valid and score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    return np.column_stack((x, y, r))
