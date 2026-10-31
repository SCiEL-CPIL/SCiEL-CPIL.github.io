"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

# Triangle geometry
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _cart_to_bary(x, y):
    """Convert Cartesian to barycentric coordinates."""
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return a, b, c

def _bary_to_cart(a, b, c):
    """Convert barycentric to Cartesian."""
    x = b + 0.5 * c
    y = H * c
    return x, y

def _clamp_bary(a, b, c, eps=1e-10):
    """Clamp barycentric coords to be inside the triangle."""
    a = max(eps, a)
    b = max(eps, b)
    c = max(eps, c)
    s = a + b + c
    return a / s, b / s, c / s

def _clamp_to_triangle(p):
    """Clamp a single point to be inside the equilateral triangle."""
    x, y = p
    a, b, c = _cart_to_bary(x, y)
    a, b, c = _clamp_bary(a, b, c)
    return np.array(_bary_to_cart(a, b, c))

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    out = np.empty_like(pts)
    for i in range(len(pts)):
        out[i] = _clamp_to_triangle(pts[i])
    return out

def _min_area(pts):
    """Compute minimum triangle area from Cartesian points (N x 2 array)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i]))
                if area < min_a:
                    min_a = area
                    if min_a < 1e-14:
                        return 0.0
    return min_a

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i]))
                log_area = np.log(area + 1e-15)
                log_sum = np.logaddexp(log_sum, log_area)
    return log_sum / beta

def _objective_hard(flat):
    """Hard objective for optimization (negative min area)."""
    pts = flat.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _objective_soft(flat, beta=50.0):
    """Soft objective for optimization."""
    pts = flat.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _initial_placement_symmetric():
    """Symmetric initial placement using triangular lattice."""
    # Use a structured approach: place points in a hexagonal-like pattern
    # 11 points: center + 6 inner ring + 4 outer
    pts = []
    # Center
    pts.append([0.5, H / 3])
    # Inner ring (6 points) at radius ~0.2
    r1 = 0.2
    for i in range(6):
        angle = np.pi / 3 * i + np.pi / 6
        x = 0.5 + r1 * np.cos(angle)
        y = H / 3 + r1 * np.sin(angle) * 0.8  # Squash y to fit triangle
        pts.append([x, y])
    # Outer ring (4 points)
    r2 = 0.38
    angles_outer = [np.pi / 2, np.pi / 2 + np.pi / 2, np.pi / 2 + np.pi, np.pi / 2 + 3 * np.pi / 2]
    for angle in angles_outer:
        x = 0.5 + r2 * np.cos(angle)
        y = H / 3 + r2 * np.sin(angle) * 0.7
        pts.append([x, y])
    pts = np.array(pts)
    return _clamp_all(pts)

def _initial_placement_grid():
    """Grid-based initial placement."""
    pts = []
    # 4 rows: 3+3+3+2 = 11
    rows = [
        (0.1, 0.15, 0.45, 0.75, 0.9),   # 5 points
        (0.2, 0.35, 0.65, 0.8),          # 4 points  
        (0.3, 0.5, 0.7),                 # 3 points
        (0.45, 0.55),                     # 2 points
    ]
    # Actually let's do 3+4+4 = 11 or similar
    # Let's use a cleaner approach
    pts = []
    y_vals = [0.08, 0.2, 0.35, 0.5, 0.65]
    counts = [4, 3, 3, 2, 1]  # 13, too many
    # Use 3+3+3+2 = 11
    y_vals = [0.1, 0.3, 0.5, 0.68]
    counts = [3, 3, 3, 2]
    for idx, (y, count) in enumerate(zip(y_vals, counts)):
        for i in range(count):
            x = 0.15 + i * (0.7 / max(count - 1, 1)) if count > 1 else 0.5
            pts.append([x, y])
    pts = np.array(pts)
    return _clamp_all(pts)

def _nelder_mead_polish(pts, maxiter=200):
    """Nelder-Mead refinement."""
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return _clamp_all(res.x.reshape(-1, 2))
    except Exception:
        return pts

def _l_bfgs_polish(pts, maxiter=500):
    """L-BFGS-B refinement."""
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return _clamp_all(res.x.reshape(-1, 2))
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill

    # Generate multiple diverse initializations
    initializations = []
    initializations.append(_initial_placement_symmetric())
    initializations.append(_initial_placement_grid())
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((N, 2)) * np.array([1.0, H])
        # Ensure some are in valid region
        pts = _clamp_all(pts)
        initializations.append(pts)
    
    # Latin hypercube initialization
    for _ in range(3):
        x = np.linspace(0.05, 0.95, N) + rng.random(N) * 0.1
        y = np.linspace(0.05, H * 0.9, N) + rng.random(N) * 0.1
        pts = np.column_stack([x, y])
        pts = _clamp_all(pts)
        initializations.append(pts)
    
    # Evaluate all initializations
    best_pts = None
    best_score = -np.inf
    for init in initializations:
        score = _min_area(init)
        if score > best_score:
            best_score = score
            best_pts = init.copy()
    
    # Strategy 1: Differential Evolution for global search
    if time.time() - start_time < deadline - 40:
        try:
            bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
            de_res = differential_evolution(
                _objective_hard,
                bounds=bounds,
                seed=42,
                maxiter=60,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
            )
            de_pts = _clamp_all(de_res.x.reshape(-1, 2))
            de_score = _min_area(de_pts)
            if de_score > best_score:
                best_pts = de_pts
                best_score = de_score
        except Exception:
            pass
    
    # Strategy 2: Multiple L-BFGS refinements from different starts
    for init in initializations[:6]:
        if time.time() - start_time > deadline - 25:
            break
        try:
            polished = _l_bfgs_polish(init, maxiter=300)
            score = _min_area(polished)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
    
    # Strategy 3: Soft objective optimization to escape local minima
    for beta in [20.0, 40.0, 80.0]:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(
                lambda x: _objective_soft(x, beta),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            soft_pts = _clamp_all(res.x.reshape(-1, 2))
            score = _min_area(soft_pts)
            if score > best_score:
                best_score = score
                best_pts = soft_pts
        except Exception:
            pass
    
    # Strategy 4: Nelder-Mead polish
    if time.time() - start_time < deadline - 15:
        nm_pts = _nelder_mead_polish(best_pts, maxiter=300)
        nm_score = _min_area(nm_pts)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts
    
    # Strategy 5: Multiple random restarts with L-BFGS
    for trial in range(20):
        if time.time() - start_time > deadline - 12:
            break
        noise_scale = 0.06
        trial_pts = best_pts + rng.normal(0, noise_scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        try:
            polished = _l_bfgs_polish(trial_pts, maxiter=200)
            score = _min_area(polished)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
    
    # Strategy 6: Simulated annealing for fine-tuning
    if time.time() - start_time < deadline - 10:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.01
        for iteration in range(5000):
            if time.time() - start_time > deadline - 3:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            new_score = _min_area(new_pts)
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            T *= 0.9997
    
    # Strategy 7: Final high-precision polish
    if time.time() - start_time < deadline - 3:
        try:
            final_pts = _l_bfgs_polish(best_pts, maxiter=1000)
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass
    
    # Final clamp
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
