"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 16
D = 2

# EVOLVE-BLOCK-START
def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances for an (N, 2) array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=-1)

def _score(pts):
    """Return (max_dist / min_dist)^2. Lower is better."""
    sqd = _pairwise_sq_dists(pts)
    iu = np.triu_indices(N, 1)
    d2 = sqd[iu]
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-15:
        return 1e18
    ratio = np.sqrt(dmax) / np.sqrt(dmin)
    return ratio ** 2

def _objective(pts):
    """Objective for optimizer: minimize (max/min)^2."""
    d2 = _pairwise_sq_dists(pts)
    iu = np.triu_indices(N, 1)
    vals = d2[iu]
    dmin = np.min(vals)
    dmax = np.max(vals)
    if dmin < 1e-15:
        return 1e18
    return (dmax / dmin)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin
    
    best_pts = None
    best_score = np.inf
    
    def _update_best(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Strategy 1: Start with hexagonal/triangular lattice arrangement
    # For 16 points, try several structured initializations
    
    # Hexagonal lattice points
    def hex_lattice(n, spacing=1.0):
        pts = []
        row = 0
        while len(pts) < n:
            for col in range(n):
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * np.sqrt(3) / 2
                pts.append((x, y))
            row += 1
        return np.array(pts[:n])
    
    # Square grid
    def square_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append((i, j))
        return np.array(pts[:n])
    
    # Circular arrangement
    def circle(n, radius=1.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Two concentric circles
    def two_circles(n):
        n1 = 8
        n2 = n - n1
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
        pts1 = np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = 0.5 * np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([pts1, pts2])
    
    # Hexagonal ring + center
    def hex_ring(n):
        # Center + hexagonal rings
        pts = [(0.0, 0.0)]
        r = 1
        while len(pts) < n:
            for i in range(6):
                if len(pts) >= n:
                    break
                angle = i * np.pi / 3
                for j in range(1, r+1):
                    if len(pts) >= n:
                        break
                    x = j * r * np.cos(angle)
                    y = j * r * np.sin(angle)
                    pts.append((x, y))
            r += 1
        return np.array(pts[:n])
    
    initializations = [
        hex_lattice(N),
        square_grid(N),
        circle(N, 2.0),
        two_circles(N),
        hex_ring(N),
    ]
    
    # Also try random initializations
    np.random.seed(42)
    for _ in range(10):
        initializations.append(np.random.randn(N, 2) * 2.0)
    
    # Refine each initialization with local optimization
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                _objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            _update_best(res.x)
        except Exception:
            pass
    
    # Strategy 2: Differential evolution for global search
    if time.time() < deadline - 20.0:
        try:
            bounds = [(-3, 3)] * (N * 2)
            res_de = differential_evolution(
                lambda x: _objective(x.reshape(N, 2)),
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,
                update_interval=1
            )
            if res_de.success or res_de.fun < best_score:
                _update_best(res_de.x.reshape(N, 2))
        except Exception:
            pass
    
    # Strategy 3: Local refinement with multiple random restarts
    if time.time() < deadline - 15.0:
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            np.random.seed(seed * 100 + 7)
            x0 = np.random.randn(N, 2) * 1.5
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                _update_best(res.x)
            except Exception:
                pass
    
    # Strategy 4: Swap-based improvement on best configuration
    if best_pts is not None and time.time() < deadline - 5.0:
        current = best_pts.copy()
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                for j in range(i+1, N):
                    trial = current.copy()
                    trial[i], trial[j] = trial[j].copy(), trial[i].copy()
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        current = trial
                        best_score = s
                        best_pts = trial.copy()
                        improved = True
                        break
                if improved:
                    break
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(
                _objective,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            _update_best(res.x)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return a simple grid
        best_pts = square_grid(N)
    
    return best_pts
# EVOLVE-BLOCK-END
