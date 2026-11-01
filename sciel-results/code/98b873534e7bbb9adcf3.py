"""Packing 50 equal circles in a unit circle - force-based optimization with simulated annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    for spacing in np.linspace(0.14, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            idx = np.random.permutation(len(pts))[:n]
            pts = pts[idx]
            if jitter > 0:
                pts += np.random.randn(n, 2) * jitter
            # Project back inside unit circle
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            return pts
    # Fallback: random
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _random_init(n):
    """Generate random initial positions within the unit disk."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _concentric_ring_init(n):
    """Generate initial positions on concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings
    remaining = n - 1
    ring_radius = 0.2
    while remaining > 0:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Minimum spacing ~ 0.25 for reasonable packing
        n_on_ring = max(1, int(circumference / 0.25))
        n_on_ring = min(n_on_ring, remaining)
        if n_on_ring == 0:
            ring_radius += 0.15
            continue
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            pts.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
        remaining -= n_on_ring
        ring_radius += 0.18
    pts = np.array(pts[:n])
    # Ensure inside unit circle
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        pts[mask] /= r[mask, None] * 1.001
    return pts


def _energy(pts):
    """Energy function: negative of minimum distance (to maximize)."""
    if len(pts) != N:
        return 1e6
    if not np.all(np.isfinite(pts)):
        return 1e6
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 1e6
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 1e6
    if N < 2:
        return -r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return -min(r_cont, r_pair)


def _force_optimize(pts, max_iters=20000, dt_init=0.05, dt_min=1e-6, decay=0.9995):
    """Force-based optimization to maximize minimum distance."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    dt = dt_init
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Precompute indices for pairwise
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute all pairwise distances
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        
        # Repulsion force
        force_mag = 1.0 / (d * d)
        fx = force_mag * dx / d
        fy = force_mag * dy / d
        
        # Accumulate forces
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        np.add.at(Fx, i_idx, -fx)
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fy, j_idx, fy)
        
        # Attraction toward center
        r = np.hypot(pts[:, 0], pts[:, 1])
        r_safe = np.maximum(r, 1e-12)
        center_force = 0.1 * (r_safe > 0.5).astype(float)
        Fx -= center_force * pts[:, 0] / r_safe
        Fy -= center_force * pts[:, 1] / r_safe
        
        # Normalize forces
        fmag = np.hypot(Fx, Fy)
        fmag_max = np.max(fmag)
        if fmag_max > 10.0:
            scale = 10.0 / fmag_max
            Fx *= scale
            Fy *= scale
        
        # Adaptive step size
        new_pts = pts + dt * np.column_stack([Fx, Fy])
        
        # Keep inside container
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001
        
        new_score = _score(new_pts)
        
        if new_score >= best_score - 1e-12:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            dt = min(dt * 1.01, 0.1)
        else:
            dt *= decay
        
        if dt < dt_min:
            break
    
    return best_pts, best_score


def _simulated_annealing(pts, max_iters=5000, T_init=0.1, T_min=1e-4, cooling=0.999):
    """Simulated annealing to escape local minima."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    T = T_init
    best_score = _score(pts)
    best_pts = pts.copy()
    current_score = best_score
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Random perturbation
        scale = 0.05 * T / T_init + 0.001
        new_pts = pts + np.random.randn(n, 2) * scale
        
        # Keep inside container
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001
        
        new_score = _score(new_pts)
        
        # Accept or reject
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta > 0 and np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_score


def _nelder_mead_optimize(pts):
    """Use scipy Nelder-Mead to optimize the packing."""
    def objective(flat_pts):
        p = flat_pts.reshape(N, 2)
        # Penalty for being outside
        r = np.hypot(p[:, 0], p[:, 1])
        penalty = np.sum(np.maximum(r - 1.0, 0.0) ** 2) * 100
        return _energy(p) + penalty
    
    flat = pts.flatten()
    try:
        result = minimize(objective, flat, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        optimized = result.x.reshape(N, 2)
        # Project inside
        r = np.hypot(optimized[:, 0], optimized[:, 1])
        mask = r > 1.0
        if np.any(mask):
            optimized[mask] /= r[mask, None] * 1.0001
        return optimized
    except:
        return pts


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations and optimization strategies
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Choose initialization strategy
        if restart == 0:
            pts = _hex_lattice_init(N, jitter=0.0)
        elif restart == 1:
            pts = _hex_lattice_init(N, jitter=0.02)
        elif restart == 2:
            pts = _concentric_ring_init(N)
        elif restart == 3:
            pts = _random_init(N)
        elif restart == 4:
            pts = _hex_lattice_init(N, jitter=0.05)
        elif restart == 5:
            pts = _concentric_ring_init(N) + np.random.randn(N, 2) * 0.03
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
        else:
            # Random perturbations of hex lattice
            base = _hex_lattice_init(N, jitter=0.0)
            pts = base + np.random.randn(N, 2) * 0.04 * (restart - 5)
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
        
        # Run force optimization
        optimized_pts, score = _force_optimize(
            pts,
            max_iters=10000,
            dt_init=0.03,
            dt_min=1e-7,
            decay=0.999
        )
        
        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()
        
        # If we have time, try simulated annealing from the best point
        if time.time() < _DEADLINE - 20.0 and restart % 2 == 0:
            sa_pts, sa_score = _simulated_annealing(
                optimized_pts,
                max_iters=3000,
                T_init=0.08,
                T_min=1e-4,
                cooling=0.9995
            )
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()
        
        # If we have time, try Nelder-Mead refinement
        if time.time() < _DEADLINE - 15.0 and restart in [0, 1, 3]:
            nm_pts = _nelder_mead_optimize(best_pts)
            nm_score = _score(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_lattice_init(N, jitter=0.0)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001
    
    return best_pts
