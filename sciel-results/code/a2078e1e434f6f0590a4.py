"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    # Precompute all triples
    indices = list(itertools.combinations(range(N), 3))
    idx_array = np.array(indices, dtype=np.int32)
    p = pts[idx_array]  # (286, 3, 2)
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(pts)
    return min_tri / hull_a

def _soft_min_triangle(pts, beta):
    """
    Compute a smooth surrogate for the minimum triangle area using soft-min.
    As beta -> infinity, this approaches the true minimum.
    """
    pts = np.asarray(pts, dtype=np.float64)
    indices = list(itertools.combinations(range(N), 3))
    idx_array = np.array(indices, dtype=np.int32)
    p = pts[idx_array]
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -beta * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    m = np.min(areas)
    if m < 1e-15:
        return m
    exp_terms = np.exp(-beta * (areas - m))
    log_sum = np.log(np.sum(exp_terms))
    return m - (1.0 / beta) * log_sum

def _jittered_grid(seed=42):
    """Generate a jittered grid initial guess."""
    rng = np.random.RandomState(seed)
    side = 4
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            x += rng.uniform(-0.08, 0.08)
            y += rng.uniform(-0.08, 0.08)
            points.append([x, y])
    return np.array(points[:N])

def _regular_polygon(n, radius=1.0, center=(0.5, 0.5), seed=0):
    """Generate points on a regular polygon with slight jitter."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    # Add small jitter
    x += rng.normal(0, 0.02, n)
    y += rng.normal(0, 0.02, n)
    return np.column_stack([x, y])

def _halton_sequence(n, base, seed=0):
    """Generate Halton sequence in [0,1]^2."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(1, n + 1):
        # Base 2
        x = 0
        p = 1
        i2 = i
        while i2 > 0:
            i2, r = divmod(i2, base)
            x += r / p
            p *= base
        # Base 3
        y = 0
        p = 1
        i3 = i
        while i3 > 0:
            i3, r = divmod(i3, 3)
            y += r / p
            p *= 3
        points.append([x, y])
    return np.array(points)

def _simulated_annealing(pts_init, deadline, max_iter=80000, T0=0.1, cooling=0.9995, seed=12345):
    """
    Simulated Annealing for local polishing.
    """
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    T = T0
    
    rng = np.random.RandomState(seed)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Move one point
        idx = rng.randint(N)
        step = rng.normal(0, T, size=2)
        new_pts = pts.copy()
        new_pts[idx] += step
        new_pts[idx] = np.clip(new_pts[idx], -2.0, 2.0)
        
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(delta / max(T, 1e-10))
            if rng.random() < prob:
                pts = new_pts
                current_score = new_score
        
        T *= cooling
        if T < 1e-8:
            T = 1e-8
            
    return best_pts, best_score

def _polish_nelder_mead(pts_init, deadline, max_iter=5000, seed=0):
    """
    Polish using Nelder-Mead on the negative score.
    """
    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        s = _score(pts)
        return -s
    
    x0 = pts_init.reshape(-1)
    
    try:
        result = minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_pts = result.x.reshape(N, 2)
        best_score = -result.fun
        return best_pts, best_score
    except Exception:
        return pts_init, _score(pts_init)

def _soft_min_optimize(pts_init, deadline, beta_start=10, beta_end=500, n_steps=20, seed=0):
    """
    Optimize using a soft-min surrogate with increasing beta.
    """
    pts = pts_init.copy().astype(np.float64)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        # Interpolate beta
        beta = beta_start * (beta_end / beta_start) ** (step / max(n_steps - 1, 1))
        
        def neg_soft_min(flat_pts):
            p = flat_pts.reshape(N, 2)
            # Soft min triangle area, plus a penalty for hull area
            sm = _soft_min_triangle(p, beta)
            hull_a = _hull_area(p)
            if hull_a < 1e-10:
                return 1e6
            # We want to maximize sm / hull_a, so minimize -sm/hull_a
            # But soft_min is not exactly min, so use -sm + penalty if hull too small
            # Actually, just minimize -sm / hull_a
            val = -sm / hull_a
            return val
        
        x0 = pts.reshape(-1)
        try:
            result = minimize(
                neg_soft_min,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    return pts, _score(pts)

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0
    
    best_pts = None
    best_score = 0.0
    
    # Generate multiple initializations
    inits = []
    inits.append(_jittered_grid(seed=42))
    inits.append(_jittered_grid(seed=123))
    inits.append(_regular_polygon(N, radius=0.5, center=(0.5, 0.5), seed=0))
    inits.append(_regular_polygon(N, radius=0.5, center=(0.5, 0.5), seed=1))
    inits.append(_halton_sequence(N, base=2, seed=0))
    inits.append(_halton_sequence(N, base=2, seed=1))
    
    # Add random inits
    rng = np.random.RandomState(999)
    for _ in range(3):
        pts = rng.uniform(0, 1, size=(N, 2))
        inits.append(pts)
    
    # Phase 1: For each init, run soft-min optimization then SA
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        # Soft-min optimization
        try:
            sm_pts, sm_score = _soft_min_optimize(init_pts, deadline, beta_start=10, beta_end=300, n_steps=15, seed=i)
            if sm_score > best_score:
                best_score = sm_score
                best_pts = sm_pts
        except Exception:
            pass
        
        # Simulated Annealing
        if time.time() > deadline - 5.0:
            break
        try:
            sa_pts, sa_score = _simulated_annealing(best_pts if best_pts is not None else init_pts, deadline, max_iter=40000, T0=0.05, cooling=0.9995, seed=1000 + i)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass
    
    # Phase 2: Final polish with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            nm_pts, nm_score = _polish_nelder_mead(best_pts, deadline, max_iter=3000, seed=0)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts
        except Exception:
            pass
    
    # Phase 3: If time remains, try more SA restarts
    if best_pts is not None and time.time() < deadline - 2.0:
        for i in range(3):
            if time.time() > deadline - 1.0:
                break
            try:
                sa_pts, sa_score = _simulated_annealing(best_pts, deadline, max_iter=30000, T0=0.02, cooling=0.9997, seed=5000 + i)
                if sa_score > best_score:
                    best_score = sa_score
                    best_pts = sa_pts
            except Exception:
                pass
    
    if best_pts is not None:
        return best_pts
    
    return _jittered_grid(N)

# EVOLVE-BLOCK-END
