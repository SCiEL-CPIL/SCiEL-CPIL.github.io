"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

# Precompute all triangle indices for N=13: C(13,3)=286
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

# Halton sequence generator
def _halton(n, dim):
    """Generate n points in [0,1)^dim using Halton sequence."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    bases = primes[:dim]
    pts = np.zeros((n, dim))
    for d, b in enumerate(bases):
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                f /= b
                r += f * (i_val % b)
                i_val //= b
            pts[i - 1, d] = r
    return pts


def _min_triangle_area_raw(pts):
    """Compute minimum triangle area for n points (vectorized)."""
    n = pts.shape[0]
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    # Area = 0.5 * |cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area via shoelace on convex hull points
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(pts)
            return h.volume
        except Exception:
            return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area_raw(pts)
    return min_ta / hull_a


def _soft_min_area(pts, beta):
    """Soft-min surrogate for minimum triangle area (vectorized)."""
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract min
    m = np.min(areas)
    if m < 1e-15:
        return m
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    return m - (1.0 / beta) * log_sum


def _optimize_from_start(start_pts, deadline, max_beta_steps=8):
    """Run L-BFGS-B with increasing beta schedule from a starting point."""
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    beta_schedule = [10, 25, 50, 100, 200, 400, 800, 1600]
    beta_schedule = beta_schedule[:max_beta_steps]
    
    current = start_pts.copy().flatten()
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        def objective(x):
            pts = x.reshape(N, 2)
            return -_soft_min_area(pts, beta)
        
        try:
            result = minimize(
                objective,
                current,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current = result.x
            pts = current.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = 0.0
    
    # Generate diverse starting points
    starts = []
    
    # 1. Halton sequence
    halton_pts = _halton(N, 2)
    starts.append(halton_pts)
    
    # 2. Regular polygon on unit circle, centered
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(poly_pts)
    
    # 3. Random starts with different seeds
    for seed in [42, 123, 777, 2024]:
        rng = np.random.RandomState(seed)
        rand_pts = rng.rand(N, 2)
        starts.append(rand_pts)
    
    # 4. Halton with slight perturbation
    for perturb_seed in [1, 2, 3]:
        rng = np.random.RandomState(perturb_seed)
        perturbed = halton_pts + rng.randn(N, 2) * 0.05
        starts.append(perturbed)
    
    # 5. Grid-like starts
    grid1 = np.array([[i / 4.0, j / 4.0] for i in range(4) for j in range(4)])[:N]
    starts.append(grid1)
    
    # 6. Concentric rings
    ring1 = np.column_stack([np.cos(angles * 0.5), np.sin(angles * 0.5)])
    ring2 = np.column_stack([0.5 * np.cos(angles * 0.5 + np.pi / N), 0.5 * np.sin(angles * 0.5 + np.pi / N)])
    ring_pts = np.vstack([ring1[:7], ring2[:6]])
    starts.append(ring_pts)
    
    # Evaluate and optimize each start
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Evaluate start
        s = _score(start)
        if s > best_score:
            best_score = s
            best_pts = start.copy()
        
        # Optimize
        opt_pts, opt_score = _optimize_from_start(start, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
        
        # If we're running low on time, do fewer beta steps for remaining starts
        remaining = deadline - time.time()
        if remaining < 30:
            break
    
    # Final refinement: one more optimization pass with high beta on best point
    if best_pts is not None and time.time() < deadline - 5:
        def final_objective(x):
            pts = x.reshape(N, 2)
            return -_soft_min_area(pts, 2000)
        
        try:
            result = minimize(
                final_objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            final_pts = result.x.reshape(N, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Halton
        best_pts = _halton(N, 2)
    
    return best_pts
# EVOLVE-BLOCK-END
