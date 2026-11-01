"""Pack 50 equal circles in the unit disk via simulated annealing + local refinement."""
import time
import numpy as np

N = 50
_DEADLINE = 140.0  # seconds wall-clock safety margin before the 150 s kill


def _score(pts):
    """Return the feasible common radius for a placement, or 0.0 if invalid."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(d >= 1.0):
        return 0.0
    # pairwise distances
    dx = pts[:, 0, None] - pts[None, :, 0]
    dy = pts[:, 1, None] - pts[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_boundary = 1.0 - d.max()
    r = min(min_pair / 2.0, min_boundary)
    return max(r, 0.0)


def _initial_placement():
    """A reasonably good starting layout: rings of circles."""
    # Use a concentric-ring initialization
    pts = np.zeros((N, 2))
    # Center
    pts[0] = [0.0, 0.0]
    remaining = N - 1
    ring_idx = 1
    count = 0
    while count < remaining:
        # Try to place a ring; radius grows
        ring_r = 0.15 + 0.12 * (ring_idx - 1)
        if ring_r >= 0.95:
            # Just scatter remaining in a small circle
            angles = np.linspace(0, 2 * np.pi, remaining - count, endpoint=False)
            for k in range(remaining - count):
                pts[count + 1 + k] = [ring_r * np.cos(angles[k]), ring_r * np.sin(angles[k])]
            break
        n_in_ring = min(remaining - count, max(6, int(2 * np.pi * ring_r / 0.26)))
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + np.pi / n_in_ring
        for k in range(n_in_ring):
            pts[count + 1 + k] = [ring_r * np.cos(angles[k]), ring_r * np.sin(angles[k])]
        count += n_in_ring
        ring_idx += 1
    return pts


def _simulated_annealing(pts0, t_start, t_end, n_iter):
    """Run SA with harmonic-repulsion objective; return best placement."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Target radius we want to achieve (slightly above current best known)
    target_r = 0.1258254895304
    
    def objective(p):
        """Energy: penalize overlaps and boundary violations. Lower is better."""
        d = np.hypot(p[:, 0], p[:, 1])
        boundary_pen = np.maximum(0.0, d - (1.0 - target_r))
        # Pairwise
        dx = p[:, 0, None] - p[None, :, 0]
        dy = p[:, 1, None] - p[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, 1e10)
        overlap_pen = np.maximum(0.0, 2 * target_r - dist)
        # Use a soft penalty
        E = np.sum(boundary_pen ** 2) + np.sum(overlap_pen ** 2)
        return E
    
    E_curr = objective(pts)
    E_best = E_curr
    
    t = t_start
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        # Random perturbation
        step = 0.02 * (t / t_start) + 0.001
        delta = np.random.randn(N, 2) * step
        new_pts = pts + delta
        # Keep inside container roughly
        d = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = d > 1.0 - 0.01
        if np.any(mask):
            new_pts[mask] = pts[mask]
        
        E_new = objective(new_pts)
        dE = E_new - E_curr
        if dE < 0 or np.random.rand() < np.exp(-dE / max(t, 1e-12)):
            pts = new_pts
            E_curr = E_new
            if E_curr < E_best:
                E_best = E_curr
                best_pts = pts.copy()
                s = _score(pts)
                if s > best_score:
                    best_score = s
        t = t_start * (t_end / t_start) ** (i + 1) / n_iter
    
    return best_pts, best_score


def _local_refinement(pts, target_r):
    """Gradient-free local refinement: nudge points to increase minimum distance."""
    pts = pts.copy()
    best_score = _score(pts)
    
    for outer in range(300):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE - 2:
                break
            # Find the most constraining direction for point i
            d = np.hypot(pts[:, 0], pts[:, 1])
            # Boundary constraint
            r_i = np.hypot(pts[i, 0], pts[i, 1])
            if r_i > 1e-12:
                norm_dir = pts[i] / r_i
            else:
                norm_dir = np.array([1.0, 0.0])
            
            # Try small moves in several directions
            best_move = None
            best_local_score = best_score
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                direction = np.array([np.cos(angle), np.sin(angle)])
                for step in [0.002, 0.005, 0.01]:
                    candidate = pts.copy()
                    candidate[i] += direction * step
                    # Check if still inside
                    d_new = np.hypot(candidate[i, 0], candidate[i, 1])
                    if d_new > 1.0 - 0.001:
                        continue
                    s = _score(candidate)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_move = candidate.copy()
            
            if best_move is not None:
                pts = best_move
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return pts, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    best_pts = None
    best_score = 0.0
    
    # Initialization 1: ring-based
    pts0 = _initial_placement()
    best_pts, best_score = _simulated_annealing(pts0, 1.0, 0.01, 5000)
    
    if time.time() > _DEADLINE - 20:
        return best_pts
    
    # Initialization 2: hexagonal lattice (from seed, but better spaced)
    spacing = 0.25
    k = 10
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    if len(points) >= N:
        pts1 = points[np.argsort(radius, kind="stable")[:N]].copy()
        p1, s1 = _simulated_annealing(pts1, 0.5, 0.005, 3000)
        if s1 > best_score:
            best_score = s1
            best_pts = p1
    
    if time.time() > _DEADLINE - 15:
        return best_pts
    
    # Initialization 3: random
    pts2 = rng.uniform(-0.8, 0.8, (N, 2))
    # Ensure inside unit circle
    d = np.hypot(pts2[:, 0], pts2[:, 1])
    pts2 = pts2 * (0.8 / np.maximum(d, 1e-12))[:, None]
    p2, s2 = _simulated_annealing(pts2, 0.8, 0.005, 4000)
    if s2 > best_score:
        best_score = s2
        best_pts = p2
    
    if time.time() > _DEADLINE - 10:
        return best_pts
    
    # Local refinement on the best
    target_r = best_score
    refined_pts, refined_score = _local_refinement(best_pts, target_r)
    if refined_score > best_score:
        best_pts = refined_pts
        best_score = refined_score
    
    return best_pts
