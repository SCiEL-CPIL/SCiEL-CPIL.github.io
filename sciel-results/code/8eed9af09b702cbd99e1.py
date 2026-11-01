"""Packing 50 equal circles in a unit circle - multi-start force-directed with refinement."""
import time
import numpy as np

N = 50
_START = time.time()
_DEADLINE = _START + 130.0  # 20s safety margin before 150s kill


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


def _hex_lattice_init(n, jitter=0.0, seed=None):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    rng = np.random.default_rng(seed)
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
            # Select the n points closest to the center for a denser core
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            pts = pts[idx]
            if jitter > 0:
                pts = pts + rng.normal(0, jitter, size=(n, 2))
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            return pts
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _concentric_ring_init(n, seed=None):
    """Initialize with concentric rings for a structured starting point."""
    rng = np.random.default_rng(seed)
    pts = []
    pts.append((0.0, 0.0))
    remaining = n - 1
    ring_radius = 0.12
    while remaining > 0 and len(pts) < n:
        circumference = 2 * np.pi * ring_radius
        max_pts = max(1, int(circumference / 0.22))
        pts_on_ring = min(max_pts, remaining)
        for k in range(pts_on_ring):
            angle = 2 * np.pi * k / pts_on_ring + rng.uniform(-0.1, 0.1)
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            if x * x + y * y <= 1.0:
                pts.append((x, y))
                remaining -= 1
        ring_radius += 0.12
    pts = np.array(pts[:n])
    if len(pts) < n:
        extra = n - len(pts)
        theta = rng.uniform(0, 2 * np.pi, extra)
        rho = np.sqrt(rng.uniform(0, 1, extra))
        extra_pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        pts = np.vstack([pts, extra_pts])
    return pts


def _force_optimize(pts, r_target, max_iters=20000, dt_init=0.01):
    """Force-directed optimization to maximize minimum separation."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    dt = dt_init
    best_score = _score(pts)
    best_pts = pts.copy()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Compute all pairwise distances
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        d = np.maximum(d, 1e-12)
        
        # Strong repulsion for close pairs, weak for far pairs
        # Use 1/d^6 for very strong short-range repulsion
        force_mag = 1.0 / (d2 * d2 * d2)  # 1/d^6
        fx = force_mag * dx / d
        fy = force_mag * dy / d
        
        # Accumulate forces
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        np.add.at(Fx, i_idx, -fx)
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fy, j_idx, fy)
        
        # Weak centering force
        Fx -= 0.005 * pts[:, 0]
        Fy -= 0.005 * pts[:, 1]
        
        # Normalize forces
        fmag = np.hypot(Fx, Fy)
        fmag_max = np.max(fmag)
        if fmag_max > 1.0:
            Fx /= fmag_max
            Fy /= fmag_max
        
        # Adaptive step
        current_r = _score(pts)
        step = dt * np.minimum(1.0, 5.0 * current_r)
        
        new_pts = pts + step * np.column_stack([Fx, Fy])
        
        # Hard constraint: keep inside unit disk
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-6)
        
        new_score = _score(new_pts)
        
        if new_score >= best_score - 1e-12:
            pts = new_pts
            if new_score > best_score + 1e-14:
                best_score = new_score
                best_pts = pts.copy()
            dt *= 0.9997
        else:
            dt *= 0.995
        
        if dt < 1e-9:
            break
    
    return best_pts, best_score


def _local_refine(pts, r_target, max_iters=5000):
    """Local refinement using small random perturbations."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Pick a random subset of points to perturb
        k = np.random.randint(1, 6)
        idx = np.random.choice(n, k, replace=False)
        
        # Small random perturbation
        perturbation = np.random.normal(0, 0.005, size=(k, 2))
        new_pts = pts.copy()
        new_pts[idx] += perturbation
        
        # Enforce containment
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-6)
        
        new_score = _score(new_pts)
        
        if new_score > best_score + 1e-14:
            pts = new_pts
            best_score = new_score
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initialization strategies with different seeds
    initializations = []
    for seed in range(20):
        initializations.append(_hex_lattice_init(N, jitter=0.0, seed=seed))
    for seed in range(10):
        initializations.append(_hex_lattice_init(N, jitter=0.02, seed=seed))
    for seed in range(10):
        initializations.append(_concentric_ring_init(N, seed=seed))
    for seed in range(10):
        initializations.append(_hex_lattice_init(N, jitter=0.05, seed=seed))
    
    # Shuffle initializations
    np.random.shuffle(initializations)
    
    for init_pts in initializations:
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Phase 1: Force-directed optimization
        optimized_pts, score = _force_optimize(init_pts, r_target=0.12, max_iters=15000, dt_init=0.01)
        
        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()
        
        # Phase 2: Local refinement
        if time.time() < _DEADLINE - 15.0:
            refined_pts, refined_score = _local_refine(best_pts, r_target=best_score, max_iters=3000)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_lattice_init(N, jitter=0.0, seed=0)
        best_score = _score(best_pts)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-6)
    
    return best_pts
