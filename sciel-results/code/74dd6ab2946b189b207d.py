"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _repulsion_optimize(pts0, max_time=120.0):
    """Optimize using repulsion forces between circles."""
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Adaptive step size
    step_size = 0.01
    n_iter = 0
    stuck_count = 0
    
    while time.time() < _DEADLINE:
        n_iter += 1
        
        # Compute forces: repulsion between circles + attraction to boundary
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push
                    forces[i] += np.random.uniform(-0.01, 0.01, 2)
                    forces[j] -= np.random.uniform(-0.01, 0.01, 2)
                else:
                    # Strong repulsion when close
                    if dist < 0.3:
                        f = 1.0 / (dist * dist + 1e-6)
                        forces[i] += f * d / dist
                        forces[j] -= f * d / dist
        
        # Attraction toward the boundary (push circles outward)
        for i in range(N):
            r = np.linalg.norm(pts[i])
            if r < 0.9:
                # Push outward gently
                forces[i] += 0.1 * pts[i] / (r + 1e-10)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update positions
        new_pts = pts + step_size * forces
        
        # Keep points inside the container
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        for i in range(N):
            if new_radii[i] > 1.0:
                new_pts[i] = new_pts[i] / new_radii[i]
        
        # Check if valid
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            # Reduce step size if stuck
            if stuck_count > 50:
                step_size *= 0.9
                stuck_count = 0
        
        # Occasionally add random perturbation to escape local minima
        if n_iter % 200 == 0:
            perturb = np.random.uniform(-0.02, 0.02, (N, 2))
            new_pts = pts + perturb
            new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            for i in range(N):
                if new_radii[i] > 1.0:
                    new_pts[i] = new_pts[i] / new_radii[i]
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run repulsion-based optimization
    try:
        if time.time() < _DEADLINE:
            opt_pts = _repulsion_optimize(pts0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # Final safety check: ensure validity
    final_score = _score(best_pts)
    if final_score <= 0:
        # Return the initial hex lattice as fallback
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
