import numpy as np
import time

N = 300
DEADLINE = 140.0  # Safety margin for 150s limit


def _score(pts):
    """Calculate the valid score: largest common radius."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Boundary distance
    dists_center = np.linalg.norm(pts, axis=1)
    min_boundary = np.min(1.0 - dists_center)
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dists[iu])
    
    r = min(min_boundary, min_pair_dist / 2.0)
    return r


def _lattice_seed():
    """Generate initial hexagonal lattice points."""
    # Approximate spacing for N circles in unit disk
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    points = []
    k = int(1.0 / s) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # Select N points closest to center
    if len(points) >= N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        return points[idx].copy()
    else:
        # Fallback: pad with random points
        while len(points) < N:
            p = np.random.uniform(-1, 1, size=2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, [p]])
        return points[:N].copy()


def _anneal(pts, start_time, deadline):
    """Vectorized simulated annealing with time guard."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Initial temperature based on current score
    current_score = best_score
    T = 0.005
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.01
    
    while time.time() < deadline and T > T_min:
        # Generate random perturbations
        n_perturb = 5
        perturb_idx = np.random.randint(0, N, size=n_perturb)
        deltas = np.random.normal(0, step_size, size=(n_perturb, 2))
        
        new_pts = pts.copy()
        new_pts[perturb_idx] += deltas
        
        # Keep points inside the unit circle
        norms = np.linalg.norm(new_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis] * 0.999
        
        new_score = _score(new_pts)
        
        # Accept if better or with Metropolis probability
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = current_score - new_score
            if np.random.random() < np.exp(-delta / T):
                pts = new_pts
                current_score = new_score
        
        T *= cooling
        # Slowly decrease step size
        step_size *= 0.999
        
        # Periodic restart: if stuck, add noise to escape local minima
        if np.random.random() < 0.001:
            pts = best_pts + np.random.normal(0, 0.02, size=(N, 2))
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
    
    return best_pts, best_score


def _polish(pts):
    """Local refinement using gradient-free optimization on worst pairs."""
    for _ in range(50):
        score = _score(pts)
        if score <= 0:
            break
        
        # Find the minimum pair distance and boundary
        dists_center = np.linalg.norm(pts, axis=1)
        min_boundary = np.min(1.0 - dists_center)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(N, k=1)
        min_pair_dist = np.min(dists[iu])
        
        r = min(min_boundary, min_pair_dist / 2.0)
        
        # Try to improve the worst constraint
        if min_boundary < min_pair_dist / 2.0:
            # Boundary is limiting: push points inward slightly
            worst_idx = np.argmin(dists_center)
            pts[worst_idx] *= 0.999
        else:
            # Pairwise is limiting: push apart the closest pair
            flat_idx = np.argmin(dists[iu])
            i, j = iu[0][flat_idx], iu[1][flat_idx]
            diff_vec = pts[i] - pts[j]
            norm = np.linalg.norm(diff_vec)
            if norm > 1e-10:
                direction = diff_vec / norm
                pts[i] += direction * 0.001
                pts[j] -= direction * 0.001
        
        # Keep inside circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
        
        new_score = _score(pts)
        if new_score < r * 0.99:
            # Revert if it got worse
            break
    
    return pts, _score(pts)


def construct():
    start_time = time.time()
    
    # Generate initial lattice
    pts = _lattice_seed()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Run simulated annealing
    remaining_deadline = start_time + DEADLINE - 5.0  # Leave 5s for polish
    best_pts, best_score = _anneal(pts, start_time, remaining_deadline)
    
    # Polish the result
    if time.time() < start_time + DEADLINE - 2.0:
        polished_pts, polished_score = _polish(best_pts.copy())
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    return best_pts
