"""Differentiable optimal transport approach for circle packing."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_target_hex_lattice(n_rows, n_cols):
    """
    Generate a target hexagonal lattice pattern centered in [-0.5, 0.5]^2.
    Returns an array of shape (n_rows * n_cols, 2) with ideal hex positions,
    scaled to fit the square.
    """
    # Hex lattice with spacing s
    # We want the lattice to roughly fill the square
    # Width: (n_cols - 1) * s + s/2 (for offset rows)
    # Height: (n_rows - 1) * s * sqrt(3)/2
    
    # Let's parameterize so the lattice fits nicely
    # Try to make the bounding box approximately square
    # Width ≈ (n_cols - 0.5) * s
    # Height ≈ (n_rows - 1) * s * sqrt(3)/2
    
    # Set s so that width = height (approximately)
    # (n_cols - 0.5) * s = (n_rows - 1) * s * sqrt(3)/2
    # This gives a specific aspect ratio; instead, let's just generate and center
    
    s = 1.0  # unit spacing
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * s + (i % 2) * s / 2
            y = i * s * np.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    
    # Center the lattice
    cx = (points[:, 0].max() + points[:, 0].min()) / 2
    cy = (points[:, 1].max() + points[:, 1].min()) / 2
    points[:, 0] -= cx
    points[:, 1] -= cy
    
    # Scale to fit in [-0.5, 0.5]^2 with some margin
    # Find the max extent
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        scale = 0.45 / max_ext
        points *= scale
    
    return points

def _wasserstein_2_gradient(centers, target, eps=0.01):
    """
    Compute gradient of a smooth approximation of W2^2 between empirical measure
    of centers and target distribution.
    
    We use a simplified approach: for each center, find its nearest target point
    and use the squared distance. This is not true W2 but serves as a structural guide.
    
    For a proper W2, we'd need to solve an OT problem, but for guidance purposes,
    we can use a soft assignment.
    """
    # Use soft assignment: for each center, compute Gaussian-weighted distance to all targets
    # This is O(N * M) where M = len(target)
    
    # For efficiency, use a simpler approach: nearest neighbor with softening
    tree = cKDTree(target)
    dists, idx = tree.query(centers, k=1)
    
    # Gradient: 2 * (center - assigned_target) / (N)
    # But we want to encourage spreading, so we add a repulsion term
    
    assigned_targets = target[idx]
    grad = 2.0 * (centers - assigned_targets) / N
    
    return grad

def _soft_barrier_penalty(centers, r_current, beta=20.0):
    """
    Compute smooth barrier penalty for non-overlap and wall constraints.
    Returns (penalty_value, gradient).
    """
    from scipy.spatial.distance import pdist
    
    # Pairwise distances
    d = pdist(centers)
    # We want d >= 2*r_current
    # Penalty: sum of soft violations
    violations = np.maximum(0, 2.0 * r_current - d)
    # Soft penalty: use exponential
    penalty = np.sum(np.exp(beta * violations) - 1.0) / beta
    
    # Gradient w.r.t. centers
    # This is complex; use finite differences or analytical
    # For now, use a simpler force-based approach
    grad = np.zeros_like(centers)
    
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=5)  # Check nearby points
    
    for i in range(N):
        for k in range(1, 5):
            j = idx[i, k]
            if j < 0:
                break
            diff = centers[i] - centers[j]
            dist = dists[i, k]
            if dist < 2.0 * r_current + 0.05:
                # Repulsion force
                strength = 1.0 / (dist + 1e-8)
                grad[i] += strength * diff
                grad[j] -= strength * diff
    
    # Wall penalties
    for i in range(N):
        for k in range(2):
            wall_dist = min(0.5 - centers[i, k], centers[i, k] + 0.5)
            if wall_dist < r_current + 0.05:
                strength = 1.0 / (wall_dist + 1e-8)
                if centers[i, k] < 0:
                    grad[i, k] += strength
                else:
                    grad[i, k] -= strength
    
    return penalty, grad

def _ot_guided_optimization(centers, max_iter=5000):
    """
    Main optimization loop combining OT guidance with barrier constraints.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # Generate target hex lattice
    # Try different grid configurations
    best_target = None
    best_target_r = 0.0
    for n_rows in range(12, 20):
        n_cols = int(np.ceil(N / n_rows))
        target = _generate_target_hex_lattice(n_rows, n_cols)
        # Take first N points or resample
        if len(target) >= N:
            target = target[:N]
        else:
            # Repeat or interpolate
            while len(target) < N:
                target = np.vstack([target, target[:N - len(target)]])
            target = target[:N]
        
        # Evaluate how well this target matches the current structure
        # Use a simple metric: mean distance after optimal assignment
        tree = cKDTree(target)
        dists, _ = tree.query(centers, k=1)
        mean_dist = np.mean(dists)
        
        if mean_dist < best_target_r or best_target is None:
            best_target_r = mean_dist
            best_target = target
    
    target = best_target
    
    lr = 0.005
    no_improve = 0
    
    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        
        # Compute current radius
        r_current = _compute_radius(centers)
        
        if r_current > best_r + 1e-10:
            best_r = r_current
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # OT guidance gradient
        ot_grad = _wasserstein_2_gradient(centers, target)
        
        # Barrier penalty gradient
        _, barrier_grad = _soft_barrier_penalty(centers, r_current, beta=15.0)
        
        # Combined gradient
        # We want to follow OT gradient (toward target structure) 
        # but respect barrier constraints
        combined_grad = 0.3 * ot_grad + 1.0 * barrier_grad
        
        # Normalize
        grad_norm = np.linalg.norm(combined_grad)
        if grad_norm > 1e-10:
            combined_grad /= grad_norm
        
        # Update
        centers += lr * combined_grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Adaptive learning rate
        if no_improve > 30:
            lr *= 0.95
            no_improve = 0
            if lr < 1e-7:
                break
        
        # Occasionally re-evaluate target
        if it % 500 == 0 and it > 0:
            for n_rows in range(12, 20):
                n_cols = int(np.ceil(N / n_rows))
                new_target = _generate_target_hex_lattice(n_rows, n_cols)
                if len(new_target) >= N:
                    new_target = new_target[:N]
                else:
                    while len(new_target) < N:
                        new_target = np.vstack([new_target, new_target[:N - len(new_target)]])
                    new_target = new_target[:N]
                
                tree = cKDTree(new_target)
                dists, _ = tree.query(centers, k=1)
                mean_dist = np.mean(dists)
                
                if mean_dist < best_target_r:
                    best_target_r = mean_dist
                    target = new_target
    
    return best_centers, best_r

def _final_polish(centers, max_iter=3000):
    """
    Final high-precision polish using pure gradient ascent on radius.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    lr = 0.001
    no_improve = 0
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        r_current = _compute_radius(centers)
        
        if r_current > best_r + 1e-12:
            best_r = r_current
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Compute gradient: move points to increase minimum constraint
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=2)
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        
        # Determine which constraint is binding
        min_pair_dist = np.min(nn_dists)
        min_wall = np.min(wall_dists)
        
        grad = np.zeros_like(centers)
        
        if min_pair_dist / 2.0 <= min_wall:
            # Pairwise constraint is binding
            for i in range(N):
                j = nn_idxs[i]
                diff = centers[i] - centers[j]
                dist = nn_dists[i]
                if dist > 1e-10:
                    grad[i] += diff / (dist ** 2)
        else:
            # Wall constraint is binding
            for i in range(N):
                for k in range(2):
                    if centers[i, k] < 0:
                        wall_d = 0.5 + centers[i, k]
                        if wall_d < 0.1:
                            grad[i, k] += 1.0 / (wall_d + 1e-8)
                    else:
                        wall_d = 0.5 - centers[i, k]
                        if wall_d < 0.1:
                            grad[i, k] -= 1.0 / (wall_d + 1e-8)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        centers += lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        if no_improve > 50:
            lr *= 0.95
            no_improve = 0
            if lr < 1e-8:
                break
    
    return best_centers, best_r

def _generate_initial_hex():
    """Generate initial hexagonal grid configuration."""
    best_centers = None
    best_r = 0.0
    
    for n_rows in range(12, 20):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing
        s = 1.0
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Center
        cx = (centers[:, 0].max() + centers[:, 0].min()) / 2
        cy = (centers[:, 1].max() + centers[:, 1].min()) / 2
        centers[:, 0] -= cx
        centers[:, 1] -= cy
        
        # Scale to fit
        max_ext = np.max(np.abs(centers))
        if max_ext > 0.48:
            scale = 0.48 / max_ext
            centers *= scale
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Phase 1: Initial hexagonal grid
    centers = _generate_initial_hex()
    r = _compute_radius(centers)
    
    # Phase 2: OT-guided optimization
    if _time_left() > 30.0:
        centers, r = _ot_guided_optimization(centers, max_iter=8000)
    
    # Phase 3: Final polish
    if _time_left() > 10.0:
        centers, r = _final_polish(centers, max_iter=5000)
    
    # Ensure validity
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
