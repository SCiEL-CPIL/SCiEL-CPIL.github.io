"""Pack 300 equal circles in unit circle, maximizing common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300
_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _hex_initial(seed=0):
    """Generate initial hexagonal packing."""
    rng = np.random.RandomState(seed)
    # Try different spacings
    best_points = None
    best_r = 0
    for spacing in [0.105, 0.106, 0.107, 0.108, 0.109, 0.110]:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by distance from center, take innermost N
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            # Compute actual min radius
            min_dist = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
            # Check pairwise
            from scipy.spatial.distance import cdist
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            r_val = min(min_dist, min_pair)
            if r_val > best_r:
                best_r = r_val
                best_points = pts.copy()
    if best_points is None:
        # Fallback: simple grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            best_points = points[idx]
        else:
            # Very rough fallback
            best_points = np.random.randn(N, 2) * 0.8
    return best_points

def _compute_score(points):
    """Compute the packing radius."""
    n = len(points)
    # Boundary constraint
    boundary = 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(boundary, min_pair)

def _soft_min_energy(points, beta=200.0):
    """
    Differentiable objective: minimize -soft_min of all constraints.
    Constraints: for each pair i<j, dist(i,j) - 2r >= 0, and 1 - |c_i| - r >= 0
    We maximize r, so minimize -r subject to constraints.
    
    Alternative: minimize a soft penalty for violations.
    """
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    
    # Soft min of all constraints (pairwise/2 and boundary)
    # We want to maximize the minimum, so minimize the negative soft-min
    # Use log-sum-exp for soft min
    all_constraints = np.concatenate([dist / 2.0, boundary])
    
    # Soft min via log-sum-exp: soft_min(x) = -log(sum(exp(-beta*x)))/beta
    # For numerical stability
    m = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    return -soft_min  # Minimize negative soft min = maximize soft min

def _gradient_soft_min(points, beta=200.0):
    """Gradient of the soft-min objective."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    
    all_constraints = np.concatenate([dist / 2.0, boundary])
    m = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min w.r.t. points
    # d(soft_min)/d(x_i) = -sum_j w_j * d(x_j)/d(p_i)
    grad = np.zeros_like(points)
    
    # Pairwise part: constraint k = dist(i,j)/2
    # d(dist(i,j)/2)/dp_i = (p_i - p_j) / (2 * dist(i,j))
    # But we have -soft_min, so gradient is -sum w * d(constraint)/dp
    # Objective is -soft_min, so d(-soft_min)/dp = -d(soft_min)/dp
    # d(soft_min)/dp_i = -sum_j w_j * d(constraint_j)/dp_i
    # So d(-soft_min)/dp_i = sum_j w_j * d(constraint_j)/dp_i
    
    # For pairwise constraint dist(i,j)/2:
    # d(dist(i,j)/2)/dp_i = (p_i - p_j) / (2 * dist(i,j))
    # d(dist(i,j)/2)/dp_j = (p_j - p_i) / (2 * dist(i,j))
    
    # Build weight matrix for pairwise
    pair_weights = weights[:n * (n - 1) // 2]  # This is tricky with the flattened array
    
    # Simpler: use numerical gradient or restructure
    # Let me use a different approach: compute gradient directly
    
    # Actually, let me use a simpler energy function that's easier to differentiate
    return None  # Use numerical gradient

def _optimize_from_init(points, max_iter=5000, tol=1e-12, beta_start=50.0, beta_end=500.0):
    """
    Optimize packing using gradient descent on a soft-min objective.
    """
    points = points.copy()
    n = len(points)
    
    # Precompute neighbor structure for efficiency
    # Use a simple approach: full pairwise but vectorized
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Simulated annealing / gradient descent with adaptive step
    step_size = 0.01
    beta = beta_start
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Compute current score
        current_score = _compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Check convergence
        if best_score > 0.0531:
            # Good enough, try to polish
            pass
        
        # Compute forces (gradient-like)
        # For each pair, if too close, push apart
        # For boundary, if too close to edge, push inward
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)
        np.fill_diagonal(dist, 1e10)
        
        # Target minimum distance
        target = 2.0 * best_score + 0.001  # slightly larger than current best
        
        # Forces from pairs
        forces = np.zeros_like(points)
        # Only consider pairs that are too close
        mask = dist < target
        # Force magnitude: proportional to (target - dist)
        force_mag = np.where(mask, (target - dist), 0.0)
        force_mag /= (dist + 1e-10)  # normalize by distance
        force_mag = np.where(dist < 1e-10, 0.0, force_mag)
        
        # Sum forces
        for i in range(n):
            forces[i] = np.sum(force_mag[i, :] * diff[i, :], axis=0)
        
        # Boundary forces
        r = np.hypot(points[:, 0], points[:, 1])
        boundary_force = np.zeros_like(points)
        close_to_edge = r > 1.0 - best_score - 0.001
        if np.any(close_to_edge):
            # Push inward
            inward = -points[close_to_edge] / (r[close_to_edge, np.newaxis] + 1e-10)
            boundary_force[close_to_edge] = inward * 2.0
        
        total_forces = forces + boundary_force
        
        # Adaptive step size
        max_force = np.max(np.linalg.norm(total_forces, axis=1))
        if max_force > 1e-10:
            step = step_size / max_force
            step = min(step, 0.05)  # cap step
        else:
            step = 0.0
        
        new_points = points + step * total_forces
        
        # Check if new points are valid (inside circle)
        new_r = np.hypot(new_points[:, 0], new_points[:, 1])
        if np.max(new_r) > 1.0:
            # Scale back
            scale = 1.0 / np.max(new_r) * 0.999
            new_points *= scale
        
        new_score = _compute_score(new_points)
        
        if new_score >= current_score - 1e-10:
            points = new_points
            # Decay step size
            step_size *= 0.995
        else:
            # Reject, reduce step
            step_size *= 0.9
            # Increase beta (sharper)
            beta *= 1.05
        
        # Periodically recompute best
        if it % 100 == 0:
            current_best = _compute_score(points)
            if current_best > best_score:
                best_score = current_best
                best_points = points.copy()
    
    return best_points, best_score

def _polish(points, iters=2000):
    """Final high-precision polish using local moves."""
    points = points.copy()
    n = len(points)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    for it in range(iters):
        if _remaining() < 3.0:
            break
        
        # Try small random perturbations on worst-constrained points
        # Find the bottleneck: which constraint is tightest
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
        
        min_pair_dist = np.min(dist) / 2.0
        min_boundary = np.min(boundary)
        
        if min_pair_dist < min_boundary:
            # Pairwise constraint is bottleneck
            # Find the closest pair
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_idx
            # Push them apart
            direction = points[i] - points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                new_i = points[i] + direction * 0.0005
                new_j = points[j] - direction * 0.0005
                # Check validity
                if np.hypot(new_i[0], new_i[1]) < 1.0 and np.hypot(new_j[0], new_j[1]) < 1.0:
                    points[i] = new_i
                    points[j] = new_j
        else:
            # Boundary constraint is bottleneck
            min_idx = np.argmin(boundary)
            # Push inward
            direction = -points[min_idx]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                new_pt = points[min_idx] + direction * 0.0005
                if np.hypot(new_pt[0], new_pt[1]) < 1.0:
                    points[min_idx] = new_pt
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        else:
            # Revert
            # Actually, let's keep the move if it doesn't hurt
            pass
    
    return best_points, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    for seed in range(8):
        if _remaining() < 20.0:
            break
        
        init = _hex_initial(seed)
        if init is None or len(init) != N:
            continue
        
        # Optimize
        points, score = _optimize_from_init(init, max_iter=3000)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        if _remaining() < 15.0:
            break
    
    # Final polish
    if best_points is not None and _remaining() > 5.0:
        best_points, best_score = _polish(best_points, iters=1000)
    
    if best_points is None:
        # Fallback
        best_points = _hex_initial(0)
    
    # Ensure validity
    # Check all points are inside
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = r < 1.0
    if not np.all(mask):
        # Scale points to fit
        scale = 0.999 / np.max(r)
        best_points *= scale
    
    # Verify shape
    if best_points.shape != (N, 2):
        # Pad or trim
        if len(best_points) < N:
            # Add random points near center
            extra = np.random.randn(N - len(best_points), 2) * 0.1
            best_points = np.vstack([best_points, extra])
        else:
            best_points = best_points[:N]
    
    return best_points
