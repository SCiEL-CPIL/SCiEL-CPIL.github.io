"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_indices(n):
    """Return indices for all pairs i < j."""
    i_idx, j_idx = np.triu_indices(n, 1)
    return i_idx, j_idx

def _score_vec(points):
    """Compute the packing radius for given centers (vectorized)."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    i_idx, j_idx = _pairwise_indices(n)
    pair_dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=1) / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _ring_initialization(n):
    """Generate ring-based initial configurations."""
    best_points = None
    best_r = 0.0
    
    # Try various ring structures
    for num_rings in range(1, 8):
        for ring_spread in np.linspace(0.3, 0.9, 15):
            points = []
            # Center point
            if num_rings >= 1:
                points.append([0.0, 0.0])
            
            for ring in range(1, num_rings + 1):
                r_ring = ring * ring_spread / num_rings
                # Number of points on this ring
                if ring == 1:
                    n_ring = 6
                else:
                    n_ring = 6 * ring
                # Adjust to fit total
                remaining = n - len(points) - sum(6 * (k + 1) for k in range(ring, num_rings))
                if remaining <= 0:
                    n_ring = max(1, remaining)
                
                angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
                for a in angles:
                    x = r_ring * np.cos(a)
                    y = r_ring * np.sin(a)
                    if np.hypot(x, y) <= 1.0:
                        points.append([x, y])
            
            points = np.array(points)
            if len(points) >= n:
                # Take the n closest to center
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists)[:n]
                cand = points[idx]
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_points = cand.copy()
    
    if best_points is None:
        # Fallback: hexagonal
        best_points = _hex_init(n)
    
    return best_points, best_r

def _hex_init(n):
    """Hexagonal lattice initialization."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def _energy_and_grad(params, n):
    """Compute energy and gradient for force-directed optimization.
    Energy penalizes: (1) pairs too close, (2) circles too far from boundary.
    We use a soft repulsive potential.
    """
    pts = params.reshape(-1, 2)
    i_idx, j_idx = _pairwise_indices(n)
    
    # Pairwise distances
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    # Avoid division by zero
    dists_safe = np.maximum(dists, 1e-10)
    
    # Repulsive energy: for pairs closer than 2*r_target, push apart
    # We use a quadratic penalty for overlap with a smooth transition
    r_target = 0.125  # approximate target radius
    threshold = 2 * r_target
    # Soft penalty: (max(0, threshold - dist))^2
    overlap = np.maximum(0, threshold - dists)
    energy = np.sum(overlap ** 2)
    
    # Boundary penalty: circles should be at distance >= r_target from boundary
    # i.e., |c_i| <= 1 - r_target
    norms = np.linalg.norm(pts, axis=1)
    boundary_overlap = np.maximum(0, norms - (1.0 - r_target))
    energy += np.sum(boundary_overlap ** 2)
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = dists[k]
        if d < threshold and d > 1e-10:
            # d(overlap^2)/d(dist) = 2*overlap * (-1) = -2*overlap
            # d(dist)/d(pts[i]) = -diffs[k]/d  (since dist = |pts[i]-pts[j]|)
            # So d(energy)/d(pts[i]) = -2*overlap * (-diffs[k]/d) = 2*overlap*diffs[k]/d
            factor = 2 * overlap[k] / d
            grad[i] += factor * diffs[k]
            grad[j] -= factor * diffs[k]
    
    # Boundary gradient
    for i in range(n):
        if norms[i] > (1.0 - r_target) and norms[i] > 1e-10:
            grad[i] += 2 * boundary_overlap[i] * pts[i] / norms[i]
    
    return energy, grad.flatten()

def _optimize_force_directed(points, n, deadline, max_iters=500):
    """Optimize using force-directed layout with adaptive cooling."""
    params = points.flatten()
    best_params = params.copy()
    best_score = _score_vec(points)
    
    # Adaptive cooling schedule
    temp = 0.05
    cool_rate = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(params, n)
        
        # Gradient descent step with adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        lr = temp / max(grad_norm, 1e-6)
        lr = min(lr, 0.01)
        
        new_params = params - lr * grad
        
        # Check if new position is valid
        new_points = new_params.reshape(-1, 2)
        if not np.all(np.isfinite(new_points)):
            temp *= cool_rate
            continue
        
        new_score = _score_vec(new_points)
        
        # Accept if better or with probability (Metropolis)
        if new_score >= best_score:
            params = new_params
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            # Metropolis acceptance
            delta = best_score - new_score
            if delta < 1e-10 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                params = new_params
        
        temp *= cool_rate
    
    return best_params.reshape(-1, 2), best_score

def _polish_lbfgs(points, n, deadline, maxiter=200):
    """Final polish using L-BFGS-B on smooth min objective."""
    i_idx, j_idx = _pairwise_indices(n)
    
    def objective(params):
        pts = params.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        diffs = pts[i_idx] - pts[j_idx]
        dists = np.linalg.norm(diffs, axis=1) / 2.0
        all_dists = np.concatenate([boundary, dists])
        # Smooth min via log-sum-exp
        b = 80.0
        neg_dists = -all_dists
        max_neg = np.max(neg_dists)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
        smooth_min = -log_sum / b
        return -smooth_min
    
    x0 = points.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if np.all(np.isfinite(res.x)):
            cand = res.x.reshape(-1, 2)
            return cand
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n = N
    best_points = None
    best_score = 0.0
    
    # Phase 1: Generate multiple initial configurations
    inits = []
    
    # Ring initialization
    ring_pts, ring_r = _ring_initialization(n)
    inits.append((ring_pts, ring_r))
    
    # Hex initialization
    hex_pts = _hex_init(n)
    hex_r = _score_vec(hex_pts)
    inits.append((hex_pts, hex_r))
    
    # Random initializations
    for _ in range(5):
        pts = np.random.randn(n, 2) * 0.4
        r = _score_vec(pts)
        inits.append((pts, r))
    
    # Sort by initial score
    inits.sort(key=lambda x: x[1], reverse=True)
    
    # Phase 2: For each init, run force-directed then polish
    for idx, (init_pts, init_r) in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Force-directed optimization
        fd_pts, fd_score = _optimize_force_directed(init_pts, n, deadline - 15.0, max_iters=300)
        
        if fd_score > best_score:
            best_score = fd_score
            best_points = fd_pts.copy()
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 5.0:
            polished = _polish_lbfgs(fd_pts, n, deadline - 5.0, maxiter=100)
            pol_score = _score_vec(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
    
    # Phase 3: Multi-start refinement around best
    if time.time() < deadline - 5.0 and best_points is not None:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            pert = best_points + np.random.randn(n, 2) * 0.01
            fd_pts, fd_score = _optimize_force_directed(pert, n, deadline - 5.0, max_iters=100)
            if fd_score > best_score:
                best_score = fd_score
                best_points = fd_pts.copy()
            if time.time() < deadline - 2.0:
                polished = _polish_lbfgs(fd_pts, n, deadline - 2.0, maxiter=50)
                pol_score = _score_vec(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished.copy()
    
    if best_points is None:
        best_points = inits[0][0]
    
    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, 2))
    
    return best_points
# EVOLVE-BLOCK-END
