"""
Packing 50 equal circles in a unit disk using a Sum-Product-inspired contact graph
approach combined with gradient-based optimization.

Key structural change: Instead of purely geometric force-directed methods, we:
1. Build a contact graph from the current configuration
2. Use message-passing (Sum-Product Algorithm) to identify critical contacts
3. Weight the gradient based on probabilistic contact estimates
4. Combine with L-BFGS-B for precise optimization
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
START_TIME = None
DEADLINE = 130.0  # safety margin before 150s kill


def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _hex_lattice_init(r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _build_contact_graph(points, r_target, n_iter=10):
    """
    Build a contact graph and run Sum-Product Algorithm to estimate
    the probability that each pair is in the maximum independent set
    of contacts. This identifies which contacts are most critical.
    
    Returns:
        contact_probs: (N, N) matrix where contact_probs[i,j] is the
                       probability that pair (i,j) is a critical contact
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Contact threshold: pairs closer than 2*r_target are "in contact"
    contact_threshold = 2.0 * r_target
    is_contact = dists < contact_threshold
    np.fill_diagonal(is_contact, False)
    
    # Build adjacency matrix for contact graph
    adj = is_contact.astype(float)
    
    # Run Sum-Product Algorithm to estimate marginal probabilities
    # of each edge being in the maximum independent set
    # Simplified: use iterative message passing to estimate importance
    
    # Initialize messages: m_ij is message from i to j
    # We want to find which contacts are most "binding"
    
    # Use a simplified approach: weight contacts by how close they are
    # to the threshold (closer = more critical)
    proximity = np.clip((contact_threshold - dists) / contact_threshold, 0, 1)
    np.fill_diagonal(proximity, 0)
    
    # Run a few iterations of message passing to refine weights
    msg = proximity.copy()
    for _ in range(n_iter):
        # Message update: each node sends message to neighbors
        # based on the product of messages from other neighbors
        new_msg = np.zeros_like(msg)
        for i in range(n):
            neighbors = np.where(adj[i] > 0.5)[0]
            if len(neighbors) == 0:
                continue
            for j in neighbors:
                # Compute "belief" that edge (i,j) is critical
                # based on messages from other neighbors of j
                other_neighbors = np.where(adj[j] > 0.5)[0]
                other_neighbors = other_neighbors[other_neighbors != i]
                
                if len(other_neighbors) == 0:
                    belief = proximity[i, j]
                else:
                    # Product of messages from other neighbors (simplified)
                    other_msgs = msg[i, other_neighbors]
                    # Use soft product to avoid underflow
                    log_prod = np.sum(np.log(other_msgs + 1e-10))
                    belief = proximity[i, j] * np.exp(log_prod / len(other_neighbors))
                
                new_msg[i, j] = belief
                new_msg[j, i] = belief
    
    # Normalize to get probabilities
    contact_probs = new_msg.copy()
    # Scale so that max is 1
    max_val = np.max(contact_probs)
    if max_val > 0:
        contact_probs /= max_val
    
    return contact_probs


def _weighted_gradient(pts, r_target, contact_probs, soft_beta):
    """
    Compute gradient of a weighted constraint violation objective.
    The weight for each constraint is based on the Sum-Product
    contact probability estimates.
    
    Returns (objective_value, gradient)
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r_target - dists
    iu = np.triu_indices(n, k=1)
    pair_viol_flat = pair_viol[iu]
    
    # Apply contact probability weights
    pair_weights = contact_probs[iu]
    # Ensure weights are at least 0.1 to avoid ignoring any constraint
    pair_weights = np.maximum(pair_weights, 0.1)
    
    all_viol = np.concatenate([cont_viol, pair_viol_flat])
    all_weights = np.concatenate([np.ones(n), pair_weights])
    
    # Weighted soft-max
    max_v = np.max(all_viol)
    if max_v > 50.0 / soft_beta:
        return max_v, np.zeros_like(pts)
    
    exp_v = np.exp(soft_beta * (all_viol - max_v)) * all_weights
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / soft_beta
    weights = exp_v / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Container gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs
    
    # Pairwise gradient
    iw = weights[n:]
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]
        
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)
    
    return soft_max_val, grad


def _spa_optimize(pts_init, r_target, start_beta=50.0, end_beta=10000.0,
                  n_stages=5, max_iter_per_stage=800, tol=1e-10):
    """
    Optimize using Sum-Product-inspired weighted gradient with L-BFGS-B.
    """
    current_pts = pts_init.copy()
    current_val = np.inf
    
    for stage in range(n_stages):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        
        # Build contact graph and get probability estimates
        contact_probs = _build_contact_graph(current_pts, r_target, n_iter=8)
        
        def objective(x):
            pts = x.reshape(N, 2)
            val, grad = _weighted_gradient(pts, r_target, contact_probs, beta)
            return val, grad.flatten()
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': tol,
                                   'gtol': 1e-8, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        current_val = result.fun
        
        # Update contact probs for next stage
        if stage < n_stages - 1:
            contact_probs = _build_contact_graph(current_pts, r_target, n_iter=8)
    
    return current_pts, current_val


def _force_directed(points, max_iters=1500, dt=0.005):
    """Force-directed optimization with strong repulsion for close pairs."""
    points = points.copy()
    n = points.shape[0]
    best_score = _score(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        forces = np.zeros_like(points)
        current_r = _score(points)
        threshold = 2.5 * current_r if current_r > 0.01 else 0.4
        
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < threshold:
                    f = diff[i, j] / (d[i, j] ** 3 + 1e-8)
                    forces[i] += f
                    forces[j] -= f
        
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.95:
                forces[i] -= 0.5 * points[i] / dists[i]
        
        points += dt * forces
        
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.99:
                points[i] *= 0.99 / dists[i]
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        dt *= 0.999
    
    return best_points, best_score


def _lbfgs_refine(points, max_iter=2000):
    """L-BFGS-B refinement of the packing."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
    return result.x.reshape(N, 2), -result.fun


def _full_optimize(points, budget_seconds=25.0):
    """Full optimization pipeline: force-directed + SPA + L-BFGS-B."""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Phase 1: Force-directed
    if time.time() - t0 < budget_seconds * 0.3:
        pts_f, score_f = _force_directed(points, max_iters=1000, dt=0.004)
        if score_f > best_score:
            best_score = score_f
            best_points = pts_f.copy()
    
    # Phase 2: SPA-based optimization
    if time.time() - t0 < budget_seconds * 0.7:
        r_target = best_score
        pts_spa, _ = _spa_optimize(best_points, r_target,
                                   start_beta=50.0, end_beta=10000.0,
                                   n_stages=4, max_iter_per_stage=600)
        score_spa = _score(pts_spa)
        if score_spa > best_score:
            best_score = score_spa
            best_points = pts_spa.copy()
    
    # Phase 3: L-BFGS-B refinement
    if time.time() - t0 < budget_seconds * 0.9:
        pts_lb, score_lb = _lbfgs_refine(best_points, max_iter=1000)
        if score_lb > best_score:
            best_score = score_lb
            best_points = pts_lb.copy()
    
    # Phase 4: Final fine-tune
    if time.time() - t0 < budget_seconds * 0.98:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=400)
        if score_final > best_score:
            best_score = score_final
            best_points = pts_final.copy()
    
    return best_points, best_score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Known optimal structure: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        ([1, 7, 12, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.20, 0.42, 0.65, 0.88]),
    ]
    
    # Also try hex lattice init
    hex_pts = _hex_lattice_init(0.12)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_points = hex_pts.copy()
    
    # Try ring configs with full optimization
    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - START_TIME > DEADLINE - 30:
            break
        
        points = _ring_init(ring_sizes, radii)
        if config_idx > 0:
            points += rng.uniform(-0.01, 0.01, size=(N, 2))
        
        remaining = DEADLINE - (time.time() - START_TIME) - 25
        budget = min(18.0, remaining)
        if budget < 5:
            break
            
        pts_opt, score_opt = _full_optimize(points, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Binary search on radius with SPA optimization
    r_low = 0.11
    r_high = 0.145
    
    for _ in range(4):
        if time.time() - START_TIME > DEADLINE - 40:
            break
        r_mid = (r_low + r_high) / 2.0
        pts_init = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])
        
        found_feasible = False
        for beta in [100.0, 500.0, 2000.0]:
            if time.time() - START_TIME > DEADLINE - 20:
                break
            pts_opt, val = _spa_optimize(pts_init, r_mid,
                                         start_beta=beta, end_beta=beta * 50,
                                         n_stages=3, max_iter_per_stage=500)
            score = _score(pts_opt)
            if score >= r_mid - 1e-8:
                found_feasible = True
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
                break
        
        if found_feasible:
            r_low = r_mid
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-5:
            break
    
    # Multi-start with symmetry breaking and SPA optimization
    n_starts = 3
    for s in range(n_starts):
        if time.time() - START_TIME > DEADLINE - 30:
            break
        r_target = max(r_low, 0.12)
        if s == 0:
            pts_init = best_points.copy()
        elif s == 1:
            pts_init = _hex_lattice_init(r_target)
        else:
            pts_init = best_points.copy()
            noise = rng.normal(0, 0.03 * (s + 1), size=pts_init.shape)
            pts_init += noise
            norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
            for i in range(N):
                if norms[i] > 1.0 - r_target:
                    pts_init[i] *= (1.0 - r_target) / norms[i]
        
        remaining = DEADLINE - (time.time() - START_TIME) - 15
        if remaining < 10:
            break
            
        pts_opt, val = _spa_optimize(pts_init, r_target,
                                     start_beta=20.0, end_beta=20000.0,
                                     n_stages=4, max_iter_per_stage=700)
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - START_TIME < DEADLINE - 5:
        r_current = _score(best_points)
        if r_current > 0.05:
            pts_polish, _ = _spa_optimize(best_points, r_current,
                                          start_beta=5000.0, end_beta=100000.0,
                                          n_stages=3, max_iter_per_stage=400)
            score_polish = _score(pts_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()
    
    if best_points is None:
        best_points = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])
    
    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.99 / max(dists[i], 1e-10)
    
    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
