"""
Packing 50 equal circles in a unit disk using direct SLSQP constraint optimization
with adaptive active-set selection based on contact graph analysis.

Structural change from soft-min surrogate to:
1. Direct minimization of -r (negative radius) with explicit nonlinear constraints
2. Adaptive active set: only enforce the K most critical distance constraints
3. Multi-stage r-increase strategy: solve feasibility at increasing r targets
4. Contact graph identification for structural refinement
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)


def _build_contact_graph(pts, r_target, k_nearest=6):
    """
    Build a contact graph: identify pairs of circles that are close to touching
    (within a small tolerance of 2*r_target).
    Returns list of (i, j) pairs that are 'active' constraints.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    
    threshold = 2.0 * r_target * 1.05  # 5% tolerance
    iu = np.triu_indices(N, k=1)
    active_pairs = []
    for idx in range(len(iu[0])):
        if dist_mat[iu[0][idx], iu[1][idx]] <= threshold:
            active_pairs.append((iu[0][idx], iu[1][idx]))
    
    # Also check boundary contacts
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_active = []
    for i in range(N):
        if dists_origin[i] >= 1.0 - r_target * 1.05:
            boundary_active.append(i)
    
    return active_pairs, boundary_active


def _slsqp_constraints(pts, r_target, active_pairs, boundary_active, max_constraints=40):
    """
    Generate constraint functions for SLSQP.
    Constraints: for active pairs, d_ij >= 2*r_target
                for boundary active, 1 - |c_i| >= r_target
    """
    constraints = []
    
    # Pairwise constraints
    for (i, j) in active_pairs[:max_constraints]:
        def pair_constraint(x, i=i, j=j):
            return np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - 2.0 * r_target
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Boundary constraints
    for i in boundary_active[:max(0, max_constraints - len(active_pairs))]:
        def boundary_constraint(x, i=i):
            return 1.0 - np.linalg.norm(x[2*i:2*i+2]) - r_target
        constraints.append({'type': 'ineq', 'fun': boundary_constraint})
    
    return constraints


def _slsqp_objective(x, r_target):
    """
    Objective: minimize sum of constraint violations (soft penalty).
    This helps guide the optimizer even when constraints are slightly violated.
    """
    pts = x.reshape(N, 2)
    
    # Penalty for boundary violations
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, r_target - (1.0 - dists_origin))
    
    # Penalty for pairwise violations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    pair_violation = np.maximum(0, 2.0 * r_target - dist_mat)
    
    return np.sum(boundary_violation ** 2) + np.sum(pair_violation ** 2)


def _slsqp_optimize(pts_init, r_target, max_iter=300, max_constraints=30):
    """
    Run SLSQP optimization to find a valid packing with radius >= r_target.
    Uses adaptive active set based on contact graph.
    """
    x0 = pts_init.flatten()
    
    # Build initial contact graph
    active_pairs, boundary_active = _build_contact_graph(pts_init, r_target)
    
    # If no active constraints, use all pairs within a larger threshold
    if not active_pairs and not boundary_active:
        diff = pts_init[:, np.newaxis, :] - pts_init[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        iu = np.triu_indices(N, k=1)
        sorted_dists = np.sort(dist_mat[iu])
        # Take the closest max_constraints pairs
        n_close = min(max_constraints, len(iu[0]))
        active_pairs = [(iu[0][k], iu[1][k]) for k in range(n_close)]
    
    constraints = _slsqp_constraints(pts_init, r_target, active_pairs, boundary_active, max_constraints)
    
    # Define objective with penalty
    def obj(x):
        return _slsqp_objective(x, r_target)
    
    # Bounds: keep points within unit disk
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(obj, x0, method='SLSQP', constraints=constraints,
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12})
    
    return result.x.reshape(N, 2), result.success


def _feasibility_search(pts_init, r_low, r_high, n_steps=10, max_iter_per_step=200):
    """
    Binary search on r to find the maximum feasible radius.
    At each step, run SLSQP to check if radius r is achievable.
    """
    best_pts = pts_init.copy()
    best_r = _score(pts_init)
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 15:
            break
        
        r_mid = (r_low + r_high) / 2.0
        r_mid = max(r_mid, best_r + 1e-6)  # Must improve
        
        pts_candidate, success = _slsqp_optimize(pts_init, r_mid, max_iter=max_iter_per_step)
        
        # Project to unit disk
        dists = np.linalg.norm(pts_candidate, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_candidate[mask] = pts_candidate[mask] / dists[mask, np.newaxis]
        
        score_candidate = _score(pts_candidate)
        
        if score_candidate >= r_mid - 1e-8:
            # Feasible: increase r_low
            r_low = r_mid
            best_pts = pts_candidate.copy()
            best_r = score_candidate
        else:
            # Not feasible: decrease r_high
            r_high = r_mid
        
        # Update pts_init to current best for next iteration
        if best_r > _score(pts_init):
            pts_init = best_pts.copy()
    
    return best_pts, best_r


def _force_directed_relax(pts, n_steps=300, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with adaptive learning rate
        lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts


def _contact_graph_refine(pts, r_target, n_iterations=50):
    """
    Refine packing using contact graph analysis.
    At each iteration, identify the most constrained pairs and optimize only those.
    """
    pts = pts.copy()
    
    for it in range(n_iterations):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        # Build contact graph
        active_pairs, boundary_active = _build_contact_graph(pts, r_target)
        
        if not active_pairs and not boundary_active:
            break
        
        # Run SLSQP with only active constraints
        x0 = pts.flatten()
        constraints = _slsqp_constraints(pts, r_target, active_pairs, boundary_active, max_constraints=25)
        
        def obj(x):
            return _slsqp_objective(x, r_target)
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(obj, x0, method='SLSQP', constraints=constraints,
                            bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            pts_new = result.x.reshape(N, 2)
        except Exception:
            break
        
        # Project to unit disk
        dists = np.linalg.norm(pts_new, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / dists[mask, np.newaxis]
        
        score_new = _score(pts_new)
        score_old = _score(pts)
        
        if score_new >= score_old - 1e-10:
            pts = pts_new
        else:
            # Try with relaxed target
            r_relaxed = r_target * 0.95
            active_pairs_r, boundary_active_r = _build_contact_graph(pts, r_relaxed)
            constraints_r = _slsqp_constraints(pts, r_relaxed, active_pairs_r, boundary_active_r, max_constraints=25)
            
            def obj_r(x):
                return _slsqp_objective(x, r_relaxed)
            
            try:
                result = minimize(obj_r, x0, method='SLSQP', constraints=constraints_r,
                                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                pts_new = result.x.reshape(N, 2)
                dists = np.linalg.norm(pts_new, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    pts_new[mask] = pts_new[mask] / dists[mask, np.newaxis]
                score_new = _score(pts_new)
                if score_new > score_old:
                    pts = pts_new
            except Exception:
                break
    
    return pts


def _hex_lattice_init(r_guess=0.12):
    """Generate a hexagonal lattice packing as initial guess."""
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


def _known_structure_init():
    """
    Initialize with the known optimal structure: 1 + 6 + 12 + 18 + 13 rings.
    Use the best known radius as a guide for ring radii.
    """
    r_known = 0.1258254895304
    ring_sizes = [1, 6, 12, 18, 13]
    
    # Ring radii based on the known structure
    # Center: 0
    # Ring 1 (6 circles): ~0.25
    # Ring 2 (12 circles): ~0.50
    # Ring 3 (18 circles): ~0.75
    # Ring 4 (13 circles): ~0.95
    ring_radii = [0.0, 0.24, 0.48, 0.72, 0.94]
    
    points = np.zeros((N, 2))
    idx = 0
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        for j in range(size):
            angle = 2 * np.pi * j / size + (np.pi / size if ring_idx % 2 == 1 else 0)
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    
    return points


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


def _full_optimize(points, budget_seconds=25.0):
    """
    Full optimization pipeline using SLSQP with adaptive active sets.
    Phase 1: Force-directed pre-conditioning
    Phase 2: Feasibility search (binary search on r)
    Phase 3: Contact graph refinement
    """
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Phase 1: Force-directed pre-conditioning
    if time.time() - t0 < budget_seconds * 0.20:
        target_r = max(best_score, 0.12)
        pts_fd = _force_directed_relax(points, n_steps=150, target_r=target_r)
        score_fd = _score(pts_fd)
        if score_fd > best_score:
            best_score = score_fd
            best_points = pts_fd.copy()
    
    # Phase 2: Feasibility search - binary search on r
    if time.time() - t0 < budget_seconds * 0.70:
        r_low = max(best_score, 0.12)
        r_high = min(0.135, r_low + 0.015)
        
        remaining = budget_seconds - (time.time() - t0)
        n_steps = max(5, int(remaining / 3.0))
        
        pts_feas, r_feas = _feasibility_search(best_points, r_low, r_high, 
                                                n_steps=n_steps, max_iter_per_step=150)
        if r_feas > best_score:
            best_score = r_feas
            best_points = pts_feas.copy()
    
    # Phase 3: Contact graph refinement
    if time.time() - t0 < budget_seconds * 0.95:
        r_target = best_score * 1.001
        pts_refined = _contact_graph_refine(best_points, r_target, n_iterations=30)
        score_refined = _score(pts_refined)
        if score_refined > best_score:
            best_score = score_refined
            best_points = pts_refined.copy()
    
    return best_points, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Known structure initialization
    known_pts = _known_structure_init()
    known_score = _score(known_pts)
    if known_score > best_score:
        best_score = known_score
        best_points = known_pts.copy()
    
    # Strategy 2: Hex lattice initialization
    hex_pts = _hex_lattice_init(0.125)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_points = hex_pts.copy()
    
    # Strategy 3: Multiple ring configurations with full SLSQP optimization
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.24, 0.48, 0.72, 0.94]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.26, 0.52, 0.76, 0.96]),
    ]
    
    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - _START_TIME > _DEADLINE - 40:
            break
        
        points = _ring_init(ring_sizes, radii)
        if config_idx > 0:
            points += rng.uniform(-0.005, 0.005, size=(N, 2))
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 25
        budget = min(20.0, remaining)
        if budget < 8:
            break
            
        pts_opt, score_opt = _full_optimize(points, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Strategy 4: Multi-start with perturbation from best
    n_starts = 2
    for s in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE - 20:
            break
        
        if s == 0:
            pts_init = best_points.copy()
            noise = rng.normal(0, 0.008, size=pts_init.shape)
        else:
            pts_init = _known_structure_init()
            noise = rng.normal(0, 0.012, size=pts_init.shape)
        
        pts_init += noise
        norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
        r_max = 1.0 - max(best_score, 0.12)
        for i in range(N):
            if norms[i] > r_max:
                pts_init[i] *= r_max / norms[i]
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 10
        if remaining < 10:
            break
            
        pts_opt, score_opt = _full_optimize(pts_init, budget_seconds=min(15.0, remaining))
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Final contact graph refinement
    if best_points is not None and time.time() - _START_TIME < _DEADLINE - 5:
        r_current = _score(best_points)
        if r_current > 0.1:
            pts_final = _contact_graph_refine(best_points, r_current * 1.0005, n_iterations=20)
            score_final = _score(pts_final)
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final.copy()
    
    if best_points is None:
        best_points = _known_structure_init()
    
    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.999 / max(dists[i], 1e-10)
    
    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
