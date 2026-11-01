"""Circle packing: 50 equal circles in unit disk.
Strategy: Contact-graph force-directed layout with hard projection.
Maintains explicit contact edges, applies repulsive forces along edges,
and projects onto feasible set (inside disk, non-overlapping) each step.
"""

import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)

def _build_contact_graph(points, r_target, threshold_factor=1.5):
    """Build contact graph: edges between pairs closer than threshold_factor * 2*r_target."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    threshold = threshold_factor * 2.0 * r_target
    i_idx, j_idx = np.triu_indices(N, k=1)
    mask = dists[i_idx, j_idx] < threshold
    edges_i = i_idx[mask]
    edges_j = j_idx[mask]
    return edges_i, edges_j

def _force_directed_step(points, edges_i, edges_j, r_target, step_size, boundary_strength=1.0):
    """
    One force-directed step:
    - Repulsive force along each contact edge pushing circles apart
    - Attractive force pulling circles toward disk center (boundary constraint)
    - Hard projection: clip to disk, resolve overlaps
    """
    forces = np.zeros_like(points)
    
    # Repulsive forces along contact edges
    for i, j in zip(edges_i, edges_j):
        diff = points[i] - points[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            # Random small perturbation
            forces[i] += 0.01 * np.random.randn(2)
            forces[j] -= 0.01 * np.random.randn(2)
        else:
            unit = diff / dist
            # Strong repulsion when close to contact
            overlap = max(0.0, 2.0 * r_target - dist)
            if overlap > 0:
                force_mag = 10.0 * overlap + 0.1
            else:
                force_mag = 0.01
            forces[i] += force_mag * unit
            forces[j] -= force_mag * unit
    
    # Boundary constraint: push inward if close to boundary
    r = np.linalg.norm(points, axis=1)
    for i in range(N):
        if r[i] > 1.0 - r_target:
            if r[i] > 1e-12:
                unit_out = points[i] / r[i]
                forces[i] -= boundary_strength * (r[i] - (1.0 - r_target)) * unit_out
    
    # Apply forces with step size
    new_points = points + step_size * forces
    
    # Hard projection: ensure inside disk
    r_new = np.linalg.norm(new_points, axis=1)
    max_r = 1.0 - r_target
    for i in range(N):
        if r_new[i] > max_r:
            new_points[i] = (max_r / r_new[i]) * new_points[i]
    
    # Resolve overlaps: iterative pairwise separation
    for iteration in range(5):
        any_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                diff = new_points[i] - new_points[j]
                dist = np.linalg.norm(diff)
                min_dist = 2.0 * r_target
                if dist < min_dist and dist > 1e-12:
                    any_overlap = True
                    push = (min_dist - dist) / 2.0
                    unit = diff / dist
                    new_points[i] += push * unit
                    new_points[j] -= push * unit
                    # Re-project to disk
                    for k in [i, j]:
                        r_k = np.linalg.norm(new_points[k])
                        if r_k > max_r:
                            new_points[k] = (max_r / r_k) * new_points[k]
        if not any_overlap:
            break
    
    return new_points

def _contact_graph_optimize(points, r_target, max_iters=5000, deadline=None):
    """
    Optimize radius r_target using contact-graph force-directed layout.
    Binary search on r_target: find max r such that feasible packing exists.
    """
    best_points = points.copy()
    best_r = _compute_score(points)
    
    # Binary search on radius
    r_lo = best_r * 0.9
    r_hi = best_r * 1.15
    
    for outer_iter in range(30):
        if deadline and time.time() > deadline:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        current_points = points.copy()
        
        # Force-directed optimization at this radius
        step_size = 0.05
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            if it % 200 == 0:
                # Rebuild contact graph
                edges_i, edges_j = _build_contact_graph(current_points, r_mid, threshold_factor=1.8)
                # Decay step size
                step_size = max(0.005, 0.05 * (0.99 ** (it // 100)))
            
            current_points = _force_directed_step(
                current_points, edges_i, edges_j, r_mid, step_size,
                boundary_strength=1.0
            )
        
        # Check feasibility
        score = _compute_score(current_points)
        if score >= r_mid - 1e-9:
            # Feasible at r_mid, try larger
            r_lo = r_mid
            if score > best_r:
                best_r = score
                best_points = current_points.copy()
        else:
            # Not feasible, try smaller
            r_hi = r_mid
        
        # Convergence check
        if r_hi - r_lo < 1e-8:
            break
    
    return best_points, best_r

def _hex_init(n, seed=0):
    """Hexagonal lattice init."""
    rng = np.random.RandomState(seed)
    best_spacing = None
    best_points = None
    
    for spacing in np.linspace(0.12, 0.25, 15):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= n:
            order = np.argsort(radius[mask], kind="stable")
            selected = points[order[:n]]
            score = _compute_score(selected)
            if best_points is None or score > _compute_score(best_points):
                best_points = selected.copy()
                best_spacing = spacing
    
    if best_points is not None:
        return best_points
    
    # Fallback random
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _ring_init(n, seed=0):
    """Ring-based initialization: place circles on concentric rings."""
    rng = np.random.RandomState(seed)
    points = np.zeros((n, 2))
    
    # Place first circle at center
    points[0] = [0.0, 0.0]
    remaining = n - 1
    
    # Distribute remaining on rings
    ring_radii = [0.25, 0.5, 0.75, 0.9]
    ring_counts = []
    for i, rr in enumerate(ring_radii):
        if i < len(ring_radii) - 1:
            count = min(remaining, int(2 * np.pi * rr / 0.25))
        else:
            count = remaining
        ring_counts.append(count)
        remaining -= count
    
    idx = 1
    for rr, count in zip(ring_radii, ring_counts):
        if count <= 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.uniform(0, 2*np.pi/count)
        for a in angles:
            if idx < n:
                points[idx] = [rr * np.cos(a), rr * np.sin(a)]
                idx += 1
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Multiple initializations
    inits = []
    inits.append(_hex_init(N, seed=0))
    inits.append(_hex_init(N, seed=1))
    inits.append(_ring_init(N, seed=0))
    
    # Random inits
    rng = np.random.RandomState(42)
    for s in range(3):
        theta = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(0, 1, N))
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    for init_pts in inits:
        if time.time() > deadline - 15.0:
            break
        
        # Contact-graph force-directed optimization
        polished_pts, score = _contact_graph_optimize(
            init_pts, 
            r_target=_compute_score(init_pts),
            max_iters=3000,
            deadline=deadline
        )
        
        if score > best_score:
            best_score = score
            best_points = polished_pts.copy()
        
        # Also try a second pass with slightly perturbed start
        if time.time() < deadline - 10.0:
            perturbed = polished_pts + 0.01 * np.random.randn(N, 2)
            # Re-project to disk
            r_p = np.linalg.norm(perturbed, axis=1)
            for i in range(N):
                if r_p[i] > 0.95:
                    perturbed[i] = 0.95 * perturbed[i] / r_p[i]
            
            polished2, score2 = _contact_graph_optimize(
                perturbed,
                r_target=score,
                max_iters=2000,
                deadline=deadline
            )
            if score2 > best_score:
                best_score = score2
                best_points = polished2.copy()
    
    # Final: try to improve with a tighter binary search on the best found
    if best_points is not None and time.time() < deadline - 5.0:
        final_pts, final_score = _contact_graph_optimize(
            best_points,
            r_target=best_score,
            max_iters=2000,
            deadline=deadline
        )
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    # Fallback
    if best_points is None:
        best_points = _hex_init(N, seed=0)
    
    return best_points

# EVOLVE-BLOCK-END
