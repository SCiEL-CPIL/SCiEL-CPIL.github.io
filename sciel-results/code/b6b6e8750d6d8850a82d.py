"""Rewritten program for circle packing in perimeter-4 rectangle using force-directed relaxation."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _build_contact_graph(centers, radii, tolerance=0.01):
    """
    Build a contact graph where edges connect circles that are nearly touching.
    Returns a list of (i, j) pairs representing contact edges.
    """
    n = len(centers)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if abs(dist - sum_r) < tolerance:
                contacts.append((i, j))
    return contacts


def _compute_bar_forces(centers, radii, contacts, stiffness=1.0):
    """
    Compute forces from the bar-joint framework.
    Each contact edge acts as a rigid bar with length = r_i + r_j.
    Forces push/pull to maintain contact distances.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i, j in contacts:
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        target_dist = radii[i] + radii[j]
        
        if dist < 1e-10:
            continue
        
        # Spring force: F = k * (dist - target)
        # If dist < target, circles overlap, push apart
        # If dist > target, circles too far, pull together
        force_mag = stiffness * (dist - target_dist)
        fx = (dx / dist) * force_mag
        fy = (dy / dist) * force_mag
        
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[j, 0] -= fx
        forces[j, 1] -= fy
    
    return forces


def _compute_wall_forces(centers, radii, w, h, stiffness=0.5):
    """
    Compute forces from wall constraints.
    Circles should be pushed toward walls to maximize packing density.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Left wall: if x - r > 0, push left
        if x - r > 0:
            forces[i, 0] -= stiffness * (x - r)
        # Right wall: if x + r < w, push right
        if x + r < w:
            forces[i, 0] += stiffness * (w - x - r)
        # Bottom wall: if y - r > 0, push down
        if y - r > 0:
            forces[i, 1] -= stiffness * (y - r)
        # Top wall: if y + r < h, push up
        if y + r < h:
            forces[i, 1] += stiffness * (h - y - r)
    
    return forces


def _topological_relaxation(centers, radii, w, h, iterations=10000, 
                            bar_stiffness=10.0, wall_stiffness=0.5,
                            lr_init=0.01, lr_decay=0.999):
    """
    Topological force-directed relaxation using contact graph.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    lr = lr_init
    
    for it in range(iterations):
        # Build contact graph
        contacts = _build_contact_graph(centers, radii, tolerance=0.02)
        
        # Compute forces
        bar_forces = _compute_bar_forces(centers, radii, contacts, stiffness=bar_stiffness)
        wall_forces = _compute_wall_forces(centers, radii, w, h, stiffness=wall_stiffness)
        
        total_forces = bar_forces + wall_forces
        
        # Apply forces
        centers += lr * total_forces
        
        # Clamp to valid region
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Decay learning rate
        lr *= lr_decay
        
        # Periodically rebuild contacts with tighter tolerance
        if it % 100 == 0:
            # Rebuild with current positions
            pass
    
    return centers, radii


def _evaluate(centers, w, h, radii):
    """Evaluate validity and objective. Returns (is_valid, objective)."""
    n = len(centers)
    
    # Check bounding box perimeter
    if w + h > 2.0 + 1e-6:
        return False, 0.0
    
    # Check all circles are within the rectangle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
            return False, 0.0
    
    # Check no overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-6:
                return False, 0.0
    
    # Objective: sum of radii
    return True, np.sum(radii)


def _grow_radii_and_relax(centers, radii, w, h, growth_rate=1e-4, iterations=5000):
    """
    Iteratively grow radii and relax the configuration.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    for step in range(iterations):
        # Grow radii slightly
        radii *= (1.0 + growth_rate)
        
        # Relax to satisfy constraints
        centers, radii = _topological_relaxation(
            centers, radii, w, h, 
            iterations=100,
            bar_stiffness=5.0,
            wall_stiffness=0.3,
            lr_init=0.005,
            lr_decay=0.99
        )
        
        # Check if still valid
        is_valid, _ = _evaluate(centers, w, h, radii)
        if not is_valid:
            # Rollback radii growth
            radii /= (1.0 + growth_rate)
            break
    
    return centers, radii


def _initial_grid_layout(n, w, h):
    """Generate initial grid layout."""
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4
    and maximize the sum of their radii.
    
    Returns: array of shape (n, 3) with [x, y, r] for each circle.
    """
    n = N
    best_solution = None
    best_score = -np.inf
    
    # Try different aspect ratios and initial configurations
    aspect_ratios = [
        (1.0, 1.0),
        (0.85, 1.15),
        (1.15, 0.85),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
        (1.2, 0.8),
    ]
    
    for w, h in aspect_ratios:
        random.seed(int(w * 1000) + int(h * 1000))
        np.random.seed(int(w * 1000) + int(h * 1000))
        
        # Initial grid layout
        centers = _initial_grid_layout(n, w, h)
        
        # Initial radii estimate
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        base_r = min(w / cols, h / rows) * 0.48
        radii = np.full(n, base_r)
        
        # Phase 1: Initial relaxation to get valid configuration
        centers, radii = _topological_relaxation(
            centers, radii, w, h,
            iterations=2000,
            bar_stiffness=10.0,
            wall_stiffness=0.5,
            lr_init=0.01,
            lr_decay=0.999
        )
        
        # Check validity
        is_valid, score = _evaluate(centers, w, h, radii)
        
        if is_valid:
            # Phase 2: Grow radii and relax
            centers, radii = _grow_radii_and_relax(
                centers, radii, w, h,
                growth_rate=5e-5,
                iterations=2000
            )
            
            is_valid, score = _evaluate(centers, w, h, radii)
            if is_valid and score > best_score:
                best_score = score
                best_solution = np.column_stack([centers, radii])
        
        # Phase 3: Fine-tuning with scipy
        params = np.concatenate([centers.flatten(), radii])
        
        def obj(p):
            pos = p[:2 * n].reshape(n, 2)
            r = p[2 * n:]
            
            # Penalty for invalid configurations
            penalty = 0.0
            
            # Wall violations
            for i in range(n):
                x, y = pos[i]
                r_i = r[i]
                if x - r_i < 0:
                    penalty += (x - r_i) ** 2
                if x + r_i > w:
                    penalty += (x + r_i - w) ** 2
                if y - r_i < 0:
                    penalty += (y - r_i) ** 2
                if y + r_i > h:
                    penalty += (y + r_i - h) ** 2
            
            # Overlap violations
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return -np.sum(r) + 50.0 * penalty
        
        try:
            result = minimize(obj, params, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10})
            
            opt_params = result.x
            opt_centers = opt_params[:2 * n].reshape(n, 2)
            opt_radii = opt_params[2 * n:]
            
            is_valid, score = _evaluate(opt_centers, w, h, opt_radii)
            if is_valid and score > best_score:
                best_score = score
                best_solution = np.column_stack([opt_centers, opt_radii])
        except Exception:
            pass
    
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
