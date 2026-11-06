"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    c = centers
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Boundary clearance
    min_boundary = 0.5 - np.max(np.abs(c), axis=1).max()
    
    r = min(min_pairwise / 2, min_boundary)
    return r

def repulsion_optimize(centers, deadline, iters=5000):
    """
    Optimize using repulsion forces: each circle pushes away from others
    with force proportional to 1/d^2, plus boundary repulsion.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = compute_radius(c)
    
    # Adaptive step size
    step = 0.005
    
    for it in range(iters):
        if time.time() > deadline - 5:
            break
        
        # Compute repulsion forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Force from each pair: F_ij = (c_i - c_j) / dist^3 (unit direction / dist^2)
        # Sum over all j != i
        inv_d3 = np.zeros_like(dists)
        mask = dists < 1e-9
        np.fill_diagonal(inv_d3, 0)
        inv_d3[~mask] = 1.0 / (dists[~mask] ** 3)
        
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Boundary repulsion: push circles back if too close to edge
        boundary_force = np.zeros_like(c)
        margin = 0.05
        for k in range(2):
            # Right boundary
            close_right = c[:, k] > 0.5 - margin
            if np.any(close_right):
                boundary_force[close_right, k] += (c[close_right, k] - (0.5 - margin)) / margin
            # Left boundary
            close_left = c[:, k] < -0.5 + margin
            if np.any(close_left):
                boundary_force[close_left, k] -= ((-0.5 + margin) - c[close_left, k]) / margin
        
        # Total force
        total_force = forces + boundary_force * 2.0
        
        # Normalize forces to prevent large jumps
        force_norms = np.linalg.norm(total_force, axis=1, keepdims=True)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            total_force = total_force / max_force
        
        # Update positions
        new_c = c + step * total_force
        
        # Project back to square
        new_c = np.clip(new_c, -0.5 + 1e-8, 0.5 - 1e-8)
        
        # Check if improvement
        new_r = compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c.copy()
            c = new_c.copy()
            # Increase step slightly on improvement
            step = min(step * 1.02, 0.02)
        else:
            # Decrease step on non-improvement
            step *= 0.95
            if step < 1e-6:
                break
    
    return best_c, best_r

def anneal_optimize(centers, deadline, iters=3000):
    """
    Simulated annealing to escape local optima.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = compute_radius(c)
    
    temp = 0.01
    cooling = 0.995
    
    for it in range(iters):
        if time.time() > deadline - 5:
            break
        
        # Perturb one or two random circles
        n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
        idx = np.random.choice(N, n_perturb, replace=False)
        delta = np.random.normal(0, 0.02, (n_perturb, 2))
        new_c = c.copy()
        new_c[idx] += delta
        new_c = np.clip(new_c, -0.5 + 1e-8, 0.5 - 1e-8)
        
        new_r = compute_radius(new_c)
        delta_r = new_r - compute_radius(c)
        
        # Accept if better, or with probability exp(delta_r / temp)
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(temp, 1e-10)):
            c = new_c.copy()
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        
        temp *= cooling
    
    return best_c, best_r

def construct():
    """
    Find 50 circle centers in [-0.5, 0.5]^2 that maximize the minimum distance
    between centers, while also respecting boundary constraints.
    """
    deadline = time.time() + 145  # Leave 5s safety margin for the 150s limit
    np.random.seed(42)
    
    # Multiple initial configurations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    c_grid += np.random.normal(0, 0.01, size=c_grid.shape)
    c_grid = np.clip(c_grid, -0.499, 0.499)
    inits.append(c_grid)
    
    # 2. Hexagonal-like
    n_rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    pts = []
    for row in range(n_rows):
        y = -0.5 + (row + 0.5) / n_rows
        offset = (row % 2) * (1.0 / (2 * np.sqrt(3) * n_rows))
        n_cols = N // n_rows + (1 if row < N % n_rows else 0)
        for col in range(min(n_cols, N - len(pts))):
            x = -0.5 + (col + 0.5) / max(1, n_cols) + offset
            pts.append([x, y])
    while len(pts) < N:
        pts.append(np.random.uniform(-0.49, 0.49, 2))
    c_hex = np.array(pts[:N])
    c_hex = np.clip(c_hex, -0.499, 0.499)
    inits.append(c_hex)
    
    # 3. Random
    c_rand = np.random.uniform(-0.49, 0.49, (N, 2))
    inits.append(c_rand)
    
    # 4. Concentric rings
    c_rings = np.zeros((N, 2))
    rings = [0, 0.15, 0.3, 0.45]
    ring_sizes = [1, 6, 12, 18]
    idx = 0
    for r, size in zip(rings, ring_sizes):
        for i in range(min(size, N - idx)):
            angle = 2 * np.pi * i / size
            c_rings[idx] = [r * np.cos(angle), r * np.sin(angle)]
            idx += 1
    while idx < N:
        c_rings[idx] = np.random.uniform(-0.49, 0.49, 2)
        idx += 1
    inits.append(c_rings)
    
    best_centers = None
    best_val = 0
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Repulsion optimization
        c_opt, r_opt = repulsion_optimize(init, deadline, iters=3000)
        if r_opt > best_val:
            best_val = r_opt
            best_centers = c_opt.copy()
        
        # Phase 2: Annealing from the best
        if time.time() < deadline - 15:
            c_anneal, r_anneal = anneal_optimize(c_opt, deadline, iters=2000)
            if r_anneal > best_val:
                best_val = r_anneal
                best_centers = c_anneal.copy()
        
        # Phase 3: Fine-tune with small steps
        if time.time() < deadline - 10:
            c_fine, r_fine = repulsion_optimize(c_anneal, deadline, iters=2000)
            if r_fine > best_val:
                best_val = r_fine
                best_centers = c_fine.copy()
    
    # Final local refinement
    if time.time() < deadline - 5 and best_centers is not None:
        c_final, r_final = repulsion_optimize(best_centers, deadline, iters=1000)
        if r_final > best_val:
            best_val = r_final
            best_centers = c_final.copy()
    
    if best_centers is None:
        # Fallback: return grid
        best_centers = inits[0]
    
    return best_centers
# EVOLVE-BLOCK-END
