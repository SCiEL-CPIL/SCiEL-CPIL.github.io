"""
Strategy: Physics-based Molecular Dynamics (MD) simulation.
Instead of gradient-based optimization, we treat the circles as particles in a 2D box.
We use a 'soft' potential (harmonic repulsion) to resolve overlaps and an 'attractive'
force (or simply a pressure via moving walls) to push them together.
By simulating this system with velocity Verlet integration and damping, we find a
locally optimal packing configuration that is often harder to reach with pure gradient
descent due to the 'rugged' energy landscape of hard-sphere packings.

The process:
1. Initialize with a hexagonal grid.
2. Run MD simulation:
   - Force = -dU/dx, where U is the sum of pair potentials + wall potentials.
   - Use a soft potential U(r) = 0.5 * k * (2r - d)^2 for d < 2r to allow smooth forces.
   - Add a small attractive force to keep them centered (or use a pressure term).
   - Damp velocities to settle into a minimum.
3. Extract the best radius found during or after the simulation.
4. Polish with a few steps of L-BFGS-B on the violation metric to clean up numerical noise.
"""

import numpy as np
import time

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distance
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles as a good initial guess."""
    best_cents = None
    best_r = 0

    # Try a few different row counts to find the best hexagonal fit
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (n_rows - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = r if (row_idx % 2 == 1) else 0.0
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _md_simulate(cents, r_target, time_limit=10.0):
    """
    Run a molecular dynamics simulation to pack circles of radius r_target.
    We use a soft harmonic potential for overlaps and a confining potential for walls.
    """
    start_time = time.time()
    pos = cents.copy()
    vel = np.zeros_like(pos)
    
    # Parameters
    k_rep = 1000.0  # Repulsion stiffness
    k_wall = 1000.0 # Wall stiffness
    damping = 0.95  # Velocity damping factor
    dt = 0.005      # Time step
    
    # Precompute pairs for efficiency (N is small, so O(N^2) is fine)
    n = len(pos)
    
    def compute_forces(p):
        forces = np.zeros_like(p)
        # Pairwise repulsion
        # We want to avoid overlaps. If dist < 2*r_target, push apart.
        # Using a soft potential: F = k * (2*r_target - dist) * unit_vec
        for i in range(n):
            for j in range(i+1, n):
                diff = p[i] - p[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r_target and dist > 1e-8:
                    # Force magnitude
                    f_mag = k_rep * (2 * r_target - dist)
                    f_vec = f_mag * (diff / dist)
                    forces[i] += f_vec
                    forces[j] -= f_vec
                elif dist < 1e-8:
                    # If coincident, push randomly
                    forces[i] += np.random.randn(2) * 10
                    forces[j] -= np.random.randn(2) * 10
        
        # Wall repulsion
        # If x < -0.5 + r, push right. If x > 0.5 - r, push left.
        for i in range(n):
            # Left wall
            if p[i, 0] < -0.5 + r_target:
                forces[i, 0] += k_wall * (-0.5 + r_target - p[i, 0])
            # Right wall
            if p[i, 0] > 0.5 - r_target:
                forces[i, 0] -= k_wall * (p[i, 0] - (0.5 - r_target))
            # Bottom wall
            if p[i, 1] < -0.5 + r_target:
                forces[i, 1] += k_wall * (-0.5 + r_target - p[i, 1])
            # Top wall
            if p[i, 1] > 0.5 - r_target:
                forces[i, 1] -= k_wall * (p[i, 1] - (0.5 - r_target))
                
        return forces
    
    # Initial forces
    f = compute_forces(pos)
    
    steps = 0
    max_steps = 2000
    
    while steps < max_steps:
        if time.time() - start_time > time_limit:
            break
        
        # Velocity Verlet
        pos += vel * dt + 0.5 * f * dt**2
        f_new = compute_forces(pos)
        vel = (vel + 0.5 * (f + f_new) * dt) * damping
        f = f_new
        
        steps += 1
        
        # Check if converged (forces are small)
        if np.linalg.norm(f) < 1e-4:
            break
            
    return pos

def _polish(cents, r_target, time_limit=10.0):
    """Polish a configuration to achieve at least radius r_target using L-BFGS-B."""
    from scipy.optimize import minimize
    start_time = time.time()
    x0 = cents.reshape(-1)
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def loss(x):
        c = x.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Violation if dist < 2*r_target
        viol = np.maximum(2 * r_target - dists, 0.0)
        loss_pair = np.sum(viol**2)
        
        # Wall violation
        wall_viol = np.maximum(r_target - (0.5 - np.abs(c[:, 0])), 0.0)
        wall_viol += np.maximum(r_target - (0.5 - np.abs(c[:, 1])), 0.0)
        loss_wall = np.sum(wall_viol**2)
        
        return loss_pair + loss_wall
    
    def grad(x):
        c = x.reshape(N, 2)
        g = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        viol = np.maximum(2 * r_target - dists, 0.0)
        coeff = 2.0 * viol / dists
        np.fill_diagonal(coeff, 0.0)
        g += -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall gradients
        mask_pos_x = c[:, 0] > 0
        mask_neg_x = c[:, 0] < 0
        mask_pos_y = c[:, 1] > 0
        mask_neg_y = c[:, 1] < 0
        
        viol_px = np.maximum(r_target - 0.5 + c[mask_pos_x, 0], 0.0)
        viol_nx = np.maximum(r_target - 0.5 - c[mask_neg_x, 0], 0.0)
        viol_py = np.maximum(r_target - 0.5 + c[mask_pos_y, 1], 0.0)
        viol_ny = np.maximum(r_target - 0.5 - c[mask_neg_y, 1], 0.0)
        
        g[mask_pos_x, 0] += 2.0 * viol_px
        g[mask_neg_x, 0] += -2.0 * viol_nx
        g[mask_pos_y, 1] += 2.0 * viol_py
        g[mask_neg_y, 1] += -2.0 * viol_ny
        
        return g.flatten()
    
    try:
        result = minimize(
            loss, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return cents
        return result.x.reshape(N, 2)
    except Exception:
        return cents

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Initial guess
    cents0, r0 = _construct_hexagonal()
    best_cents = cents0.copy()
    best_r = r0
    
    # 2. MD Simulation to improve packing
    # We try to pack for a radius slightly larger than the current best
    r_target_md = best_r * 1.02
    if time.time() - start_time < deadline - 20:
        # Run MD
        cents_md = _md_simulate(cents0, r_target_md, time_limit=15.0)
        r_md = _score(cents_md)
        
        # If MD found a better configuration, use it
        if r_md > best_r:
            best_r = r_md
            best_cents = cents_md.copy()
            
        # Even if r_md didn't beat best_r directly, the configuration might be closer
        # to a good local minimum for a slightly higher radius.
        # Let's try to polish the MD result for the current best_r to see if we can
        # reach it cleanly, then use that as a base for higher radii.
        polished_md = _polish(cents_md, best_r, time_limit=10.0)
        r_polished_md = _score(polished_md)
        if r_polished_md > best_r:
            best_r = r_polished_md
            best_cents = polished_md.copy()
            
    # 3. Binary search for the maximum feasible radius
    # Start from the best found so far
    r_low = best_r
    r_high = best_r * 1.05
    
    # Check if r_high is feasible
    if time.time() - start_time < deadline - 10:
        polished_high = _polish(best_cents, r_high, time_limit=10.0)
        r_polished_high = _score(polished_high)
        if r_polished_high >= r_high - 1e-6:
            r_low = r_high
            best_cents = polished_high.copy()
            best_r = r_polished_high
        else:
            r_high = best_r * 1.02
    
    # Binary search iterations
    x0 = best_cents.reshape(-1)
    
    for i in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Use the current best configuration as the starting point for polish
        polished = _polish(best_cents, r_mid, time_limit=10.0)
        r_polished = _score(polished)
        
        if r_polished >= r_mid - 1e-6:
            r_low = r_mid
            best_r = r_polished
            best_cents = polished.copy()
        else:
            r_high = r_mid
    
    # 4. Final polish
    if time.time() - start_time < deadline - 2:
        best_cents = _polish(best_cents, best_r, time_limit=5.0)
        final_r = _score(best_cents)
        if final_r > best_r:
            best_r = final_r
    
    # Ensure validity
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents

# EVOLVE-BLOCK-END
