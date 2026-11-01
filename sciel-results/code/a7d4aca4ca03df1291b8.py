"""
Strategy: 3D Harmonic Oscillator / Hard-Core Repulsion Potential
Instead of approximating the min-distance with log-sum-exp (which requires high
sharpness to become accurate and suffers from vanishing gradients or stiffness),
we define a smooth energy function E = sum (1/d^2) for all pairs and walls.
Minimizing E maximizes the minimum distance.
We use vectorized gradient descent with momentum, which is much faster per step
than L-BFGS-B for this specific structure, allowing many more iterations in the time budget.
"""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Jittered Simple Cubic Lattice
    # A 4x4x4 grid has 64 points. We take the first 50.
    # This provides a good starting spacing (0.25) which is better than random.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    rng = np.random.default_rng(42)
    centers = grid[:N].copy()
    centers += rng.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Momentum variables
    velocity = np.zeros_like(centers)
    momentum = 0.9
    
    # Optimization parameters
    # We will use a variable step size (learning rate) that decays over time.
    # Start with a larger step to escape local minima quickly, then refine.
    
    # We define the energy as sum of 1/d^2 for pairs and 1/wall_dist^2 for walls.
    # Gradient of 1/d^2 wrt pos is -2/d^3 * diff_vec.
    # Gradient of 1/w^2 wrt pos is -2/w^3 * sign(pos) (for the limiting coordinate).
    
    # To make the gradient computation vectorized and stable:
    # We iterate in a loop.
    
    current_centers = centers.copy()
    
    # We will run a loop that adapts the learning rate.
    # Phase 1: Large steps to find a good basin (0-20s)
    # Phase 2: Medium steps (20-60s)
    # Phase 3: Small steps for precision (60-110s)
    
    phases = [
        {"lr": 0.005, "until": 20.0},
        {"lr": 0.001, "until": 60.0},
        {"lr": 0.0001, "until": 110.0},
        {"lr": 0.00001, "until": 115.0}
    ]
    
    current_phase_idx = 0
    lr = phases[0]["lr"]
    
    while time.time() < deadline:
        # Check phase transition
        if current_phase_idx < len(phases) - 1 and time.time() > phases[current_phase_idx]["until"]:
            current_phase_idx += 1
            lr = phases[current_phase_idx]["lr"]
            # Reset velocity on phase change to avoid carrying over large momentum
            velocity *= 0.5 
        
        # Compute Gradient
        # 1. Pairwise repulsion
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero (should not happen with jitter, but safe)
        dists_sq = np.maximum(dists_sq, 1e-12)
        
        # Force magnitude: d(1/d^2)/d(pos) = -2 * (pos_i - pos_j) / d^4
        # We want to minimize energy, so we move in direction of -grad.
        # Grad E wrt c_i = sum_{j != i} -2 * (c_i - c_j) / d_ij^4
        # So -grad E wrt c_i = sum_{j != i} 2 * (c_i - c_j) / d_ij^4
        # This pushes i away from j.
        
        # Compute 1/d^4 for all pairs
        inv_d4 = 1.0 / (dists_sq**2)
        np.fill_diagonal(inv_d4, 0.0)
        
        # Gradient contribution from pairs:
        # grad_pairs[i] = -2 * sum_j (c_i - c_j) * inv_d4[i,j]
        # Note: diff[i,j] = c_i - c_j
        # So grad_pairs[i] = -2 * sum_j diff[i,j] * inv_d4[i,j]
        grad_pairs = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        # 2. Wall repulsion
        # Energy term: 1 / (0.5 - |c_k|)^2 for the limiting coordinate?
        # Actually, to keep it smooth and simple, let's use the distance to the nearest wall.
        # d_wall = 0.5 - max(|c|)
        # E_wall = sum 1/d_wall^2
        # This is non-differentiable at axes.
        # Better: Sum 1/(0.5 - c)^2 + 1/(0.5 + c)^2 for each coordinate?
        # This creates a "box" potential.
        # E_box = sum_k sum_dim [ 1/(0.5 - c_kd)^2 + 1/(0.5 + c_kd)^2 ]
        # Grad wrt c_kd: 2/(0.5 - c_kd)^3 - 2/(0.5 + c_kd)^3
        # This is smooth everywhere inside the box.
        
        # Let's use this box potential. It's stronger near walls.
        c = current_centers
        # Term 1: 1/(0.5 - c)^2
        d1 = 0.5 - c
        d1 = np.maximum(d1, 1e-6)
        grad_wall_1 = 2.0 / (d1**3) # d/dc (1/d^2) = -2/d^3 * d/dc(0.5-c) = -2/d^3 * (-1) = 2/d^3
        
        # Term 2: 1/(0.5 + c)^2
        d2 = 0.5 + c
        d2 = np.maximum(d2, 1e-6)
        grad_wall_2 = -2.0 / (d2**3) # d/dc (1/d^2) = -2/d^3 * 1 = -2/d^3
        
        grad_walls = grad_wall_1 + grad_wall_2
        
        # Total Gradient
        grad = grad_pairs + grad_walls
        
        # Update with Momentum
        velocity = momentum * velocity - lr * grad
        new_centers = current_centers + velocity
        
        # Clip to box (hard constraint)
        # Note: Clipping can cause sticking to walls. 
        # The wall potential should handle this, but clipping ensures validity.
        new_centers = np.clip(new_centers, -0.49999, 0.49999)
        
        # Check validity and score
        # To save time, we don't compute full radius every single step.
        # We compute it every 10 steps or if time allows.
        
        # Simple check: ensure no NaN/Inf
        if not np.all(np.isfinite(new_centers)):
            break
            
        current_centers = new_centers
        
        # Periodic scoring
        # We can't afford O(N^2) radius calc every step if N=50 and steps are high?
        # O(2500) ops is very fast. Let's do it every 5 steps.
        if int(time.time() * 10) % 5 == 0: 
             # Better: use a counter
             pass
        
        # Let's just use a time-based check for scoring to be safe
        # Or simply score every 10 iterations.
        
        # We'll track iteration count
        # (Implementing a simple counter in the loop is easier)
        
    # The loop above needs a bit more structure for scoring.
    # Let's rewrite the loop properly.
    
    # RE-IMPLEMENTATION OF LOOP FOR CLARITY AND CORRECTNESS
    
    current_centers = centers.copy()
    velocity = np.zeros_like(centers)
    iter_count = 0
    
    while time.time() < deadline:
        if current_phase_idx < len(phases) - 1 and time.time() > phases[current_phase_idx]["until"]:
            current_phase_idx += 1
            lr = phases[current_phase_idx]["lr"]
            velocity *= 0.5 
            
        # Compute Gradient
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq = np.maximum(dists_sq, 1e-12)
        
        inv_d4 = 1.0 / (dists_sq**2)
        np.fill_diagonal(inv_d4, 0.0)
        
        grad_pairs = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        c = current_centers
        d1 = np.maximum(0.5 - c, 1e-6)
        grad_wall_1 = 2.0 / (d1**3)
        
        d2 = np.maximum(0.5 + c, 1e-6)
        grad_wall_2 = -2.0 / (d2**3)
        
        grad_walls = grad_wall_1 + grad_wall_2
        grad = grad_pairs + grad_walls
        
        velocity = momentum * velocity - lr * grad
        new_centers = current_centers + velocity
        new_centers = np.clip(new_centers, -0.49999, 0.49999)
        
        if not np.all(np.isfinite(new_centers)):
            break
            
        current_centers = new_centers
        iter_count += 1
        
        if iter_count % 10 == 0:
            r = _compute_radius(current_centers)
            if r > best_radius:
                best_radius = r
                best_centers = current_centers.copy()
                
    # Final check
    r = _compute_radius(current_centers)
    if r > best_radius:
        best_radius = r
        best_centers = current_centers.copy()
        
    return best_centers

# EVOLVE-BLOCK-END
