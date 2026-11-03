"""Physics-based repulsion packer for 100 equal spheres in a unit cube."""
import numpy as np
import time

N = 100
CUBE_HALF = 0.5

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _energy(points, r_target=0.11):
    """
    Compute a repulsion energy.
    We want to minimize the sum of repulsive forces between spheres that are
    closer than a target distance, and forces from walls.
    This acts as a smooth surrogate for the minimum distance constraint.
    """
    pts = np.asarray(points, dtype=np.float64)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Repulsion: 1/d^2 or similar. We only care about pairs closer than some cutoff
    # To make it smooth and focus on the minimum, we can use a soft-min or just sum of inverse squares
    # Sum of 1/d^2 is good because it strongly penalizes close pairs.
    # But it can be numerically unstable if d is very small.
    # Let's use a potential: E = sum_{i<j} (r_target / d_ij)^4 
    # This is the WCA (Weeks-Chandler-Andersen) potential form, which is purely repulsive.
    
    # We compute pairwise potential
    # To avoid O(N^2) memory if N is large, but N=100 is fine.
    # d_ij is the distance.
    # We want to maximize min distance.
    # Minimizing sum (1/d)^k is a good proxy.
    
    # Wall repulsion: Push points away from walls if they are too close.
    # Wall distance w_i = min(0.5 - |x_i|, 0.5 - |y_i|, 0.5 - |z_i|)
    # If w_i < r_target, add penalty (r_target / w_i)^4
    
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    # Avoid division by zero
    wall_dists = np.maximum(wall_dists, 1e-6)
    wall_penalty = np.sum((r_target / wall_dists) ** 4)
    
    # Pairwise penalty
    # Only sum upper triangle to avoid double counting and self-interaction
    i, j = np.triu_indices(N, k=1)
    d = dists[i, j]
    d = np.maximum(d, 1e-6)
    pair_penalty = np.sum((r_target / d) ** 4)
    
    return pair_penalty + wall_penalty

def _relax(points, r_target=0.11, steps=500, lr=0.01):
    """
    Simple gradient descent on the energy landscape.
    Since we don't have analytic gradients easily for the discrete min, 
    we can use numerical gradients or just random jitter with acceptance based on energy.
    Actually, let's use a "Repulsion Step" approach:
    For each point, calculate the net repulsive force from all other points and walls.
    Move the point in the direction of the force.
    """
    pts = points.copy()
    current_energy = _energy(pts, r_target)
    
    for step in range(steps):
        # Calculate forces
        # Force on i from j is proportional to (r_target^4 / d^5) * (r_i - r_j)
        # This is the gradient of (r_target/d)^4 w.r.t r_i
        # dE/dr_i = sum_j -4 * r_target^4 * d^-5 * (r_i - r_j)
        # So force direction is (r_i - r_j) / d^5
        
        forces = np.zeros_like(pts)
        
        # Pairwise forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, 1e-6) # Avoid div by 0, diagonal force is 0 anyway
        
        # Weight matrix: W_ij = 4 * r_target^4 / dists^5
        # We only want to apply force if dists < some cutoff? 
        # The potential is always repulsive, but decays fast.
        # For N=100, computing all pairs is fine.
        
        # To save time, we can ignore pairs with dist > 3 * r_target
        mask = dists < 3.0 * r_target
        # Set diagonal to 0
        mask[np.diag_indices(N)] = False
        
        weights = np.zeros_like(dists)
        weights[mask] = 4.0 * (r_target ** 4) / (dists[mask] ** 5)
        
        # Force on i is sum_j weights_ij * (pts_i - pts_j)
        # F_i = sum_j W_ij * (P_i - P_j) = (sum_j W_ij) * P_i - sum_j W_ij * P_j
        sum_w = weights.sum(axis=1, keepdims=True)
        weighted_sum = weights @ pts
        
        pair_forces = sum_w * pts - weighted_sum
        
        # Wall forces
        # Gradient of (r_target / w)^4 where w = 0.5 - |x|
        # dE/dx = -4 * r_target^4 * w^-5 * dw/dx
        # dw/dx = -sign(x)
        # So dE/dx = 4 * r_target^4 * w^-5 * sign(x)
        # Force is -dE/dx = -4 * r_target^4 * w^-5 * sign(x)
        # This pushes x towards 0 if x>0, and towards 0 if x<0. Correct.
        
        abs_pts = np.abs(pts)
        max_abs = abs_pts.max(axis=1)
        # Find which coordinate is the limiting one for the wall distance
        # If max_abs is from x, then w = 0.5 - |x|
        
        wall_forces = np.zeros_like(pts)
        for i in range(N):
            w = 0.5 - max_abs[i]
            if w < 3.0 * r_target: # Only apply if close to wall
                w_safe = max(w, 1e-6)
                mag = 4.0 * (r_target ** 4) / (w_safe ** 5)
                # Direction is towards center: -sign(x)
                # But we need to apply to the coordinate that determined w
                # Actually, the gradient is only non-zero for the coordinate that is max.
                # If multiple are max, it's a corner, gradient is sum of signs.
                # Let's just apply to all coordinates that are close to their respective walls?
                # No, the potential is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
                # The gradient of min is the gradient of the active term.
                
                # Identify active coordinate(s)
                active = np.where(np.isclose(abs_pts[i], max_abs[i]))[0]
                for k in active:
                    wall_forces[i, k] -= mag * np.sign(pts[i, k])
        
        total_forces = pair_forces + wall_forces
        
        # Update
        pts += lr * total_forces
        
        # Clamp to cube (with small margin to avoid numerical issues on boundary)
        pts = np.clip(pts, -0.49, 0.49)
        
        # Optional: Simulated Annealing style acceptance?
        # For now, simple gradient descent.
        
    return pts

def _push_local_search(points, iterations=200):
    """
    Local search that specifically targets the minimum distance pairs.
    1. Find the pair (i, j) with minimum distance.
    2. Move i and j apart in the direction of their separation vector.
    3. Also check walls and push away if needed.
    """
    pts = points.copy()
    best_r = _score(pts)
    
    for _ in range(iterations):
        # Find min distance pair
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(dists.argmin(), dists.shape)
        i, j = min_idx
        d = dists[i, j]
        
        if d > 2.0 * best_r + 0.01: # If min distance is already well above 2*r, we are limited by walls or other pairs
             # In this case, try to optimize the worst wall constraint
             margins = 0.5 - np.abs(pts).max(axis=1)
             min_margin_idx = margins.argmin()
             # Move that point towards center
             p = pts[min_margin_idx]
             # Direction to center is -p
             # But we need to move it in a way that increases the min margin.
             # The min margin is determined by the component with max abs.
             k = np.argmax(np.abs(p))
             # Move p[k] towards 0
             step = 0.01
             new_p = p.copy()
             new_p[k] -= np.sign(p[k]) * step
             if abs(new_p[k]) < 0.5:
                 pts[min_margin_idx] = new_p
        else:
            # Move i and j apart
            vec = pts[i] - pts[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Coincident, random kick
                pts[i] += np.random.normal(0, 0.01, 3)
                pts[j] -= np.random.normal(0, 0.01, 3)
            else:
                unit_vec = vec / norm
                step = 0.005
                pts[i] += unit_vec * step
                pts[j] -= unit_vec * step
        
        # Clamp
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check if improved
        new_r = _score(pts)
        if new_r > best_r:
            best_r = new_r
        else:
            # Revert? Or keep and try to fix?
            # For simplicity, revert if it got worse significantly
            # Actually, moving apart should always increase min dist for that pair,
            # but might decrease for others.
            # Let's just keep the move if it doesn't violate bounds.
            pass
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Physics relaxation with increasing target radius
    # Start with a small target to get them separated, then increase
    targets = [0.05, 0.08, 0.10, 0.11, 0.111, 0.1113]
    
    for target in targets:
        if time.time() > deadline - 10:
            break
        # Relax with this target
        relaxed = _relax(best_points, r_target=target, steps=100, lr=0.005)
        score = _score(relaxed)
        if score > best_score:
            best_points = relaxed
            best_score = score
            
    # Phase 2: Push local search to polish
    if time.time() < deadline - 5:
        polished = _push_local_search(best_points, iterations=500)
        score = _score(polished)
        if score > best_score:
            best_points = polished
            best_score = score
            
    # Final check
    return best_points

# EVOLVE-BLOCK-END
