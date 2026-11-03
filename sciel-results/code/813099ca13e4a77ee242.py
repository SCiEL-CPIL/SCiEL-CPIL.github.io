"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 300

# EVOLVE-BLOCK-START
import numpy as np

def _init_rings(N):
    """
    Initialize points in concentric rings.
    """
    points = []
    # Use rings with 6k points for k=1,2,...
    # Plus 1 at center
    points.append([0.0, 0.0])
    count = 1
    k = 1
    # Estimate max radius needed
    # Total area pi. N circles. r ~ 1/sqrt(N).
    # We want to fill up to radius ~0.95
    
    while count < N:
        n_in_ring = 6 * k
        # If adding this whole ring exceeds N, take only what's needed
        if count + n_in_ring > N:
            n_in_ring = N - count
        
        # Radius for this ring
        # Roughly proportional to k
        # Max radius approx 0.95
        # Let's say K_max is such that 3*K_max^2 ~ N
        K_max = int(np.sqrt(N / 3.0)) + 1
        r = 0.95 * (k / K_max)
        
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        # Add small random perturbation to break symmetry
        angles += np.random.uniform(-0.01, 0.01, n_in_ring)
        
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        
        count += n_in_ring
        k += 1
        
    points = np.array(points[:N])
    return points

def _energy_component(points, idx):
    """
    Calculate the energy contribution of point idx to the total system.
    E_i = sum_{j != i} 1/d_ij^2 + 1/(1-|p_i|)^2
    """
    p_i = points[idx]
    # Distance to all other points
    diffs = points - p_i
    dists_sq = np.sum(diffs**2, axis=1)
    # Set self-distance to inf to ignore in sum
    dists_sq[idx] = np.inf
    
    # Avoid division by zero for very close points (though idx is inf)
    # If any other point is extremely close, energy is huge
    inv_dists_sq = 1.0 / dists_sq
    inv_dists_sq[idx] = 0.0
    
    pair_energy = np.sum(inv_dists_sq)
    
    # Boundary energy
    norm_i = np.linalg.norm(p_i)
    if norm_i >= 1.0:
        # Should not happen if we clamp, but just in case
        bound_energy = 1e10
    else:
        bound_energy = 1.0 / (1.0 - norm_i)**2
        
    return pair_energy + bound_energy

def _simulate_annealing(points, steps=20000, T_start=0.05, T_end=0.0001):
    """
    Simulated Annealing to minimize energy (maximize min distance).
    """
    pts = points.copy()
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / steps)
    
    # Initial energy of each point
    # We don't need total energy, just delta.
    # But we need to be able to compute E_i quickly.
    # Precomputing all E_i is O(N^2). Doing it once is fine.
    
    for step in range(steps):
        # Pick a random point to move
        i = np.random.randint(N)
        old_p = pts[i].copy()
        
        # Propose new position
        # Step size decreases with temperature? Or fixed small step?
        # Let's use a step size that scales with T
        step_size = 0.01 * (T / T_start) + 0.001
        noise = np.random.normal(0, step_size, 2)
        new_p = old_p + noise
        
        # Clamp to unit circle
        norm_new = np.linalg.norm(new_p)
        if norm_new > 1.0:
            new_p = new_p / norm_new * 0.999999
            
        # Calculate Delta Energy
        # E_old_i = energy of point i with old_p
        # E_new_i = energy of point i with new_p
        # Note: The energy of *other* points j is unchanged because we only moved i?
        # Wait. E_total = sum_i E_i? No.
        # My energy definition was E_i = sum_{j!=i} 1/d_ij^2 + ...
        # The total system energy is NOT sum of E_i, because E_i includes interaction with j, and E_j includes interaction with i.
        # So E_total = 0.5 * sum_i (sum_{j!=i} 1/d_ij^2) + sum_i 1/(1-|p_i|)^2
        # If we move i, the change in total energy is:
        # Delta E = (New interaction of i with others) - (Old interaction of i with others)
        #          + (New boundary term for i) - (Old boundary term for i)
        # The interactions of j with k (j!=i, k!=i) are unchanged.
        # The interaction of j with i is the same as interaction of i with j.
        # So yes, Delta E is just the change in the "row" i of the interaction matrix + boundary term.
        
        # Calculate Old contribution of i
        # We can compute it on the fly or cache it.
        # Computing on the fly is O(N).
        
        # Old
        diffs_old = pts - old_p
        dists_sq_old = np.sum(diffs_old**2, axis=1)
        dists_sq_old[i] = np.inf
        inv_dists_sq_old = 1.0 / dists_sq_old
        inv_dists_sq_old[i] = 0.0
        old_pair = np.sum(inv_dists_sq_old)
        old_norm = np.linalg.norm(old_p)
        old_bound = 1.0 / (1.0 - old_norm)**2 if old_norm < 1.0 else 1e10
        old_E_i = old_pair + old_bound
        
        # New
        pts[i] = new_p # Temporarily update to use vectorized ops? No, let's just compute with new_p
        diffs_new = pts - new_p
        dists_sq_new = np.sum(diffs_new**2, axis=1)
        dists_sq_new[i] = np.inf
        inv_dists_sq_new = 1.0 / dists_sq_new
        inv_dists_sq_new[i] = 0.0
        new_pair = np.sum(inv_dists_sq_new)
        new_norm = np.linalg.norm(new_p)
        new_bound = 1.0 / (1.0 - new_norm)**2 if new_norm < 1.0 else 1e10
        new_E_i = new_pair + new_bound
        
        delta_E = new_E_i - old_E_i
        
        # Acceptance criterion
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            pts[i] = new_p
        else:
            pts[i] = old_p # Revert (already old_p if we didn't assign, but we did assign in diffs_new calc? 
                           # Wait, I did pts[i] = new_p before calculating diffs_new.
                           # So if rejected, I must revert.
        
        # Cool down
        T *= cooling
        
    return pts

def _score(points):
    """
    Calculate the score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    # Run multiple SA trials to find the best
    best_points = None
    best_score = -1.0
    
    # 3 restarts
    for _ in range(3):
        init_pts = _init_rings(N)
        # Small perturbation
        init_pts += np.random.normal(0, 0.001, size=init_pts.shape)
        
        # Ensure initial points are inside
        norms = np.linalg.norm(init_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            init_pts[mask] = init_pts[mask] / norms[mask, np.newaxis] * 0.999999
            
        optimized = _simulate_annealing(init_pts, steps=15000, T_start=0.05, T_end=0.0001)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # Final polish: A few steps of pure gradient descent or very low T SA?
    # Let's just do a low T SA on the best result
    if best_points is not None:
        final_optimized = _simulate_annealing(best_points, steps=5000, T_start=0.001, T_end=0.00001)
        final_score = _score(final_optimized)
        if final_score > best_score:
            best_points = final_optimized
            best_score = final_score

    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999999
        
    return best_points
# EVOLVE-BLOCK-END
