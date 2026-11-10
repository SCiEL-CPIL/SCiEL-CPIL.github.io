import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    def calc_score(pts):
        if len(pts) < 2:
            return np.min(1.0 - np.linalg.norm(pts, axis=1))
        
        # Vectorized pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Min boundary distance
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(boundary_dists)
        
        return min(min_pair_dist, min_bound_dist)

    def gen_hex_lattice(spacing):
        points = []
        max_coord = 1.0
        k = int(max_coord / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > N:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii)[:N]
            points = points[idx]
        elif len(points) < N:
            # If not enough, add random points near the boundary
            while len(points) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, 1.0 - spacing / 2.0)
                points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
                
        return points

    # 1. Initial Configuration
    # Start with a slightly looser lattice to ensure we have enough points, then optimize
    # The optimal radius is ~0.053, so spacing ~0.106.
    # Let's try a few different initial spacings and pick the best one after a quick optimization?
    # Or just use a good starting point.
    # The seed program uses 0.105.
    # Let's use the adaptive spacing logic from inspiration 1 to ensure validity and good initial distribution.
    
    # Strategy:
    # 1. Generate initial points using a hex lattice with spacing tuned to fit N points.
    # 2. Use a smooth optimization loop (gradient ascent on soft-min) to maximize the minimum distance.
    # 3. The smooth objective allows for efficient vectorized gradient updates.
    
    # Let's determine a good initial spacing.
    # Area of unit circle is pi.
    # N circles of radius r have area N * pi * r^2.
    # Packing density ~ 0.9.
    # N * pi * r^2 / 0.9 <= pi  => r <= sqrt(0.9/N) ~ 0.0547.
    # Spacing d = 2r ~ 0.109.
    # However, boundary effects reduce density.
    # Let's try spacing = 0.108.
    
    points = gen_hex_lattice(0.108)
    
    # Add small noise to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    
    # Ensure initial validity (clip to circle)
    norms = np.linalg.norm(points, axis=1)
    points[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]

    best_points = points.copy()
    best_score = calc_score(points)
    
    # Optimization Loop
    # We use a soft-min approximation of the objective function.
    # Objective: J = -eps * log( sum exp(-d_k / eps) )
    # where d_k are the critical distances (pairwise and boundary).
    # Maximizing J is equivalent to maximizing the min distance when eps -> 0.
    
    eps = 0.01  # Softness parameter
    step_size = 0.005
    anneal_rate = 0.999
    step_decay = 0.999
    
    it = 0
    # We will run as many iterations as possible before deadline
    while time.time() < deadline:
        # Check time more frequently in later stages
        if it % 50 == 0 and time.time() > deadline - 5.0:
            break
            
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - norms
        
        # Weights for soft-min
        # w_k = exp(-d_k / eps) / Z
        pair_weights_raw = np.exp(-dists / eps)
        np.fill_diagonal(pair_weights_raw, 0.0)
        
        bound_weights_raw = np.exp(-bound_dists / eps)
        
        Z = np.sum(pair_weights_raw) + np.sum(bound_weights_raw)
        
        if Z == 0:
            break
            
        pair_weights = pair_weights_raw / Z
        bound_weights = bound_weights_raw / Z
        
        # Gradient of J = -eps * log(Z) w.r.t points
        # dJ/dp_i = sum_j pair_weights[i,j] * (p_i - p_j)/d_ij + bound_weights[i] * (-p_i/|p_i|)
        
        # Compute pair gradients
        # unit_vecs shape (N, N, 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[..., np.newaxis]
            # Handle zero distances (should not happen with noise, but safe)
            unit_vecs[~np.isfinite(unit_vecs)] = 0.0
            
        grad_pairs = np.sum(pair_weights[..., np.newaxis] * unit_vecs, axis=1)
        
        # Compute boundary gradients
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_norms = points / norms[:, np.newaxis]
            unit_norms[norms == 0] = 0.0
            
        grad_bounds = -bound_weights[:, np.newaxis] * unit_norms
        
        total_grad = grad_pairs + grad_bounds
        
        # Update
        points = points + step_size * total_grad
        
        # Project to unit circle to ensure validity
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
            
        # Anneal
        eps *= anneal_rate
        step_size *= step_decay
        
        # Check validity and score
        # To save time, we can check score every 10 iterations
        if it % 10 == 0:
            current_score = calc_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        it += 1
        
    # Final polish: check best_points validity
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
         best_points[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]
         
    return best_points
