"""PSO + local polish for packing 50 equal circles in a unit disk."""
import numpy as np
import time

N = 50
TARGET_R = 0.1258254895304  # best known packing radius
DEADLINE = 115.0  # seconds, safety margin under 120s


def _score(pts):
    """Compute the packing radius for a set of centers."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint
    dist_to_origin = np.hypot(pts[:, 0], pts[:, 1])
    r_container = np.min(1.0 - dist_to_origin)
    if r_container < 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pairs = np.min(dists) / 2.0
    return min(r_container, r_pairs)


def _soft_score(pts, beta=50.0):
    """Soft-min surrogate for the objective, smoother for PSO."""
    if len(pts) != N:
        return 0.0
    dist_to_origin = np.hypot(pts[:, 0], pts[:, 1])
    r_container_vals = 1.0 - dist_to_origin
    # Soft min of container distances
    log_sum_exp_c = np.log(np.sum(np.exp(-beta * r_container_vals))) / beta
    r_container = -log_sum_exp_c
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pairs_vals = dists / 2.0
    # Soft min over all pairs
    flat_pairs = r_pairs_vals[np.triu_indices(N, k=1)]
    log_sum_exp_p = np.log(np.sum(np.exp(-beta * flat_pairs))) / beta
    r_pairs = -log_sum_exp_p
    return min(r_container, r_pairs)


def _hex_init():
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Try to fit N circles in a hex pattern
    for r in np.linspace(0.14, 0.08, 50):
        spacing = 2.0 * r
        pts = []
        for row in range(-10, 11):
            y = row * spacing * np.sqrt(3) / 2.0
            x_offset = (row % 2) * spacing / 2.0
            for col in range(-10, 11):
                x = col * spacing + x_offset
                if x * x + y * y <= (1.0 - r) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Sort by distance from center, take closest N
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists)[:N]
            return pts[idx]
    # Fallback: random
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.8, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _local_polish(pts, max_iter=200, tol=1e-12):
    """Local optimization: iteratively adjust points to maximize min distance."""
    pts = pts.copy()
    best_r = _score(pts)
    
    for it in range(max_iter):
        # Compute current min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the closest pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        min_dist = dists[i, j]
        
        # Also check container
        dist_to_origin = np.hypot(pts[:, 0], pts[:, 1])
        min_cont_idx = np.argmin(1.0 - dist_to_origin)
        min_cont_dist = 1.0 - dist_to_origin[min_cont_idx]
        
        # Determine which constraint is binding
        if min_dist / 2.0 < min_cont_dist:
            # Move i and j apart
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            # Move each by a small step
            step = 0.001
            pts[i] += direction * step
            pts[j] -= direction * step
        else:
            # Move the point closest to the boundary inward
            idx = min_cont_idx
            direction = -pts[idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            step = 0.001
            pts[idx] += direction * step
        
        # Check if improved
        new_r = _score(pts)
        if new_r > best_r:
            best_r = new_r
        elif it > 50 and new_r < best_r - 1e-10:
            # Revert
            pts[i] -= direction * step if min_dist / 2.0 < min_cont_dist else 0
            if min_dist / 2.0 >= min_cont_dist:
                pts[idx] -= direction * step
            break
    
    return pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initialize with hexagonal packing
    best_pts = _hex_init()
    best_score = _score(best_pts)
    
    # PSO setup
    n_particles = 8
    dim = N * 2  # 2D coordinates for N points
    
    # Initialize swarm
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    p_best = np.zeros((n_particles, dim))
    p_best_scores = np.zeros(n_particles)
    g_best = best_pts.flatten()
    g_best_score = best_score
    
    # Initialize particles around the hex solution with some noise
    for i in range(n_particles):
        noise_scale = 0.05 if i == 0 else 0.15
        particles[i] = best_pts.flatten() + np.random.randn(dim) * noise_scale
        # Ensure points stay within container
        pts = particles[i].reshape(N, 2)
        pts[:, 0] = np.clip(pts[:, 0], -0.95, 0.95)
        pts[:, 1] = np.clip(pts[:, 1], -0.95, 0.95)
        particles[i] = pts.flatten()
        p_best[i] = particles[i].copy()
        p_best_scores[i] = _score(particles[i].reshape(N, 2))
        if p_best_scores[i] > g_best_score:
            g_best_score = p_best_scores[i]
            g_best = p_best[i].copy()
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    max_vel = 0.1
    
    iteration = 0
    while time.time() < deadline - 30.0:  # Leave 30s for polishing
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            # Update velocity
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (g_best - particles[i]))
            
            # Clip velocity
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            # Update position
            particles[i] += velocities[i]
            
            # Keep points within container
            pts = particles[i].reshape(N, 2)
            pts[:, 0] = np.clip(pts[:, 0], -0.98, 0.98)
            pts[:, 1] = np.clip(pts[:, 1], -0.98, 0.98)
            particles[i] = pts.flatten()
            
            # Evaluate
            score = _score(pts)
            
            # Update personal best
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best[i] = particles[i].copy()
                # Update global best
                if score > g_best_score:
                    g_best_score = score
                    g_best = particles[i].copy()
        
        iteration += 1
        # Decay inertia
        w = max(0.4, w * 0.995)
    
    # Get best solution from PSO
    best_pts = g_best.reshape(N, 2)
    best_score = g_best_score
    
    # Also check the initial hex solution
    hex_score = _score(best_pts)
    if hex_score > best_score:
        best_pts = _hex_init()
        best_score = hex_score
    
    # Local polish
    if time.time() < deadline - 5.0:
        polished_pts, polished_score = _local_polish(best_pts, max_iter=100)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to hex
        best_pts = _hex_init()
    
    return best_pts


# Run construct to test
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Best known: {TARGET_R}")
