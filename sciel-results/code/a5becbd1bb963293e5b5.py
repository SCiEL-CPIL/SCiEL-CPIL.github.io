"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50
rng = np.random.default_rng(42)
deadline = 0.0

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _force_directed_step(centers, dt=0.01, repulsion_strength=1.0, wall_strength=10.0):
    """
    One step of force-directed relaxation.
    Circles repel each other (inverse square law) and are pushed away from walls.
    """
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion: F = k * r_hat / r^2
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small perturbation if coincident
                diff = rng.normal(0, 0.001, 2)
                dist = np.linalg.norm(diff)
            force_mag = repulsion_strength / (dist * dist + 1e-6)
            force_dir = diff / dist
            forces[i] += force_mag * force_dir
            forces[j] -= force_mag * force_dir
    
    # Wall repulsion: strong force when close to wall
    for i in range(N):
        x, y = centers[i]
        # Distance to each wall
        dist_left = x + 0.5
        dist_right = 0.5 - x
        dist_bottom = y + 0.5
        dist_top = 0.5 - y
        
        # Force proportional to 1/d^2 when close
        if dist_left < 0.1:
            forces[i, 0] += wall_strength / (dist_left * dist_left + 1e-6)
        if dist_right < 0.1:
            forces[i, 0] -= wall_strength / (dist_right * dist_right + 1e-6)
        if dist_bottom < 0.1:
            forces[i, 1] += wall_strength / (dist_bottom * dist_bottom + 1e-6)
        if dist_top < 0.1:
            forces[i, 1] -= wall_strength / (dist_top * dist_top + 1e-6)
    
    # Update positions with velocity damping
    new_centers = centers + dt * forces
    new_centers = np.clip(new_centers, -0.4999, 0.4999)
    
    return new_centers

def _local_refinement(centers, max_iter=500, tol=1e-10):
    """
    Local refinement using coordinate descent on the minimum distance.
    For each circle, try small moves in 8 directions to improve min distance.
    """
    current_radius = _compute_radius(centers)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]) / np.sqrt(2)
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    for ss in step_sizes:
        improved = True
        while improved and max_iter > 0:
            improved = False
            for i in range(N):
                for d in directions:
                    new_centers = centers.copy()
                    new_centers[i] += ss * d
                    new_centers = np.clip(new_centers, -0.4999, 0.4999)
                    r = _compute_radius(new_centers)
                    if r > current_radius + 1e-12:
                        centers = new_centers
                        current_radius = r
                        improved = True
                        max_iter -= 1
                        if max_iter <= 0:
                            break
                if max_iter <= 0:
                    break
            if max_iter <= 0:
                break
    
    return centers, current_radius

def _anneal_with_forces(centers, T_start=0.05, T_end=0.001, max_steps=50000):
    """
    Simulated annealing where moves are guided by force-directed physics.
    """
    rng = np.random.default_rng(42)
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    alpha = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() > deadline - 60.0:
            break
        
        T = T_start * (alpha ** step)
        
        # Choose 1-3 circles to move
        num_movers = rng.integers(1, 4)
        indices = rng.choice(N, size=num_movers, replace=False)
        
        # Step size based on temperature
        step_size = 0.03 * (0.3 + 0.7 * T / T_start)
        
        new_centers = centers.copy()
        new_centers[indices] += rng.normal(0, step_size, (num_movers, 2))
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        # Apply a few force-directed steps to relax
        for _ in range(3):
            new_centers = _force_directed_step(new_centers, dt=0.005, repulsion_strength=0.5, wall_strength=5.0)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            delta_r = new_radius - current_radius
            if delta_r > -10 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    centers = new_centers
                    current_radius = new_radius
    
    return best_centers, best_radius

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    inits = []
    
    # Init 1: Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:N].copy()
    init1 += rng.normal(0, 0.02, init1.shape)
    inits.append(np.clip(init1, -0.49, 0.49))
    
    # Init 2: Hexagonal
    points = []
    spacing = 0.18
    for i in range(12):
        for j in range(12):
            x = -0.5 + j * spacing + (i % 2) * spacing / 2
            y = -0.5 + i * spacing * 0.866
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
    rng.shuffle(points)
    init2 = np.array(points[:N])
    init2 += rng.normal(0, 0.015, init2.shape)
    inits.append(np.clip(init2, -0.49, 0.49))
    
    # Init 3: Random
    init3 = rng.uniform(-0.45, 0.45, (N, 2))
    inits.append(init3)
    
    # Evaluate all inits
    radii = [_compute_radius(c) for c in inits]
    best_init_idx = np.argmax(radii)
    
    best_centers = inits[best_init_idx].copy()
    best_radius = radii[best_init_idx]
    
    # Phase 1: Force-directed relaxation from best init
    if time.time() < deadline - 80.0:
        relaxed = best_centers.copy()
        for step in range(2000):
            if time.time() > deadline - 80.0:
                break
            relaxed = _force_directed_step(relaxed, dt=0.01, repulsion_strength=1.0, wall_strength=10.0)
            r = _compute_radius(relaxed)
            if r > best_radius:
                best_radius = r
                best_centers = relaxed.copy()
    
    # Phase 2: Annealing with force guidance
    if time.time() < deadline - 30.0:
        centers_a, radius_a = _anneal_with_forces(best_centers, T_start=0.04, T_end=0.0005, max_steps=30000)
        if radius_a > best_radius:
            best_radius = radius_a
            best_centers = centers_a
    
    # Phase 3: Local refinement
    if time.time() < deadline - 5.0:
        refined, radius_r = _local_refinement(best_centers, max_iter=1000)
        if radius_r > best_radius:
            best_radius = radius_r
            best_centers = refined
    
    # Phase 4: Final force-directed polish
    if time.time() < deadline - 2.0:
        polished = best_centers.copy()
        for step in range(500):
            if time.time() > deadline - 2.0:
                break
            polished = _force_directed_step(polished, dt=0.005, repulsion_strength=0.5, wall_strength=5.0)
            r = _compute_radius(polished)
            if r > best_radius:
                best_radius = r
                best_centers = polished
    
    return best_centers
# EVOLVE-BLOCK-END
