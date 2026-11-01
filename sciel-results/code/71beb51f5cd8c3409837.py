"""
Strategy: Physical Simulation (Molecular Dynamics) with Radius Inflation.
Instead of optimizing a smooth surrogate of the minimum distance, we simulate
particles with a strong repulsive force (like hard spheres with a soft core)
and gradually increase the "target" radius. This drives the system to explore
configurations where the minimum distance is large, effectively performing
a simulated annealing-like search in the configuration space.
"""
import numpy as np
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    if n < 2:
        return np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    # Vectorized pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Container margin
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    
    return min(min_pair, min_margin)

def _md_step(pts, target_r, dt, k_rep, k_wall):
    """
    Perform one step of molecular dynamics.
    pts: (N, 2) array of positions.
    target_r: Current target radius (we want min_dist >= 2*target_r).
    dt: Time step.
    k_rep: Repulsion strength.
    k_wall: Wall repulsion strength.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsive forces
    # We want to push particles apart if they are closer than 2*target_r.
    # Force magnitude: k_rep * (2*target_r - dist) if dist < 2*target_r, else 0.
    # To make it smoother and avoid singularities, we can use a soft potential.
    # Let's use a linear repulsion for simplicity and stability, clipped.
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small perturbation
                diff = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
                dist = np.linalg.norm(diff)
            
            min_dist = 2.0 * target_r
            if dist < min_dist:
                # Overlap penalty
                # Force direction: i pushed away from j
                f_mag = k_rep * (min_dist - dist)
                f_vec = f_mag * (diff / dist)
                forces[i] += f_vec
                forces[j] -= f_vec
            
            # Add a small attractive force to keep them from flying apart?
            # No, we want them to pack tight. The wall will keep them in.
            # But without attraction, they might drift to the wall immediately.
            # Actually, for packing, we just want to maximize min dist.
            # The "target_r" inflation acts as the driving force.
            
    # Wall repulsion
    # If particle is closer to wall than target_r, push it in.
    for i in range(n):
        r = np.linalg.norm(pts[i])
        if r > 1.0 - target_r:
            # Overlap with wall
            f_mag = k_wall * (r - (1.0 - target_r))
            # Force direction: towards center
            if r > 1e-10:
                f_vec = -f_mag * (pts[i] / r)
            else:
                f_vec = np.array([0.0, 0.0])
            forces[i] += f_vec

    # Velocity Verlet or simple Euler with damping
    # Let's use simple damped Euler for stability.
    # v = v * damping + forces * dt
    # p = p + v * dt
    
    # We don't explicitly track velocity in this simple version, 
    # but we can add a small random jitter to escape local minima.
    # Let's implement a simple "push" method which is effectively 
    # gradient ascent on the repulsive potential.
    
    # Actually, let's stick to a simple iterative relaxation:
    # Move each particle in the direction of the net force.
    # Add some randomness to escape local minima.
    
    pts_new = pts.copy()
    for i in range(n):
        # Add small random noise
        noise = np.random.normal(0, 0.001, size=2)
        pts_new[i] = pts[i] + forces[i] * dt + noise
        
        # Project back into disk if it went outside
        r = np.linalg.norm(pts_new[i])
        if r > 1.0:
            pts_new[i] = pts_new[i] / r * 0.999

    return pts_new

def _construct():
    t_start = time.time()
    
    # Best result found so far
    best_pts = None
    best_score = -1.0
    
    # Parameters
    # Start with a small target radius and increase it.
    # Initial radius estimate ~ 0.12
    r_start = 0.10
    r_end = 0.13
    steps = 500
    dt = 0.05
    k_rep = 10.0
    k_wall = 50.0
    
    # Initial configuration: Hexagonal lattice or Random
    def hex_init():
        pts = []
        # Generate a hexagonal grid
        spacing = 0.2
        for i in range(-10, 10):
            for j in range(-10, 10):
                x = i * spacing
                y = j * spacing * np.sqrt(3)
                if i % 2 != 0:
                    y += spacing * np.sqrt(3) / 2
                if np.hypot(x, y) < 0.8:
                    pts.append([x, y])
        pts = np.array(pts)
        # Randomly select N points
        if len(pts) > N:
            indices = np.random.choice(len(pts), N, replace=False)
            pts = pts[indices]
        return pts
    
    def random_init():
        pts = np.random.uniform(-0.8, 0.8, size=(N, 2))
        # Push inside
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 0.9
        pts[mask] = pts[mask] / radii[mask][:, None] * 0.9
        return pts

    # Run multiple restarts if time permits
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - t_start > DEADLINE - 10:
            break
            
        # Choose initial config
        if restart % 2 == 0:
            pts = hex_init()
        else:
            pts = random_init()
            
        # Ensure we have exactly N points
        if len(pts) < N:
            # Pad with random points
            extra = np.random.uniform(-0.5, 0.5, size=(N - len(pts), 2))
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:N]
        
        # MD Simulation with Annealing/Inflation
        # We will run the simulation for a fixed number of steps,
        # increasing target_r linearly.
        
        current_r = r_start
        # Determine how many steps to run for this restart
        # We want to finish within the time budget.
        # Let's allocate 10 seconds per restart.
        time_per_restart = 10.0
        
        # Estimate steps per second
        # We can't easily estimate without running, so we'll use a fixed number
        # and check time periodically.
        total_steps = 2000
        
        for step in range(total_steps):
            if time.time() - t_start > DEADLINE - 5:
                break
                
            # Animate target radius
            # Use a slow ramp-up
            progress = step / total_steps
            # Ease-in-out or linear? Linear is fine.
            current_r = r_start + (r_end - r_start) * progress
            
            # Perform MD step
            pts = _md_step(pts, current_r, dt, k_rep, k_wall)
            
            # Occasionally check score
            if step % 100 == 0:
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
        # Final check
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # If we didn't find anything (shouldn't happen), fallback
    if best_pts is None:
        best_pts = random_init()
        best_score = _compute_score(best_pts)

    return best_pts

def construct():
    return _construct()
