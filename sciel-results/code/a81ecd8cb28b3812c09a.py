import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial position: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    def compute_radius(p):
        # p is Nx3
        # Check validity
        if not np.all(np.isfinite(p)):
            return -1.0
        if np.any(p < -0.5) or np.any(p > 0.5):
            return -1.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(p)
        min_wall = np.min(wall_dists)
        
        # Pairwise distance
        # For N=50, calculating all pairs is cheap
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                dists[i, j] = d
                dists[j, i] = d
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r
    
    best_r = compute_radius(pos)
    best_pos = pos.copy()
    
    # Local Search: Push-Apart with Annealing
    # We will run this for a few restarts if time permits
    
    def local_search(initial_pos, time_limit):
        p = initial_pos.copy()
        current_r = compute_radius(p)
        best_r_local = current_r
        best_p_local = p.copy()
        
        step_size = 0.05
        patience = 0
        max_patience = 200
        
        while time.time() < time_limit:
            # Compute forces
            forces = np.zeros_like(p)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = p[i] - p[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 1e-6:
                        # Random kick if coincident
                        d_vec = np.random.randn(3) * 1e-6
                        dist = np.linalg.norm(d_vec)
                    
                    # Only repel if closer than current best diameter (or slightly more)
                    # Using best_r_local * 2 as threshold helps focus on the bottleneck
                    threshold = best_r_local * 2.0 * 1.1 
                    
                    if dist < threshold:
                        # Inverse square force
                        f_mag = 1.0 / (dist * dist + 1e-6)
                        forces[i] += (d_vec / dist) * f_mag
                        forces[j] -= (d_vec / dist) * f_mag
            
            # Wall repulsion
            for i in range(N):
                for k in range(3):
                    if p[i][k] < -0.5 + best_r_local:
                        forces[i][k] += 1.0 / (best_r_local + 1e-6)
                    elif p[i][k] > 0.5 - best_r_local:
                        forces[i][k] -= 1.0 / (best_r_local + 1e-6)
            
            # Update positions
            # Normalize forces to prevent huge jumps
            force_mags = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_mags)
            if max_force > 0:
                forces = forces / max_force
            
            p += step_size * forces
            
            # Clip to cube
            p = np.clip(p, -0.5, 0.5)
            
            # Evaluate
            new_r = compute_radius(p)
            
            if new_r > current_r:
                current_r = new_r
                patience = 0
                # Decay step size slightly on improvement
                step_size *= 0.95
                step_size = max(step_size, 0.001)
                
                if current_r > best_r_local:
                    best_r_local = current_r
                    best_p_local = p.copy()
            else:
                patience += 1
                # If stuck, increase step size to escape
                if patience % 10 == 0:
                    step_size = min(0.1, step_size * 1.2)
                
                if patience > max_patience:
                    break
            
            # Optional: Add small random noise to escape local minima
            if np.random.rand() < 0.1:
                p += np.random.randn(N, 3) * 0.005
                p = np.clip(p, -0.5, 0.5)
                
        return best_p_local, best_r_local

    # Run local search from grid
    time_left = deadline - time.time()
    if time_left > 5:
        best_pos, best_r = local_search(pos, time.time() + time_left * 0.5)
        
    # If time remains, try a random restart
    time_left = deadline - time.time()
    if time_left > 10:
        rng = np.random.RandomState(42)
        # Random initialization within bounds
        rand_pos = rng.uniform(-0.4, 0.4, size=(N, 3))
        rand_pos, rand_r = local_search(rand_pos, time.time() + time_left * 0.8)
        
        if rand_r > best_r:
            best_r = rand_r
            best_pos = rand_pos

    # Final check
    final_r = compute_radius(best_pos)
    if final_r < 0:
        # Fallback to grid
        best_pos = grid[:N].copy()
        final_r = compute_radius(best_pos)
        
    return best_pos
