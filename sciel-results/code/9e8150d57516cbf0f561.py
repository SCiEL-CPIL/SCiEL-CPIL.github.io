import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s wall-clock limit, 5s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - time.time()
    
    # Initial configuration: random placement
    def random_init(seed):
        rng = np.random.RandomState(seed)
        pos = rng.rand(N, 2) * 0.8 + 0.1  # Start in [0.1, 0.9]^2
        radii = np.full(N, 0.04)  # Small initial radius
        return pos, radii
    
    def check_validity(pos, radii):
        # Check all circles are within [0,1]^2 and non-overlapping
        # Boundary check
        if np.any(pos[:, 0] < radii) or np.any(pos[:, 1] < radii):
            return False
        if np.any(pos[:, 0] + radii > 1.0) or np.any(pos[:, 1] + radii > 1.0):
            return False
        # Pairwise distance check
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-10:
                    return False
        return True
    
    def objective(pos, radii):
        """Negative of sum of radii (minimize)."""
        return -np.sum(radii)
    
    def lsd_step(pos, radii, lr, alpha, temperature, rng):
        """
        One step of a simplified local search:
        - Compute pairwise repulsion forces for overlapping circles
        - Apply gradient-like updates
        - Add small random perturbation (Langevin-like)
        - Update radii to maximize sum while maintaining feasibility
        """
        n = len(pos)
        forces = np.zeros_like(pos)
        
        # Pairwise interactions
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < 1e-10:
                    # Avoid division by zero
                    angle = rng.rand() * 2 * np.pi
                    dx, dy = np.cos(angle), np.sin(angle)
                    dist = 1e-10
                
                if dist < min_dist:
                    # Repulsive force
                    overlap = min_dist - dist
                    force_magnitude = alpha * overlap
                    fx = force_magnitude * dx / dist
                    fy = force_magnitude * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Boundary forces
        for i in range(n):
            # Left boundary
            if pos[i, 0] < radii[i]:
                forces[i, 0] += alpha * (radii[i] - pos[i, 0])
            # Right boundary
            if pos[i, 0] + radii[i] > 1.0:
                forces[i, 0] -= alpha * (pos[i, 0] + radii[i] - 1.0)
            # Bottom boundary
            if pos[i, 1] < radii[i]:
                forces[i, 1] += alpha * (radii[i] - pos[i, 1])
            # Top boundary
            if pos[i, 1] + radii[i] > 1.0:
                forces[i, 1] -= alpha * (pos[i, 1] + radii[i] - 1.0)
        
        # Update positions with Langevin dynamics
        noise = rng.randn(n, 2) * temperature
        pos_new = pos + lr * forces + noise
        
        # Update radii: try to grow each circle as much as possible
        radii_new = radii.copy()
        for i in range(n):
            # Max radius limited by boundary
            r_max = min(pos_new[i, 0], 1.0 - pos_new[i, 0], pos_new[i, 1], 1.0 - pos_new[i, 1])
            # Max radius limited by neighbors
            for j in range(n):
                if i != j:
                    dx = pos_new[i, 0] - pos_new[j, 0]
                    dy = pos_new[i, 1] - pos_new[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        r_max = min(r_max, (dist - radii_new[j]) * 0.99)
            r_max = max(0.001, r_max)
            radii_new[i] = min(r_max, radii[i] * 1.02 + 0.005)
        
        return pos_new, radii_new
    
    best_score = -np.inf
    best_pos, best_radii = None, None
    
    # Multiple restarts with cooling
    num_restarts = 30
    steps_per_restart = 2000
    
    for restart in range(num_restarts):
        if elapsed() > deadline - 10:
            break
        
        seed = 42 + restart * 1000
        pos, radii = random_init(seed)
        
        # Cooling schedule
        T0 = 0.01
        Tend = 0.0001
        lr0 = 0.005
        lr_end = 0.0001
        alpha = 5.0
        
        current_score = -np.inf
        
        for step in range(steps_per_restart):
            if elapsed() > deadline - 10:
                break
            
            # Linear cooling
            t_frac = step / steps_per_restart
            temperature = T0 * np.exp(-5 * t_frac) + Tend
            lr = lr0 * (1 - t_frac) + lr_end
            
            pos, radii = lsd_step(pos, radii, lr, alpha, temperature, np.random.RandomState(seed + step))
            
            # Project to valid region
            for i in range(N):
                radii[i] = max(0.001, min(radii[i], pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]))
                pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
                pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
            
            score = np.sum(radii)
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        if elapsed() > deadline - 15:
            break
    
    # Final validation and polish
    if best_pos is not None and best_radii is not None:
        # Try to grow radii further with a final pass
        for _ in range(50):
            improved = False
            for i in range(N):
                r_max = min(best_pos[i, 0], 1.0 - best_pos[i, 0], best_pos[i, 1], 1.0 - best_pos[i, 1])
                for j in range(N):
                    if i != j:
                        dx = best_pos[i, 0] - best_pos[j, 0]
                        dy = best_pos[i, 1] - best_pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - best_radii[j]) * 0.999)
                r_new = max(0.001, r_max)
                if r_new > best_radii[i] * 1.001:
                    best_radii[i] = r_new
                    improved = True
            if not improved:
                break
        
        # Ensure validity
        if check_validity(best_pos, best_radii):
            result = np.column_stack([best_pos, best_radii])
            return result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
