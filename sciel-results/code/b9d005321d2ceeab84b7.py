import numpy as np

N = 26

def construct():
    def compute_radii(positions):
        xs = positions[:N]
        ys = positions[N:]
        
        # Wall constraints
        r_walls = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
        
        # Neighbor constraints
        # Compute distance matrix
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Min distance to any neighbor / 2
        r_neighbors = np.min(dist, axis=1) / 2.0
        
        # Final radius is min of wall and neighbor constraints
        rs = np.minimum(r_walls, r_neighbors)
        return rs

    def disk_init():
        # Generate hexagonal grid points in unit disk
        points = []
        s = 0.2
        h = s * np.sqrt(3) / 2
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = j * s + (i % 2) * s / 2
                y = i * h
                if x**2 + y**2 <= 1.0:
                    points.append((x, y))
        
        np.random.shuffle(points)
        if len(points) < N:
            # Fallback to random
            return np.random.rand(2*N) * 0.8 + 0.1
        
        selected = points[:N]
        pos = np.zeros(2*N)
        for i, (x, y) in enumerate(selected):
            # Map disk coordinates (approx -1 to 1) to square [0, 1]
            pos[i] = 0.5 + 0.5 * x
            pos[N+i] = 0.5 + 0.5 * y
        return pos

    best_pos = None
    best_score = -np.inf
    
    # Force-Directed Optimization with Multiple Restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if restart == 0:
            current_pos = disk_init()
        else:
            # Random initialization
            current_pos = np.random.rand(2*N)
            current_pos = np.clip(current_pos, 0.1, 0.9)
        
        # Force-Directed Simulation
        step_size = 0.05
        steps = 2000
        
        for step in range(steps):
            xs = current_pos[:N].copy()
            ys = current_pos[N:].copy()
            rs = compute_radii(current_pos)
            
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Neighbor repulsion forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    
                    if dist < min_dist:
                        if dist < 1e-6:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-6
                        # Repulsive force proportional to overlap
                        force = (min_dist - dist) * 10.0
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
            
            # Wall attraction/constraint forces
            for i in range(N):
                if xs[i] < rs[i]:
                    fx[i] += (rs[i] - xs[i]) * 10.0
                if xs[i] > 1 - rs[i]:
                    fx[i] -= (xs[i] - (1 - rs[i])) * 10.0
                if ys[i] < rs[i]:
                    fy[i] += (rs[i] - ys[i]) * 10.0
                if ys[i] > 1 - rs[i]:
                    fy[i] -= (ys[i] - (1 - rs[i])) * 10.0
            
            # Update positions
            current_pos[:N] += fx * step_size
            current_pos[N:] += fy * step_size
            
            # Clip to valid range
            current_pos = np.clip(current_pos, 0.0, 1.0)
            
            # Decay step size
            step_size *= 0.999

        rs = compute_radii(current_pos)
        score = np.sum(rs)
        
        if score > best_score:
            best_score = score
            best_pos = current_pos.copy()

    xs = best_pos[:N]
    ys = best_pos[N:]
    rs = compute_radii(best_pos)
    
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    
    return result
