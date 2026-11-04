import numpy as np

N = 26

def construct():
    def compute_radii(positions):
        xs = positions[:N]
        ys = positions[N:]
        rs = np.zeros(N)
        
        # Wall constraints
        for i in range(N):
            rs[i] = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            
        # Neighbor constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                val = dist / 2.0
                if val < rs[i]:
                    rs[i] = val
                if val < rs[j]:
                    rs[j] = val
        return rs

    def objective(positions):
        rs = compute_radii(positions)
        return -np.sum(rs)

    # Initialization: Hexagonal Grid
    def hex_init():
        points = []
        s = 0.25
        h = s * np.sqrt(3) / 2
        for i in range(10):
            for j in range(10):
                x = j * s + (i % 2) * s / 2
                y = i * h
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    points.append((x, y))
        
        np.random.shuffle(points)
        if len(points) < N:
            return np.random.rand(2*N)
        selected = points[:N]
        pos = np.zeros(2*N)
        for i, (x, y) in enumerate(selected):
            pos[i] = x
            pos[N+i] = y
        return pos

    best_pos = None
    best_score = -np.inf
    
    # Multiple restarts with Simulated Annealing
    for restart in range(5):
        if restart == 0:
            current_pos = hex_init()
        else:
            current_pos = np.random.rand(2*N)
            current_pos = np.clip(current_pos, 0.05, 0.95)
            
        current_obj = objective(current_pos)
        
        T = 0.1
        T_min = 0.001
        alpha = 0.99
        steps = 3000
        
        for step in range(steps):
            # Perturb
            new_pos = current_pos.copy()
            idx = np.random.randint(2*N)
            # Scale perturbation by T
            delta = np.random.normal(0, T)
            new_pos[idx] += delta
            new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)
            
            new_obj = objective(new_pos)
            
            delta_obj = new_obj - current_obj
            
            if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
                current_pos = new_pos
                current_obj = new_obj
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        # Evaluate final
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
