import numpy as np

def construct():
    N = 300
    
    # 1. Initialization: Hexagonal Lattice
    # Provides a good starting configuration to avoid random chaos.
    s = 0.1
    k = int(1.0 / s) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 0.9:
                rows.append((x, y))
    points = np.array(rows)
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
                
    # 2. Simulated Annealing with Inflation
    r_start = 0.02
    r_end = 0.054
    num_inflation_steps = 20
    
    total_iterations = 200000 
    iters_per_step = total_iterations // num_inflation_steps
    
    radii = np.linspace(r_start, r_end, num_inflation_steps)
    
    for r in radii:
        T_start = 0.05
        T_end = 0.0005
        move_start = 0.1
        move_end = 0.005
        
        for it in range(iters_per_step):
            i = np.random.randint(N)
            
            progress = it / iters_per_step
            T = T_start * (T_end / T_start) ** progress
            move_scale = move_start * (move_end / move_start) ** progress
            
            r_move = move_scale * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            delta = np.array([r_move * np.cos(theta), r_move * np.sin(theta)])
            
            new_pos = points[i] + delta
            
            dists_old = np.linalg.norm(points - points[i], axis=1)
            dists_new = np.linalg.norm(points - new_pos, axis=1)
            
            overlap_old = np.sum(np.maximum(0, 2*r - dists_old)**2)
            overlap_new = np.sum(np.maximum(0, 2*r - dists_new)**2)
            
            bound_old = max(0, np.linalg.norm(points[i]) - (1-r))**2
            bound_new = max(0, np.linalg.norm(new_pos) - (1-r))**2
            
            delta_E = (overlap_new + bound_new) - (overlap_old + bound_old)
            
            if delta_E < 0:
                points[i] = new_pos
            else:
                if delta_E / T < 700: 
                    if np.random.rand() < np.exp(-delta_E / T):
                        points[i] = new_pos

    return points
