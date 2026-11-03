import numpy as np
import time

N = 12
DEADLINE = 145.0  # Safety margin

def construct():
    start_time = time.time()
    
    # Outer hexagon normals (pointy top, circumradius S)
    # Normals at 90, 150, 210, 270, 330, 30 degrees
    angles_outer = np.deg2rad(np.array([90, 150, 210, 270, 330, 30]))
    n_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner hexagon vertices relative to center (side 1)
    angles_inner_rel = np.deg2rad(np.arange(0, 360, 60) + 30)
    vertices_rel = np.column_stack([np.cos(angles_inner_rel), np.sin(angles_inner_rel)])
    
    def get_lattice_positions():
        positions = []
        positions.append([0.0, 0.0])
        for i in range(6):
            ang = np.deg2rad(i * 60)
            positions.append([np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang)])
        for i in range(5):
            ang = np.deg2rad(i * 60 + 30)
            positions.append([2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang)])
        return np.array(positions)
        
    def calculate_cost(x, y, theta):
        max_proj = 0.0
        overlap_pen = 0.0
        
        vertices = np.zeros((N, 6, 2))
        for i in range(N):
            c, s = np.cos(theta[i]), np.sin(theta[i])
            R = np.array([[c, -s], [s, c]])
            verts = (vertices_rel @ R.T) + np.array([x[i], y[i]])
            vertices[i] = verts
            
        for i in range(N):
            dots = np.dot(vertices[i], n_outer.T)
            current_max = np.max(dots)
            if current_max > max_proj:
                max_proj = current_max
                
        S_val = max_proj / (np.sqrt(3) / 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                vi = vertices[i]
                vj = vertices[j]
                
                axes = []
                for k in range(6):
                    e = vi[(k+1)%6] - vi[k]
                    n = np.array([-e[1], e[0]])
                    axes.append(n)
                for k in range(6):
                    e = vj[(k+1)%6] - vj[k]
                    n = np.array([-e[1], e[0]])
                    axes.append(n)
                    
                is_separated = False
                min_overlap_depth = float('inf')
                
                for ax in axes:
                    norm = np.linalg.norm(ax)
                    if norm < 1e-9: continue
                    ax = ax / norm
                    
                    proj_i = np.dot(vi, ax)
                    proj_j = np.dot(vj, ax)
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    if max_i < min_j or max_j < min_i:
                        is_separated = True
                        break
                    else:
                        depth = min(max_i, max_j) - max(min_i, min_j)
                        if depth < min_overlap_depth:
                            min_overlap_depth = depth
                
                if not is_separated:
                    overlap_pen += min_overlap_depth * 1000.0
                    
        return S_val + overlap_pen

    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    steps = 100000
    
    # Initial state
    pos = get_lattice_positions()
    x = pos[:, 0]
    y = pos[:, 1]
    theta = np.zeros(N)
    
    current_cost = calculate_cost(x, y, theta)
    best_x, best_y, best_theta = x.copy(), y.copy(), theta.copy()
    best_cost = current_cost
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
            
        i = np.random.randint(N)
        
        dx = np.random.uniform(-0.2, 0.2)
        dy = np.random.uniform(-0.2, 0.2)
        dth = np.random.uniform(-0.2, 0.2)
        
        new_x = x.copy()
        new_y = y.copy()
        new_theta = theta.copy()
        
        new_x[i] += dx
        new_y[i] += dy
        new_theta[i] += dth
        
        new_cost = calculate_cost(new_x, new_y, new_theta)
        
        if new_cost < current_cost:
            x, y, theta = new_x, new_y, new_theta
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x, best_y, best_theta = x.copy(), y.copy(), theta.copy()
        else:
            delta = new_cost - current_cost
            if np.random.rand() < np.exp(-delta / T):
                x, y, theta = new_x, new_y, new_theta
                current_cost = new_cost
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final S calculation
    max_proj = 0.0
    for i in range(N):
        c, s = np.cos(best_theta[i]), np.sin(best_theta[i])
        R = np.array([[c, -s], [s, c]])
        verts = (vertices_rel @ R.T) + np.array([best_x[i], best_y[i]])
        dots = np.dot(verts, n_outer.T)
        current_max = np.max(dots)
        if current_max > max_proj:
            max_proj = current_max
            
    final_S = max_proj / (np.sqrt(3) / 2)
    
    inner = []
    for i in range(N):
        inner.append([float(best_x[i]), float(best_y[i]), float(np.rad2deg(best_theta[i]))])
        
    return inner, (0.0, 0.0), float(final_S), 0.0

