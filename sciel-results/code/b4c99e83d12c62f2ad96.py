import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 100.0
    
    N = 11
    
    # Helper functions
    def get_vertices(x, y, theta_deg):
        theta = np.radians(theta_deg)
        verts = np.zeros((6, 2))
        for k in range(6):
            angle = theta + k * np.pi / 3
            verts[k, 0] = x + np.cos(angle)
            verts[k, 1] = y + np.sin(angle)
        return verts

    def project(verts, axis):
        proj = np.dot(verts, axis)
        return np.min(proj), np.max(proj)

    def overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(3):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                norm = np.array([-edge[1], edge[0]])
                norm = norm / np.linalg.norm(norm)
                axes.append(norm)
        
        for axis in axes:
            min1, max1 = project(v1, axis)
            min2, max2 = project(v2, axis)
            if max1 < min2 or max2 < min1:
                return False
        return True

    def calc_cost(config):
        verts_list = [get_vertices(x, y, t) for x, y, t in config]
        
        # Check overlap
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if overlap(verts_list[i], verts_list[j]):
                    return float('inf')
        
        # Calc min side
        max_s = 0
        for verts in verts_list:
            for v in verts:
                x, y = v
                s1 = abs(x)
                s2 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
                s3 = abs(-np.sqrt(3)*x + y) / np.sqrt(3)
                s = max(s1, s2, s3)
                if s > max_s:
                    max_s = s
        return max_s

    # Initial configuration
    # 7 hexagon cluster
    config = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = k * 60
        rad = np.radians(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        config.append((x, y, 0.0))
    
    # 4 more hexagons far away
    for k in range(4):
        angle = (k * 90 + 45)
        rad = np.radians(angle)
        x = 10.0 * np.cos(rad)
        y = 10.0 * np.sin(rad)
        config.append((x, y, 0.0))
        
    best_config = [list(c) for c in config]
    best_cost = calc_cost(best_config)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    step_size = 1.0
    
    while time.time() - start_time < deadline:
        new_config = [list(c) for c in best_config]
        idx = random.randint(0, N-1)
        
        # Perturb
        new_config[idx][0] += random.gauss(0, step_size)
        new_config[idx][1] += random.gauss(0, step_size)
        new_config[idx][2] += random.gauss(0, 5.0)
        
        new_cost = calc_cost(new_config)
        
        if new_cost < best_cost:
            best_config = new_config
            best_cost = new_cost
        else:
            delta = new_cost - best_cost
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            if random.random() < prob:
                best_config = new_config
                best_cost = new_cost
        
        T = max(T_min, T * 0.99)
        step_size = max(0.01, step_size * 0.99)
        
        # Reset occasionally?
        if T == T_min:
             T = 0.5
             step_size = 0.5

    # Return best
    inner = [[x, y, t] for x, y, t in best_config]
    return inner, (0.0, 0.0), best_cost, 0.0
