import numpy as np
import math
import time

def construct():
    N = 11
    start_time = time.time()
    deadline = 150.0
    
    d = math.sqrt(3)
    
    centers = [(0, 0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
    
    extra_outward = []
    for i in range(6):
        angle = i * math.pi / 3
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        extra_outward.append((x, y))
        
    extra_gap = []
    for i in range(6):
        angle = (i + 0.5) * math.pi / 3
        x = 1.5 * d * math.cos(angle)
        y = 1.5 * d * math.sin(angle)
        extra_gap.append((x, y))

    configs = []
    for start in range(6):
        extras = [extra_outward[(start + i) % 6] for i in range(4)]
        configs.append(centers + extras)
    for start in range(6):
        extras = [extra_gap[(start + i) % 6] for i in range(4)]
        configs.append(centers + extras)
        
    best_S = float('inf')
    best_inner = None
    best_phi = 0.0
    best_cfg = None
    
    for cfg in configs:
        all_verts = []
        for c in cfg:
            for i in range(6):
                alpha = i * math.pi / 3
                x = c[0] + math.cos(alpha)
                y = c[1] + math.sin(alpha)
                all_verts.append((x, y))
        
        verts_arr = np.array(all_verts)
        
        for phi_deg in np.linspace(0, 60, 200):
            if time.time() > deadline - 10.0:
                break
            phi = math.radians(phi_deg)
            max_dot = 0.0
            for k in range(6):
                n_angle = phi + math.pi/2 + k * math.pi/3
                n = np.array([math.cos(n_angle), math.sin(n_angle)])
                dots = verts_arr @ n
                max_dot = max(max_dot, np.max(dots))
            S = 2 * max_dot / math.sqrt(3)
            if S < best_S:
                best_S = S
                best_inner = [[c[0], c[1], 0.0] for c in cfg]
                best_phi = phi_deg
                best_cfg = cfg
                
        if time.time() > deadline - 10.0:
            break
            
    # Fine search
    if best_inner is not None:
        # Re-prepare verts for best_cfg
        all_verts = []
        for c in best_cfg:
            for i in range(6):
                alpha = i * math.pi / 3
                x = c[0] + math.cos(alpha)
                y = c[1] + math.sin(alpha)
                all_verts.append((x, y))
        verts_arr = np.array(all_verts)
        
        phi_range = np.linspace(best_phi - 5, best_phi + 5, 200)
        for phi_deg in phi_range:
            if time.time() > deadline - 2.0:
                break
            phi = math.radians(phi_deg)
            max_dot = 0.0
            for k in range(6):
                n_angle = phi + math.pi/2 + k * math.pi/3
                n = np.array([math.cos(n_angle), math.sin(n_angle)])
                dots = verts_arr @ n
                max_dot = max(max_dot, np.max(dots))
            S = 2 * max_dot / math.sqrt(3)
            if S < best_S:
                best_S = S
                best_phi = phi_deg
                
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    else:
        return best_inner, (0.0, 0.0), best_S, best_phi
