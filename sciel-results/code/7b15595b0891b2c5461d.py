import numpy as np
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Initial configuration: 11 hexagons from lattice
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
        
    ring2_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]
    for angle in ring2_angles:
        centers.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
        
    positions = [[x, y, 0.0] for x, y in centers]
    
    def calc_S_and_penalty(positions, container_rot):
        normals = []
        for k in range(6):
            angle = container_rot + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
            
        max_proj = 0.0
        for x, y, ang in positions:
            for k in range(6):
                v_angle = ang + k * np.pi / 3
                vx = x + np.cos(v_angle)
                vy = y + np.sin(v_angle)
                for nx, ny in normals:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        S = (2.0 / np.sqrt(3)) * max_proj
        
        penalty = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3) - 1e-9:
                    penalty += (np.sqrt(3) - dist) ** 2 * 1000
            
        return S, penalty

    current_positions = [p[:] for p in positions]
    current_rot = 0.0
    current_S, current_pen = calc_S_and_penalty(current_positions, current_rot)
    
    best_positions = [p[:] for p in current_positions]
    best_rot = current_rot
    best_S = current_S if current_pen < 1e-6 else float('inf')
    
    T = 0.5
    cooling = 0.99
    min_T = 1e-5
    iterations = 3000
    
    for i in range(iterations):
        if T < min_T:
            break
            
        new_positions = [p[:] for p in current_positions]
        new_rot = current_rot
        
        if random.random() < 0.8:
            idx = random.randint(0, N-1)
            new_positions[idx][0] += random.gauss(0, 0.1)
            new_positions[idx][1] += random.gauss(0, 0.1)
            new_positions[idx][2] += random.gauss(0, 0.1)
        else:
            new_rot += random.gauss(0, 0.05)
            
        new_S, new_pen = calc_S_and_penalty(new_positions, new_rot)
        new_cost = new_S + new_pen
        current_cost = current_S + current_pen
        
        if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
            current_positions = new_positions
            current_rot = new_rot
            current_S = new_S
            current_pen = new_pen
            
            if current_pen < 1e-6 and current_S < best_S:
                best_S = current_S
                best_positions = [p[:] for p in current_positions]
                best_rot = current_rot
                
        T *= cooling
        
    if best_S == float('inf'):
        # Fallback to initial
        best_S = current_S
        best_positions = current_positions
        best_rot = current_rot

    return best_positions, (0.0, 0.0), best_S, best_rot
# EVOLVE-BLOCK-END
