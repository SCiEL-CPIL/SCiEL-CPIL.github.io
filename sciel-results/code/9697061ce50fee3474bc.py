import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inradius = np.sqrt(3) / 2
    sqrt3 = np.sqrt(3)
    
    def support_hex(angle):
        """Support function of unit regular hexagon (circumradius 1) in direction with given angle."""
        # Vertices at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        return max(np.cos(angle - k * np.pi / 3) for k in range(6))
    
    def compute_outer_side(positions, theta, cx, cy):
        """Compute minimum side length of outer regular hexagon."""
        angles = theta + np.arange(6) * np.pi / 3
        max_val = 0.0
        for pos in positions:
            rel = pos - np.array([cx, cy])
            for k in range(6):
                a = angles[k]
                h_inner = support_hex(a)
                val = rel[0] * np.cos(a) + rel[1] * np.sin(a) + h_inner
                if val > max_val:
                    max_val = val
        return (2.0 / sqrt3) * max_val
    
    def optimize_outer(positions):
        """Optimize outer hexagon center and orientation for given inner positions."""
        def objective(params):
            if time.time() > deadline:
                return 1000.0
            theta, cx, cy = params
            return compute_outer_side(positions, theta, cx, cy)
        
        best_S = float('inf')
        best_params = None
        
        for theta_init in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/12, 2*np.pi/3, np.pi/2, np.pi/5]:
            if time.time() > deadline:
                break
            res = minimize(objective, [theta_init, 0.0, 0.0], method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        
        if best_params is None:
            best_params = [0.0, 0.0, 0.0]
            best_S = compute_outer_side(positions, *best_params)
        
        return best_S, best_params
    
    # Generate candidate arrangements
    arrangements = []
    
    # Arrangement 1: 3 rows of 4 in hex lattice
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2, 1.5])
    pos1 = []
    for i in range(3):
        for j in range(4):
            pos1.append(i * a2 + j * a1)
    arrangements.append(np.array(pos1))
    
    # Arrangement 2: 4 rows of 3 in hex lattice
    pos2 = []
    for i in range(4):
        for j in range(3):
            pos2.append(i * a2 + j * a1)
    arrangements.append(np.array(pos2))
    
    # Arrangement 3: Hexagonal cluster (center + 5 + 6)
    # Center at origin, ring 1: 6 hexagons, ring 2: 6 more
    pos3 = [np.array([0.0, 0.0])]
    for k in range(6):
        angle = k * np.pi / 3
        pos3.append(sqrt3 * np.array([np.cos(angle), np.sin(angle)]))
    # Add 5 more from ring 2
    for k in range(5):
        angle = k * np.pi / 3 + np.pi / 6
        pos3.append(2 * sqrt3 * np.array([np.cos(angle), np.sin(angle)]))
    arrangements.append(np.array(pos3))
    
    # Arrangement 4: Two rows of 6
    pos4 = []
    for j in range(6):
        pos4.append(j * a1)
    for j in range(6):
        pos4.append(a2 + j * a1)
    arrangements.append(np.array(pos4))
    
    # Arrangement 5: 2 rows of 4 + 1 row of 4 (staggered)
    pos5 = []
    for j in range(4):
        pos5.append(j * a1)
    for j in range(4):
        pos5.append(a2 + (j + 0.5) * a1)  # staggered
    for j in range(4):
        pos5.append(2 * a2 + j * a1)
    arrangements.append(np.array(pos5))
    
    # Evaluate all arrangements
    best_S = float('inf')
    best_arrangement = None
    best_params = None
    
    for idx, arr in enumerate(arrangements):
        if time.time() > deadline:
            break
        # Center the arrangement
        center = np.mean(arr, axis=0)
        arr_centered = arr - center
        
        S, params = optimize_outer(arr_centered)
        if S < best_S:
            best_S = S
            best_arrangement = arr_centered
            best_params = params
    
    if best_arrangement is None:
        best_arrangement = arrangements[0] - np.mean(arrangements[0], axis=0)
        best_params = [0.0, 0.0, 0.0]
        best_S = compute_outer_side(best_arrangement, *best_params)
    
    theta_opt, cx_opt, cy_opt = best_params
    
    inner = []
    for pos in best_arrangement:
        inner.append([float(pos[0]), float(pos[1]), 0.0])
    
    outer_center = [float(cx_opt), float(cy_opt)]
    outer_side = float(best_S)
    outer_angle = float(np.degrees(theta_opt))
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
