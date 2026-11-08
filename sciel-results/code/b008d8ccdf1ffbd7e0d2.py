"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(x, y, angle_deg):
        angle = np.radians(angle_deg)
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        verts = np.array([[np.cos(base_angles[i] + angle), np.sin(base_angles[i] + angle)] for i in range(6)])
        return verts + np.array([x, y])
    
    def compute_outer_side(positions, outer_center, outer_angle_deg):
        outer_angle = np.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            ang = outer_angle + k * np.pi / 3
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        
        max_proj = 0.0
        for pos in positions:
            x, y, ang_deg = pos
            verts = hex_vertices(x, y, ang_deg)
            for v in verts:
                rel = v - outer_center
                for n in normals:
                    proj = np.dot(rel, n)
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / np.sqrt(3.0)
    
    dx = np.sqrt(3)
    dy = 1.5
    
    arrangements = []
    
    # Arrangement 1: single row of 11
    arr1 = [[i * dx - 5 * dx, 0.0, 0.0] for i in range(11)]
    arrangements.append(arr1)
    
    # Arrangement 2: 3-4-3-1
    arr2 = []
    for i in range(3):
        arr2.append([i * dx - dx, 0.0, 0.0])
    for i in range(4):
        arr2.append([i * dx - dx * 1.5 + dx/2, dy, 0.0])
    for i in range(3):
        arr2.append([i * dx - dx, 2*dy, 0.0])
    arr2.append([0.0, 3*dy, 0.0])
    arrangements.append(arr2)
    
    # Arrangement 3: 4-3-4
    arr3 = []
    for i in range(4):
        arr3.append([i * dx - 1.5 * dx, 0.0, 0.0])
    for i in range(3):
        arr3.append([i * dx - dx + dx/2, dy, 0.0])
    for i in range(4):
        arr3.append([i * dx - 1.5 * dx, 2*dy, 0.0])
    arrangements.append(arr3)
    
    # Arrangement 4: 5-6
    arr4 = []
    for i in range(5):
        arr4.append([i * dx - 2 * dx, 0.0, 0.0])
    for i in range(6):
        arr4.append([i * dx - 2.5 * dx + dx/2, dy, 0.0])
    arrangements.append(arr4)
    
    # Arrangement 5: 2-3-2-2-2
    arr5 = []
    for i in range(2):
        arr5.append([i * dx - dx/2, 0.0, 0.0])
    for i in range(3):
        arr5.append([i * dx - dx + dx/2, dy, 0.0])
    for i in range(2):
        arr5.append([i * dx - dx/2, 2*dy, 0.0])
    for i in range(2):
        arr5.append([i * dx - dx/2, 3*dy, 0.0])
    for i in range(2):
        arr5.append([i * dx - dx/2, 4*dy, 0.0])
    arrangements.append(arr5)
    
    # Arrangement 6: 1-2-3-2-1-2 (hexagonal cluster)
    arr6 = []
    arr6.append([0.0, 0.0, 0.0])
    for i in range(2):
        arr6.append([i * dx - dx/2, dy, 0.0])
    for i in range(3):
        arr6.append([i * dx - dx, 2*dy, 0.0])
    for i in range(2):
        arr6.append([i * dx - dx/2, 3*dy, 0.0])
    arr6.append([0.0, 4*dy, 0.0])
    arr6.append([dx/2, 5*dy, 0.0])
    arrangements.append(arr6)
    
    # Find best arrangement
    best_side = np.inf
    best_arr = arrangements[0]
    best_outer_angle = 0.0
    
    for arr in arrangements:
        for oa in [0, 30, 60, 90, 120, 150]:
            xs = [p[0] for p in arr]
            ys = [p[1] for p in arr]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            centered_arr = [[p[0] - cx, p[1] - cy, p[2]] for p in arr]
            
            side = compute_outer_side(centered_arr, np.array([0.0, 0.0]), oa)
            if side < best_side:
                best_side = side
                best_arr = centered_arr
                best_outer_angle = oa
    
    final_side = best_side * 1.005
    
    inner = [[p[0], p[1], p[2]] for p in best_arr]
    return inner, (0.0, 0.0), final_side, best_outer_angle
# EVOLVE-BLOCK-END
