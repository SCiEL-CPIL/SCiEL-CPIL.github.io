import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate all 19 centers of the 3-side hexagon cluster
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
    
    # Sort by distance from origin and take 12 most central
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:12]
    
    # Initial positions
    x = np.array([c[0] for c in selected_centers], dtype=float)
    y = np.array([c[1] for c in selected_centers], dtype=float)
    angles = np.zeros(N)
    
    # Edge normals for outer hexagon (angle 0)
    edge_angles = [math.pi/6 + k * math.pi/3 for k in range(6)]
    edge_vectors = np.array([[math.cos(ea), math.sin(ea)] for ea in edge_angles])
    
    # Vertex angles for unit hexagon
    vertex_angles = np.array([k * math.pi/3 for k in range(6)])
    
    def min_outer_L(x, y, angles):
        """Compute minimum outer hexagon side length."""
        max_proj = 0.0
        for i in range(N):
            cx, cy = x[i], y[i]
            theta = angles[i]
            # Vertices of this hexagon
            for va in vertex_angles:
                vx = math.cos(va + theta)
                vy = math.sin(va + theta)
                vxg = cx + vx
                vyg = cy + vy
                # Project onto each edge normal
                for ev in edge_vectors:
                    proj = vxg * ev[0] + vyg * ev[1]
                    if proj > max_proj:
                        max_proj = proj
        return max_proj / (math.sqrt(3) / 2)
    
    def check_no_overlap(x, y):
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx*dx + dy*dy < (math.sqrt(3) - 1e-7)**2:
                    return False
        return True
    
    best_L = min_outer_L(x, y, angles)
    best_x = x.copy()
    best_y = y.copy()
    best_angles = angles.copy()
    
    # Local optimization with small steps
    np.random.seed(42)
    for it in range(2000):
        if time.time() > deadline - 15:
            break
        
        # Try moving one hexagon
        i = np.random.randint(N)
        dx = np.random.randn() * 0.05
        dy = np.random.randn() * 0.05
        da = np.random.randn() * 0.01
        
        new_x = x.copy()
        new_y = y.copy()
        new_angles = angles.copy()
        new_x[i] += dx
        new_y[i] += dy
        new_angles[i] += da
        
        if not check_no_overlap(new_x, new_y):
            continue
        
        L = min_outer_L(new_x, new_y, new_angles)
        if L < best_L:
            best_L = L
            best_x = new_x.copy()
            best_y = new_y.copy()
            best_angles = new_angles.copy()
            x = new_x.copy()
            y = new_y.copy()
            angles = new_angles.copy()
    
    inner = []
    for i in range(N):
        inner.append([best_x[i], best_y[i], best_angles[i] * 180.0 / math.pi])
    
    outer_center = (0.0, 0.0)
    side = best_L + 0.005
    angle = 0.0
    
    return inner, outer_center, side, angle
