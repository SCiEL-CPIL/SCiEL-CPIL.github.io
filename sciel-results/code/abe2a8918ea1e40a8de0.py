import numpy as np
import time
import random

N = 11

V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2.0]
])

# Precompute triangle indices
tri_indices = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))

# For each point, which triangles does it belong to?
point_tris = [[] for _ in range(N)]
for t_idx, (i, j, k) in enumerate(tri_indices):
    point_tris[i].append(t_idx)
    point_tris[j].append(t_idx)
    point_tris[k].append(t_idx)

def random_point_in_triangle(vertices=V):
    r1 = random.random()
    r2 = random.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    p = r1 * vertices[0] + r2 * vertices[1] + (1 - r1 - r2) * vertices[2]
    return p

def is_inside_triangle(p, vertices=V, eps=1e-10):
    def area(a, b, c):
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
    a_total = area(vertices[0], vertices[1], vertices[2])
    a1 = area(p, vertices[1], vertices[2])
    a2 = area(vertices[0], p, vertices[2])
    a3 = area(vertices[0], vertices[1], p)
    return abs(a_total - (a1 + a2 + a3)) < eps and a1 > -eps and a2 > -eps and a3 > -eps

def calc_areas(points):
    areas = np.zeros(len(tri_indices))
    for t_idx, (i, j, k) in enumerate(tri_indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        areas[t_idx] = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    return areas

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = 0.0
    
    # Initial heuristic: try a few random starts
    # Also try a "grid" start?
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize points
        points = np.array([random_point_in_triangle() for _ in range(N)])
        areas = calc_areas(points)
        current_score = np.min(areas)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = 0.005
        T_end = 1e-7
        steps = 10000
        cooling_rate = (T_end / T) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            idx = random.randint(0, N-1)
            old_point = points[idx].copy()
            
            # Move
            # 10% large jump, 90% small step
            if random.random() < 0.1:
                new_point = random_point_in_triangle()
            else:
                # Random direction and magnitude
                angle = random.random() * 2 * np.pi
                dist = random.random() * 0.05
                new_point = old_point + np.array([np.cos(angle), np.sin(angle)]) * dist
                if not is_inside_triangle(new_point):
                    new_point = random_point_in_triangle()
            
            points[idx] = new_point
            
            # Update areas for triangles involving idx
            # Triangles not involving idx remain unchanged
            for t_idx in point_tris[idx]:
                i, j, k = tri_indices[t_idx]
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                areas[t_idx] = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            
            new_score = np.min(areas)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                points[idx] = old_point
                # Restore areas? 
                # Since we only modified areas for triangles involving idx, 
                # and we are reverting points[idx], we must restore those areas.
                # It's faster to just recalculate them from the old point? 
                # Or store the old areas?
                # Storing old areas for 45 triangles is cheap.
                # Let's just recalculate to be safe, or store.
                # Actually, since we have the old_point, we can recalculate.
                for t_idx in point_tris[idx]:
                    i, j, k = tri_indices[t_idx]
                    # One of i,j,k is idx.
                    # We need to put old_point back in the calculation.
                    # Easier: just recalculate using current points array which has old_point restored.
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    areas[t_idx] = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

            T *= cooling_rate
            
    return best_points
