import numpy as np
import time
import random
import itertools

N = 11
H = np.sqrt(3) / 2
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, H])

def construct():
    # Pre-compute triangle indices for vectorized area calculation
    indices = np.array(list(itertools.combinations(range(N), 3)))
    
    def project_point_to_triangle(p):
        # p is (2,)
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
        d1 = sign(p, V1, V2)
        d2 = sign(p, V2, V3)
        d3 = sign(p, V3, V1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        if not (has_neg and has_pos):
            return p
            
        def project_to_segment(p, a, b):
            v = b - a
            w = p - a
            c1 = np.dot(w, v)
            if c1 <= 0:
                return a
            c2 = np.dot(v, v)
            if c1 >= c2:
                return b
            t = c1 / c2
            return a + t * v
            
        p1 = project_to_segment(p, V1, V2)
        p2 = project_to_segment(p, V2, V3)
        p3 = project_to_segment(p, V3, V1)
        
        d1_sq = np.sum((p - p1)**2)
        d2_sq = np.sum((p - p2)**2)
        d3_sq = np.sum((p - p3)**2)
        
        min_d_sq = min(d1_sq, d2_sq, d3_sq)
        if min_d_sq == d1_sq:
            return p1
        elif min_d_sq == d2_sq:
            return p2
        else:
            return p3

    def min_area_vectorized(points):
        p1 = points[indices[:, 0]]
        p2 = points[indices[:, 1]]
        p3 = points[indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        return np.min(areas)

    def random_points_in_triangle():
        pts = []
        for _ in range(N):
            r1 = random.random()
            r2 = random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            b1 = 1 - r1 - r2
            b2 = r1
            b3 = r2
            p = b1 * V1 + b2 * V2 + b3 * V3
            pts.append(p)
        return np.array(pts)

    def simulated_annealing(start_points, time_limit):
        current_points = start_points.copy()
        current_score = min_area_vectorized(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        T = 0.02
        T_min = 1e-6
        cooling_rate = 0.9995
        step_size = 0.02
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            idx = random.randint(0, N-1)
            new_points = current_points.copy()
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] = new_points[idx] + noise
            new_points[idx] = project_point_to_triangle(new_points[idx])
            
            new_score = min_area_vectorized(new_points)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            T *= cooling_rate
            if T < T_min:
                break
                
        return best_points, best_score

    total_time_limit = 110.0
    start_global = time.time()
    
    best_points_global = None
    best_score_global = -float('inf')
    
    while time.time() - start_global < total_time_limit:
        time_remaining = total_time_limit - (time.time() - start_global)
        if time_remaining < 5:
            break
            
        chunk_time = min(10.0, time_remaining)
        
        init_points = random_points_in_triangle()
        pts, score = simulated_annealing(init_points, chunk_time)
        
        if score > best_score_global:
            best_score_global = score
            best_points_global = pts

    if best_points_global is None:
        best_points_global = random_points_in_triangle()
        
    return best_points_global
