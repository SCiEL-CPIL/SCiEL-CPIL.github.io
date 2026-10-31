"""Heilbronn Triangles Optimizer"""
import numpy as np
import random
import itertools
import math

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, math.sqrt(3)/2])

# Precompute indices for all triples
TRIPLES = list(itertools.combinations(range(N), 3))
I_IDX = np.array([t[0] for t in TRIPLES])
J_IDX = np.array([t[1] for t in TRIPLES])
K_IDX = np.array([t[2] for t in TRIPLES])

def min_triangle_area_vec(points):
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def point_in_triangle(p):
    x, y = p
    # d1 = cross(p, A, B)
    d1 = (B[0]-A[0])*(y-A[1]) - (B[1]-A[1])*(x-A[0])
    # d2 = cross(p, B, C)
    d2 = (C[0]-B[0])*(y-B[1]) - (C[1]-B[1])*(x-B[0])
    # d3 = cross(p, C, A)
    d3 = (A[0]-C[0])*(y-C[1]) - (A[1]-C[1])*(x-C[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def construct():
    rng = np.random.default_rng(42)
    
    def random_point_in_triangle():
        r1 = rng.random()
        r2 = rng.random()
        sqrt_r1 = math.sqrt(r1)
        u = 1 - sqrt_r1
        v = (1 - sqrt_r1) * r2
        w = 1 - u - v
        p = v * B + w * C
        return p

    points = np.array([random_point_in_triangle() for _ in range(N)])
    
    T = 0.05
    T_min = 0.00001
    cooling = 0.999
    max_iters = 100000
    
    current_area = min_triangle_area_vec(points)
    best_points = points.copy()
    best_area = current_area
    
    for i in range(max_iters):
        if T < T_min:
            break
            
        idx = rng.integers(0, N)
        old_pos = points[idx].copy()
        
        step = 0.1 * (T / 0.05) 
        new_pos = old_pos + rng.normal(0, step, 2)
        
        if not point_in_triangle(new_pos):
            continue
        
        points[idx] = new_pos
        new_area = min_triangle_area_vec(points)
        
        delta = new_area - current_area
        
        if delta > 0:
            current_area = new_area
            if new_area > best_area:
                best_area = new_area
                best_points = points.copy()
        else:
            if rng.random() < math.exp(delta / T):
                current_area = new_area
            else:
                points[idx] = old_pos
                
        T *= cooling
        
    return best_points
