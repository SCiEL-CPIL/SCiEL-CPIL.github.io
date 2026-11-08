import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
# Vertices of the standard equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute indices for all C(11, 3) = 165 triangles
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _min_triangle_area(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    points: shape (N, 2)
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_to_triangle(p):
    """
    Project a single point p to the triangle T using barycentric coordinates clipping.
    """
    v1_p = V1 - p
    v2_p = V2 - p
    w0 = v1_p[0]*v2_p[1] - v1_p[1]*v2_p[0]
    
    p_v0 = p - V0
    v2_v0 = V2 - V0
    w1 = p_v0[0]*v2_v0[1] - p_v0[1]*v2_v0[0]
    
    v1_v0 = V1 - V0
    w2 = v1_v0[0]*p_v0[1] - v1_v0[1]*p_v0[0]
    
    total = w0 + w1 + w2
    
    if total == 0:
        return V0.copy()
        
    l0 = w0 / total
    l1 = w1 / total
    l2 = w2 / total
    
    # Clip negative weights to 0
    l0 = max(0.0, l0)
    l1 = max(0.0, l1)
    l2 = max(0.0, l2)
    
    s = l0 + l1 + l2
    if s == 0:
        return V0.copy()
        
    l0 /= s
    l1 /= s
    l2 /= s
    
    return l0*V0 + l1*V1 + l2*V2

def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected

def _objective_flat(x):
    """
    Objective function for the optimizer.
    x: flat array of length 2*N
    Returns negative min area (to maximize min area, we minimize -min area)
    """
    points = x.reshape(N, 2)
    points = _project_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    rng = random.Random(42)
    
    def _local_search(points, time_limit, initial_temp=0.01, initial_step=0.05):
        current_points = points.copy()
        current_score = _min_triangle_area(current_points)
        best_local_points = current_points.copy()
        best_local_score = current_score
        
        temp = initial_temp
        cooling = 0.999
        step_size = initial_step
        stagnation_count = 0
        max_stagnation = 500
        
        while time.time() < time_limit:
            # Pick a random point to move
            i = rng.randint(0, N-1)
            
            # Generate a random displacement
            dx = rng.gauss(0, step_size)
            dy = rng.gauss(0, step_size)
            
            new_points = current_points.copy()
            new_points[i] = [current_points[i][0] + dx, current_points[i][1] + dy]
            
            # Project back to triangle
            new_points[i] = _project_to_triangle(new_points[i])
            
            new_score = _min_triangle_area(new_points)
            
            # Accept if better, or with probability based on temperature
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                stagnation_count = 0
                if current_score > best_local_score:
                    best_local_points = current_points.copy()
                    best_local_score = current_score
            else:
                delta = current_score - new_score
                if delta < 1e-9:
                    continue
                prob = np.exp(-delta / temp)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.005
                step_size = 0.02
            
            # If stuck for too long, perturb the best solution to escape local optimum
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                # Perturb current points slightly
                for j in range(N):
                    current_points[j] = current_points[j] + rng.gauss(0, 0.01, 2)
                    current_points[j] = _project_to_triangle(current_points[j])
                current_score = _min_triangle_area(current_points)
                temp = 0.01
                step_size = 0.05
                stagnation_count = 0
                
        return best_local_points, best_local_score

    # Run multiple restarts of local search
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial points
        if restart == 0:
            # Simple random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        elif restart % 3 == 1:
            # Perturb the best found so far
            if best_points is not None:
                points = best_points + rng.gauss(0, 0.1, (N, 2))
                # Project all points
                for i in range(N):
                    points[i] = _project_to_triangle(points[i])
            else:
                for i in range(N):
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1.0:
                            points[i] = [u + 0.5 * v, H * v]
                            break
        else:
            # New random initialization with different seed
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        
        # Run local search with a time slice
        time_slice = min(8.0, deadline - 5.0 - time.time())
        if time_slice <= 0:
            break
            
        res_points, res_score = _local_search(points, time.time() + time_slice)
        
        if res_score > best_score:
            best_points = res_points
            best_score = res_score

    # If no optimization ran or failed, return best random
    if best_points is None:
        # Fallback to random
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_points = np.array(points)
        
    # Final check: ensure points are inside
    best_points = _project_points(best_points)
    
    return best_points
