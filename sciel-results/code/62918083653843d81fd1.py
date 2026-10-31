import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
tri_indices = np.array(list(combinations(range(N), 3)))
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

def min_area(points):
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside(points):
    if np.any(points[:, 1] < -1e-9): return False
    if np.any(SQRT3 * points[:, 0] - points[:, 1] < -1e-9): return False
    if np.any(-SQRT3 * points[:, 0] + SQRT3 - points[:, 1] < -1e-9): return False
    return True

def objective(params):
    points = params.reshape(N, 2)
    if not is_inside(points):
        return 1e6
    return -min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Grid initialization
    grid_points = []
    for i in range(4):
        grid_points.append([i/3.0, 0])
    for i in range(3):
        grid_points.append([1/6.0 + i/3.0, H/3.0])
    for i in range(2):
        grid_points.append([1/3.0 + i/3.0, 2*H/3.0])
    grid_points.append([0.5, H])
    # 11th point
    grid_points.append([0.5, H/6.0])
    
    initial_guesses = [np.array(grid_points)]
    
    # Add a random guess
    np.random.seed(42)
    rand_points = []
    while len(rand_points) < N:
        u, v = np.random.random(), np.random.random()
        if u + v < 1:
            p = u * V[1] + v * V[2]
            rand_points.append(p)
    initial_guesses.append(np.array(rand_points))
    
    # Add a perturbed grid
    perturbed = initial_guesses[0].copy()
    perturbed += np.random.normal(0, 0.05, perturbed.shape)
    # Project back inside? Just let penalty handle it, or clip
    # Simple clip to triangle is hard, let's just use it as is
    initial_guesses.append(perturbed)
    
    best_points = initial_guesses[0]
    best_score = min_area(best_points)
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            # Use Nelder-Mead
            res = minimize(objective, guess.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'initial_simplex': 0.1})
            if res.fun < 0:
                curr_points = res.x.reshape(N, 2)
                if is_inside(curr_points):
                    curr_score = min_area(curr_points)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = curr_points
        except Exception:
            pass
            
    return best_points
