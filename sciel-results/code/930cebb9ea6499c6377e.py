import numpy as np
from scipy.optimize import basinhopping, minimize
import time
import itertools

N = 16
D = 2

def compute_ratio(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    d_max = np.max(dist_sq)
    d_min = np.min(dist_sq)
    if d_min < 1e-12:
        return 1e12
    return (d_max / d_min)**2

def objective(x):
    points = x.reshape(N, D)
    return compute_ratio(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_val = float('inf')
    best_x = None
    
    inits = []
    
    # 1. 4x4 Grid
    grid = np.array(list(itertools.product(range(4), repeat=2)), dtype=float)
    grid -= np.mean(grid, axis=0)
    inits.append(grid.flatten())
    
    # 2. Hexagonal lattice
    hex_points = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            hex_points.append((x, y))
    hex_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    hex_16 = np.array(hex_points[:16], dtype=float)
    hex_16 -= np.mean(hex_16, axis=0)
    inits.append(hex_16.flatten())
    
    # 3. Sunflower
    golden_angle = np.pi * (3 - np.sqrt(5))
    sunflower = np.array([[np.sqrt(i/N)*np.cos(i*golden_angle), np.sqrt(i/N)*np.sin(i*golden_angle)] for i in range(N)])
    inits.append(sunflower.flatten())
    
    # 4. Random Circle
    np.random.seed(42)
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts.flatten())
        
    # 5. Random Square
    for _ in range(5):
        pts = np.random.uniform(-1, 1, (N, D))
        inits.append(pts.flatten())
        
    # 6. Two Rings
    for r1, r2 in [(0.5, 1.0), (0.6, 1.0), (0.4, 0.9)]:
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(np.pi/8, 2*np.pi, 8, endpoint=False) # Offset
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        inits.append(pts.flatten())

    # Optimization
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = basinhopping(
                objective, 
                x0, 
                niter=150, 
                T=1.0, 
                stepsize=0.5, 
                minimizer_kwargs={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    if best_x is None:
        best_x = inits[0]
        best_val = objective(best_x)

    points = best_x.reshape(N, D)
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, points, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000})
            if res.fun < best_val:
                best_val = res.fun
                points = res.x.reshape(N, D)
        except Exception:
            pass

    return points
