import numpy as np
from scipy.optimize import minimize
import time

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])
VERTS = np.array([A, B, C])

def barycentric_coords(p, v1, v2, v3):
    v0 = v2 - v1
    v1v = v3 - v1
    v2v = p - v1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1v)
    d11 = np.dot(v1v, v11) if False else np.dot(v1v, v1v) # Typo check
    d20 = np.dot(v2v, v0)
    d21 = np.dot(v2v, v1v)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def all_inside(points):
    for p in points:
        u, v, w = barycentric_coords(p, A, B, C)
        if u < -1e-8 or v < -1e-8 or w < -1e-8:
            return False
    return True

def min_area(points):
    min_a = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_a:
                    min_a = area
    return min_a

def objective(x):
    points = x.reshape(N, 2)
    if not all_inside(points):
        return 1e10
    return -min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -float('inf')
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Structured Initialization
    # 3 near vertices
    # 3 near midpoints
    # 5 central
    
    def get_structured_init():
        pts = []
        # Vertices with small offset inwards
        offset = 0.05
        pts.append(A + offset * ( (B+C)/2 - A ))
        pts.append(B + offset * ( (A+C)/2 - B ))
        pts.append(C + offset * ( (A+B)/2 - C ))
        
        # Midpoints
        pts.append((A+B)/2)
        pts.append((B+C)/2)
        pts.append((C+A)/2)
        
        # Center
        center = (A+B+C)/3
        pts.append(center)
        
        # 4 points around center
        r = 0.15
        angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        for ang in angles:
            p = center + r * np.array([np.cos(ang), np.sin(ang)])
            pts.append(p)
            
        return np.array(pts)
    
    # Try structured init first
    x0 = get_structured_init().flatten()
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < 1e9:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
    except:
        pass

    # Random restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random init
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * A + v * B + (1 - u - v) * C
                pts.append(p)
        
        x0 = np.array(pts).flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < 1e9:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except:
            pass

    if best_points is None:
        # Fallback
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * A + v * B + (1 - u - v) * C
                pts.append(p)
        return np.array(pts)
        
    return best_points
