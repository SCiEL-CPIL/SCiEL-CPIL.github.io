import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute indices for all combinations of 3 points from N
_i, _j, _k = [], [], []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            _i.append(i)
            _j.append(j)
            _k.append(k)
_idx_i = np.array(_i)
_idx_j = np.array(_j)
_idx_k = np.array(_k)

def min_area(pts):
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def project_to_triangle(p):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
    c1 = cross(A, B, p)
    c2 = cross(B, C, p)
    c3 = cross(C, A, p)
    
    if c1 >= -1e-9 and c2 >= -1e-9 and c3 >= -1e-9:
        return p

    def dist_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj), proj
        
    d1, p1 = dist_to_segment(p, A, B)
    d2, p2 = dist_to_segment(p, B, C)
    d3, p3 = dist_to_segment(p, C, A)
    
    min_d = min(d1, d2, d3)
    if min_d == d1:
        return p1
    elif min_d == d2:
        return p2
    else:
        return p3

def objective(x):
    pts = x.reshape(N, 2)
    proj_pts = np.array([project_to_triangle(pts[i]) for i in range(N)])
    area = min_area(proj_pts)
    return -area

def construct():
    rng = np.random.default_rng(42)
    best_score = 0
    best_points = None
    
    num_restarts = 150
    max_iter = 1000
    
    for restart in range(num_restarts):
        x0 = np.zeros(N * 2)
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v <= 1:
                    p = u * B + v * C
                    x0[2*i] = p[0]
                    x0[2*i+1] = p[1]
                    break
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
        except Exception:
            continue
            
    return best_points
