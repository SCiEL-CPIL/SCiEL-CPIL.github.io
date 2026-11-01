import numpy as np
import scipy.optimize
import time
import itertools

N = 14
D = 3

def _pairwise_distances(points):
    # points: (N, D)
    # return (N, N) distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists

def _score(points):
    dists = _pairwise_distances(points)
    # Get upper triangle
    i, j = np.triu_indices(N, k=1)
    d = dists[i, j]
    if np.any(d < 1e-9):
        return 1e10
    d_min = np.min(d)
    d_max = np.max(d)
    return (d_max / d_min) ** 2

def _objective(points_flat):
    points = points_flat.reshape(N, D)
    return _score(points)

def _fibonacci_sphere(n):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def _icosahedron_plus_2():
    # Vertices of regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add 2 more points. 
    # Let's place them at the north and south poles (0, 1, 0) and (0, -1, 0) 
    # and then rotate the icosahedron slightly or just use them as is?
    # The icosahedron already has points near the poles.
    # A better guess: place the 2 extra points at the centers of two opposite faces?
    # Or just random points on the sphere?
    # Let's try adding points at (0, 0, 1) and (0, 0, -1) after rotating the icosahedron 
    # so that no vertex is at the poles.
    # Rotation matrix to move a vertex away from z-axis.
    # Actually, simpler: just take the 12 icosahedron points and add 2 random points on the sphere.
    # Or, use a known 14-point configuration if I can recall one.
    # For now, let's use icosahedron + 2 points at (0,0,1) and (0,0,-1) and let the optimizer fix it.
    extra = np.array([[0, 0, 1], [0, 0, -1]])
    # Rotate icosahedron by 90 degrees around x-axis to avoid poles?
    # Icosahedron vertices are symmetric.
    # Let's just concatenate and let NM do the work.
    points = np.vstack([verts, extra])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron + 2
    inits.append(_icosahedron_plus_2())
    
    # 3. Random points on sphere
    for _ in range(10):
        pts = np.random.randn(N, D)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 4. Maybe a grid on sphere?
    # Just use random for now.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        # Use Nelder-Mead
        try:
            res = scipy.optimize.minimize(_objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_score:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_points = res.x.reshape(N, D)
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline:
            break

    if best_points is None:
        # Fallback to grid or random
        return _fibonacci_sphere(N)
        
    return best_points
