import numpy as np
from scipy.optimize import minimize

N = 12
EPS = 0.02  # margin for non-overlap

def get_centers():
    # 4-4-2-2 topology
    # Scale: distance between centers in x is sqrt(3), in y is 1.5
    # Let's use standard hex grid coords
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 2 hexes (centered)
    # Row 3: 2 hexes (centered)
    
    centers = []
    # Row 0
    for j in range(4):
        cx = (j - 1.5) * np.sqrt(3)
        cy = 0.0
        centers.append([cx, cy])
    # Row 1
    for j in range(4):
        cx = (j - 1.5) * np.sqrt(3)
        cy = 1.5
        centers.append([cx, cy])
    # Row 2
    for j in range(2):
        cx = (j - 0.5) * np.sqrt(3)
        cy = 3.0
        centers.append([cx, cy])
    # Row 3
    for j in range(2):
        cx = (j - 0.5) * np.sqrt(3)
        cy = 4.5
        centers.append([cx, cy])
        
    return np.array(centers)

def inner_vertices(phi, centers):
    verts = []
    for c in centers:
        # rotate center
        c_rot = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
        for k in range(6):
            a = phi + np.pi/3*k
            verts.append([c_rot[0]+np.cos(a), c_rot[1]+np.sin(a)])
    return np.array(verts)

def s_min(params, centers):
    phi, theta, cx, cy = params
    verts = inner_vertices(phi, centers)
    c = np.array([cx, cy])
    # normals of outer hexagon
    maxdot = -np.inf
    for i in range(6):
        a = theta + np.pi/6 + np.pi/3*i
        n = np.array([np.cos(a), np.sin(a)])
        dots = verts @ n - (c @ n)
        maxdot = max(maxdot, dots.max())
    return maxdot / np.cos(np.pi/6)

def construct():
    centers = get_centers()
    best = None
    best_s = np.inf
    
    # Multi-start over rotations
    for phi0 in np.linspace(0, np.pi/3, 7):
        for theta0 in np.linspace(0, np.pi/3, 7):
            x0 = [phi0, theta0, 0.0, 0.0]
            res = minimize(s_min, x0, args=(centers,), method='Nelder-Mead', options={'maxiter': 2000, 'xatol':1e-6, 'fatol':1e-6})
            s = res.fun
            if s < best_s:
                best_s = s
                best = res.x
                
    phi, theta, cx, cy = best
    s = best_s
    
    # Build inner list
    inner = []
    for c in centers:
        c_rot = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
        inner.append([c_rot[0], c_rot[1], np.degrees(phi)])
        
    return inner, (cx, cy), s, np.degrees(theta)
