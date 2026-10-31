import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11

# Vertices of the equilateral triangle
# A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) with w = 1-u-v to Cartesian.
    P = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, h)
    P = v*B + (1-u-v)*C
    P = (v, 0) + (1-u-v)*(0.5, h)
    P = (v + 0.5 - 0.5u - 0.5v, h - hu - hv)
    P = (0.5v + 0.5 - 0.5u, h(1-u-v))
    """
    h = np.sqrt(3) / 2.0
    w = 1.0 - u - v
    # P = u*A + v*B + w*C
    # Since A is origin, P = v*B + w*C
    x = v * B[0] + w * C[0]
    y = v * B[1] + w * C[1]
    return np.array([x, y])

def calculate_min_area(points_cart):
    """
    Calculate the minimum area of any triangle formed by the points.
    points_cart: array of shape (N, 2)
    """
    min_area = float('inf')
    # Iterate over all combinations of 3 points
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points_cart[i]
                p2 = points_cart[j]
                p3 = points_cart[k]
                # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """
    Objective function for optimization.
    params: array of shape (2*N,) containing u and v for each point.
    Returns negative of the minimum area (to be minimized).
    """
    u = params[0::2]
    v = params[1::2]
    
    # Check constraints: u >= 0, v >= 0, u+v <= 1
    # If violated, return a large penalty
    if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(u + v > 1.0 + 1e-6):
        return 1e10
    
    # Clip to valid range to avoid numerical issues during optimization steps
    u_clipped = np.clip(u, 0.0, 1.0)
    v_clipped = np.clip(v, 0.0, 1.0 - u_clipped)
    
    points_cart = np.array([bary_to_cart(u_clipped[i], v_clipped[i]) for i in range(N)])
    
    min_area = calculate_min_area(points_cart)
    
    # We want to maximize min_area, so minimize -min_area
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin for 120s limit
    
    best_min_area = -1.0
    best_params = None
    
    # Initializations
    # 1. Grid-like initialization
    # 2. Random initializations
    
    inits = []
    
    # Grid init: 11 points roughly
    # Let's try a 3x3 grid plus 2 extra, or just random
    # A known good structure for 11 points might be 3 corners, 3 edge midpoints, 5 interior.
    # Let's start with a perturbed grid.
    
    # Initialization 1: Regular grid
    # 3x3 grid = 9 points. Add 2 more.
    # Barycentric grid: u in [0, 0.5, 1], v in [0, 0.5, 1]
    # But must satisfy u+v<=1.
    # Valid grid points in barycentric (u,v):
    # (0,0) -> A
    # (0, 0.5) -> Mid AC? No. C is (0,1) in (u,v) if A=(1,0,0)? 
    # Let's stick to Cartesian for init, then convert to Bary for optimization.
    
    def cart_to_bary(p):
        # p = v*B + w*C, with u = 1-v-w
        # x = v + 0.5w
        # y = h w
        # w = y/h
        # v = x - 0.5w
        # u = 1 - v - w
        h = np.sqrt(3) / 2.0
        w = p[1] / h
        v = p[0] - 0.5 * w
        u = 1.0 - v - w
        return u, v
        
    # Init 1: 3 corners + 3 midpoints + 5 interior (roughly)
    corners = [A, B, C]
    mids = [(A+B)/2, (B+C)/2, (C+A)/2]
    # Centroid
    centroid = (A+B+C)/3
    # 4 points around centroid
    # Let's just use random for simplicity and robustness, with a few structured starts.
    
    # Structured Start 1: 3 corners, 3 midpoints, 5 interior (pentagon around centroid)
    pts_struct = corners + mids
    # Add 5 points in a small circle around centroid
    for i in range(5):
        angle = 2 * np.pi * i / 5.0
        r = 0.15
        # Offset in Cartesian
        p = centroid + np.array([r * np.cos(angle), r * np.sin(angle)])
        # Check if inside
        u, v = cart_to_bary(p)
        if u >= 0 and v >= 0 and u+v <= 1:
            pts_struct.append(p)
        else:
            pts_struct.append(centroid) # Fallback
            
    inits.append(pts_struct)
    
    # Structured Start 2: 3x3 grid + 2
    # Barycentric coordinates for grid
    # u in {0, 0.5, 1}, v in {0, 0.5, 1}
    # Valid if u+v <= 1
    grid_pts = []
    for u in [0.0, 0.5, 1.0]:
        for v in [0.0, 0.5, 1.0]:
            if u + v <= 1.0 + 1e-9:
                p = bary_to_cart(u, v)
                grid_pts.append(p)
    # We have 6 points: (0,0), (0,0.5), (0,1), (0.5,0), (0.5,0.5), (1,0)
    # Wait, (0,1) is C. (1,0) is B. (0,0) is A.
    # (0, 0.5) is mid AC. (0.5, 0) is mid AB. (0.5, 0.5) is... v=0.5, u=0.5, w=0. P = 0.5B + 0.5C. Mid BC.
    # So we have 3 corners and 3 midpoints.
    # Need 5 more. Let's add centroid and 4 more.
    pts_grid = grid_pts + [centroid]
    for i in range(4):
        angle = 2 * np.pi * i / 4.0 + np.pi/4
        r = 0.2
        p = centroid + np.array([r * np.cos(angle), r * np.sin(angle)])
        u, v = cart_to_bary(p)
        if u >= 0 and v >= 0 and u+v <= 1:
            pts_grid.append(p)
        else:
            pts_grid.append(centroid)
    inits.append(pts_grid)
    
    # Random Inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts_rand = []
        for _ in range(N):
            # Random point in triangle
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            p = bary_to_cart(u, v)
            pts_rand.append(p)
        inits.append(pts_rand)
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        # Convert initial points to barycentric params
        params0 = []
        for p in init_pts:
            u, v = cart_to_bary(p)
            # Ensure valid
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0 - u, v))
            params0.extend([u, v])
            
        params0 = np.array(params0)
        
        # Optimize
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < objective(params0):
                final_params = res.x
                u = final_params[0::2]
                v = final_params[1::2]
                # Clip
                u = np.clip(u, 0.0, 1.0)
                v = np.clip(v, 0.0, 1.0 - u)
                
                final_points = np.array([bary_to_cart(u[i], v[i]) for i in range(N)])
                
                # Calculate actual min area
                min_area = calculate_min_area(final_points)
                
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_params = final_points
                    
        except Exception as e:
            pass
            
    if best_params is not None:
        return best_params
    else:
        # Fallback: return the structured one if optimization failed
        return np.array(inits[0])

# EVOLVE-BLOCK-END
