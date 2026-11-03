import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 110 # seconds

def area(p1, p2, p3):
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_area(points):
    # points shape (11, 2)
    min_a = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                a = area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def is_inside_triangle(p):
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3)*x  (Left edge: y = sqrt(3)x) -> Actually, the edge is from (0,0) to (0.5, sqrt(3)/2). 
    #    Wait, the slope is sqrt(3). The line is y = sqrt(3)x.
    #    Points inside must satisfy y <= sqrt(3)x? 
    #    Let's check (0.5, 0.1). y=0.1, sqrt(3)*0.5 = 0.866. 0.1 <= 0.866. Yes.
    #    Let's check (0.1, 0.8). y=0.8, sqrt(3)*0.1 = 0.17. 0.8 > 0.17. Outside.
    #    So y <= sqrt(3)*x is correct for the left boundary?
    #    Wait, the triangle is to the RIGHT of the line y=sqrt(3)x?
    #    Center (0.5, 0.28). 0.28 <= 0.866. Yes.
    #    3. y <= -sqrt(3)*x + sqrt(3) (Right edge)
    return y >= 0 and y <= np.sqrt(3)*x and y <= -np.sqrt(3)*x + np.sqrt(3)

def objective_flat(flat_points):
    points = flat_points.reshape(N, 2)
    # Penalty if outside
    for p in points:
        if not is_inside_triangle(p):
            return -1.0 # Very bad
    return -min_area(points) # Minimize negative max-min-area

def construct():
    start_time = time.time()
    best_points = None
    best_score = -float('inf')
    
    # Symmetric Initialization
    # Using barycentric coordinates for robustness
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*A + v*B + w*C
    
    def bary_to_cart(u, v, w):
        x = v + 0.5*w
        y = np.sqrt(3)/2 * w
        return np.array([x, y])
        
    # Define a set of symmetric initial configurations
    inits = []
    
    # Config 1: Center + 3 mid-sides + 6 hexagon
    center = bary_to_cart(1/3, 1/3, 1/3)
    mids = [
        bary_to_cart(0, 0.5, 0.5),
        bary_to_cart(0.5, 0, 0.5),
        bary_to_cart(0.5, 0.5, 0)
    ]
    # 6 points around center
    # In barycentric, rotate around center?
    # Simpler: Just pick points from a grid
    
    # Grid Initialization
    # Generate points on a triangular grid
    grid_points = []
    M = 6 # Resolution
    for i in range(M+1):
        for j in range(M+1-i):
            k = M - i - j
            u, v, w = i/M, j/M, k/M
            p = bary_to_cart(u, v, w)
            grid_points.append(p)
            
    # Select 11 points from grid that maximize min-area?
    # Or just take a symmetric subset.
    # Let's take the center, 3 near corners, 3 near mid-sides, 4 intermediate.
    
    # Instead of complex selection, let's just use a few random symmetric inits and optimize.
    # Or better: Use the grid points, sort by distance to center, take 11?
    # Let's try taking points from the grid that are well distributed.
    
    # A better init: 
    # 1. Center
    # 2. 3 points at 1/3 way to vertices
    # 3. 3 points at 1/3 way to mid-sides
    # 4. 4 points ...
    
    # Let's just use a simple heuristic: 
    # Take the 11 grid points that are farthest from each other?
    
    # Actually, let's just run Nelder-Mead from a few good starting points.
    # Starting Point 1: 
    # 1 center
    # 3 at (1/6, 1/3, 1/2) perms? 
    # Let's generate a hexagonal packing init.
    
    hex_init = []
    # Center
    hex_init.append(bary_to_cart(1/3, 1/3, 1/3))
    # 6 neighbors
    # In barycentric, neighbors of (1/3,1/3,1/3) in a hex grid are not trivial.
    # Let's just use Cartesian hex grid.
    # Center (0.5, sqrt(3)/6)
    # Radius r = 0.1
    # Angles 0, 60, ...
    
    center_c = np.array([0.5, np.sqrt(3)/6])
    r1 = 0.15
    for i in range(6):
        theta = i * np.pi / 3
        x = center_c[0] + r1 * np.cos(theta)
        y = center_c[1] + r1 * np.sin(theta)
        if is_inside_triangle([x,y]):
            hex_init.append(np.array([x,y]))
        else:
            # Project to boundary or skip?
            # Just add it, penalty will handle it, or clip
            hex_init.append(np.array([x,y]))
            
    # We have 7 points. Need 4 more.
    # Add points near vertices?
    # (0.1, 0.1), (0.9, 0.1), (0.5, 0.4)
    # Let's just append random points inside and let optimizer fix them.
    # Or better, add 4 points from the grid that are far from existing ones.
    
    # Let's use a simpler robust init:
    # Take 11 points from the M=10 grid that are uniformly distributed.
    
    # For now, let's just use the hex_init + 4 random points inside.
    while len(hex_init) < N:
        # Random point in triangle
        u = np.random.random()
        v = np.random.random()
        if u + v < 1:
            # Map to triangle
            # Standard mapping: 
            # x = (1-u)*0 + u*(1-v)*1 + v*0.5 ? No.
            # Barycentric: 
            # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
            # P = (1-u-v)A + uB + vC
            # x = u + 0.5v
            # y = sqrt(3)/2 v
            x = u + 0.5*v
            y = np.sqrt(3)/2 * v
            hex_init.append(np.array([x,y]))
            
    inits.append(np.array(hex_init))
    
    # Add a few more inits with different r1
    for r in [0.1, 0.2, 0.25]:
        init_pts = []
        init_pts.append(bary_to_cart(1/3, 1/3, 1/3))
        for i in range(6):
            theta = i * np.pi / 3
            x = center_c[0] + r * np.cos(theta)
            y = center_c[1] + r * np.sin(theta)
            init_pts.append(np.array([x,y]))
        while len(init_pts) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v < 1:
                x = u + 0.5*v
                y = np.sqrt(3)/2 * v
                init_pts.append(np.array([x,y]))
        inits.append(np.array(init_pts))
        
    # Run optimization
    for init in inits:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        flat_init = init.flatten()
        try:
            res = minimize(objective_flat, flat_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success or res.fun < -best_score:
                # Check if valid
                pts = res.x.reshape(N, 2)
                valid = all(is_inside_triangle(p) for p in pts)
                if valid:
                    score = min_area(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to init
        best_points = inits[0]
        
    return best_points

construct()
