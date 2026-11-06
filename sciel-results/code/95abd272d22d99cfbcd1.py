import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    # Diameter 2 -> Radius 1
    P_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # We will try to optimize around this
    # We fix P[0] and P[1] to be at (-1,0) and (1,0)?
    # The regular octagon has vertices at 0, 45, 90...
    # Vertex 0 is (1,0). Vertex 4 is (-1,0).
    # So fixing 0 and 4 as the diameter pair makes sense for the octagon.
    
    # Let's fix P_0 = (1,0) and P_4 = (-1,0).
    # Indices: 0, 4 fixed.
    # Free: 1, 2, 3, 5, 6, 7.
    
    def log_product(pts):
        # pts is 8x2
        d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(d, 1.0) # avoid log(0)
        # Check for zero distances (distinct points)
        if np.any(d < 1e-9):
            return -1e10
        return np.sum(np.log(d))
        
    def objective(x):
        # x is 12 values: x1,y1, x2,y2, x3,y3, x5,y5, x6,y6, x7,y7
        pts = np.zeros((8, 2))
        pts[0] = [1.0, 0.0]
        pts[4] = [-1.0, 0.0]
        
        # Map x to pts
        pts[1] = [x[0], x[1]]
        pts[2] = [x[2], x[3]]
        pts[3] = [x[4], x[5]]
        pts[5] = [x[6], x[7]]
        pts[6] = [x[8], x[9]]
        pts[7] = [x[10], x[11]]
        
        # Constraint: all distances <= 2
        d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(d, 0)
        max_d = np.max(d)
        if max_d > 2.0 + 1e-6:
            # Penalize violation
            return -log_product(pts) + 100 * (max_d - 2.0)**2
        
        return -log_product(pts)

    def constraint_func(x):
        # Inequality constraints: d_ij <= 2
        # We can just use the penalty in objective for simplicity if SLSQP is slow
        # But SLSQP with constraints is better.
        pass

    # Let's use SLSQP with explicit constraints
    def constraints(x):
        pts = np.zeros((8, 2))
        pts[0] = [1.0, 0.0]
        pts[4] = [-1.0, 0.0]
        pts[1] = [x[0], x[1]]
        pts[2] = [x[2], x[3]]
        pts[3] = [x[4], x[5]]
        pts[5] = [x[6], x[7]]
        pts[6] = [x[8], x[9]]
        pts[7] = [x[10], x[11]]
        
        d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
        np.fill_diagonal(d, 0)
        # Return 2 - d_ij for all i<j
        cons = []
        for i in range(8):
            for j in range(i+1, 8):
                cons.append(2.0 - d[i,j])
        return cons

    # Initial guess from regular octagon
    # Octagon vertices:
    # 0: (1, 0)
    # 1: (cos 45, sin 45)
    # 2: (0, 1)
    # 3: (-cos 45, sin 45)
    # 4: (-1, 0)
    # 5: (-cos 45, -sin 45)
    # 6: (0, -1)
    # 7: (cos 45, -sin 45)
    
    c = np.cos(np.pi/4)
    s = np.sin(np.pi/4)
    
    x0 = np.array([
        c, s,       # P1
        0, 1,       # P2
        -c, s,      # P3
        -c, -s,     # P5
        0, -1,      # P6
        c, -s       # P7
    ])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Optimization
    for i in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Add small random perturbation for restarts
        x_start = x0 + np.random.normal(0, 0.1, 12)
        
        res = minimize(objective, x_start, method='SLSQP', bounds=[(-2, 2)]*12)
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Construct final points
    pts = np.zeros((8, 2))
    pts[0] = [1.0, 0.0]
    pts[4] = [-1.0, 0.0]
    pts[1] = [best_x[0], best_x[1]]
    pts[2] = [best_x[2], best_x[3]]
    pts[3] = [best_x[4], best_x[5]]
    pts[5] = [best_x[6], best_x[7]]
    pts[6] = [best_x[8], best_x[9]]
    pts[7] = [best_x[10], best_x[11]]
    
    # Verify diameter is 2
    d = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=-1))
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    
    # Scale if necessary (should be close to 2)
    if diam > 0:
        pts = pts * (2.0 / diam)
        
    # Convert to list of lists
    return pts.tolist()
