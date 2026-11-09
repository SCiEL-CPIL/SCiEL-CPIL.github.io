import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial guess: Regular Octagon
    # P1=(-1,0), P5=(1,0)
    # P3=(0,1), P7=(0,-1)
    # P2=(-0.707, 0.707), P4=(0.707, 0.707), P6=(0.707, -0.707), P8=(-0.707, -0.707)
    # Variables: x2, y2, y3
    # P2 = (x2, y2)
    # P4 = (-x2, y2)
    # P3 = (0, y3)
    # P6 = (-x2, -y2)
    # P8 = (x2, -y2)
    # P7 = (0, -y3)
    
    # Regular octagon values:
    # x2 = -1/sqrt(2) approx -0.7071
    # y2 = 1/sqrt(2) approx 0.7071
    # y3 = 1.0
    
    x0 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 1.0])
    
    def get_points(vars):
        x2, y2, y3 = vars
        P1 = np.array([-1.0, 0.0])
        P5 = np.array([1.0, 0.0])
        P2 = np.array([x2, y2])
        P4 = np.array([-x2, y2])
        P3 = np.array([0.0, y3])
        P6 = np.array([-x2, -y2])
        P8 = np.array([x2, -y2])
        P7 = np.array([0.0, -y3])
        return np.array([P1, P2, P3, P4, P5, P6, P7, P8])
    
    def objective(vars):
        pts = get_points(vars)
        # Calculate all pairwise distances
        # 8 points -> 28 pairs
        d = []
        for i in range(8):
            for j in range(i+1, 8):
                d.append(np.linalg.norm(pts[i] - pts[j]))
        d = np.array(d)
        
        # Check constraints: all distances <= 2
        # If any distance > 2, return a large penalty
        if np.any(d > 2.0 + 1e-9):
            return 1e10
        
        # Also ensure points are distinct (distance > epsilon)
        if np.any(d < 1e-6):
            return 1e10
            
        # Maximize product of distances = Minimize negative sum of log distances
        # Avoid log(0)
        if np.any(d <= 0):
            return 1e10
            
        return -np.sum(np.log(d))
    
    best_score = -1e10
    best_vars = x0
    
    # Try a few random restarts or perturbations around the regular octagon
    # Since the search space is small, we can do a grid search or multiple minimizations
    
    # Method 1: Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_vars = res.x
    except Exception:
        pass
    
    # Method 2: Powell
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_vars = res.x
    except Exception:
        pass
    
    # Try some perturbations
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        pert = np.random.normal(0, 0.1, 3)
        x_start = x0 + pert
        # Clamp to reasonable ranges
        x_start[0] = np.clip(x_start[0], -1.5, 0)
        x_start[1] = np.clip(x_start[1], 0, 1.5)
        x_start[2] = np.clip(x_start[2], 0, 1.5)
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_vars = res.x
        except Exception:
            pass
            
    # Finalize points
    pts = get_points(best_vars)
    
    # Scale to ensure diameter is exactly 2?
    # The problem says "after scaling the set to diameter 2".
    # Our points already have diameter 2 (P1 to P5 is 2).
    # But we should check if any other pair is > 2. The objective function penalizes it.
    # If the optimizer found a point with d > 2, the score would be 1e10, which is bad.
    # We assume the best_vars found are valid.
    
    # Convert to list of lists of floats
    return pts.tolist()
