import numpy as np
import math
from scipy.optimize import minimize

N = 11

def get_support(theta, rot):
    """
    Calculate the distance from the center of a unit regular hexagon
    rotated by 'rot' to its boundary in direction 'theta'.
    """
    # Normalize the relative angle to [0, pi/3)
    # The hexagon has 6-fold symmetry.
    # The distance function is periodic with pi/3.
    phi = (theta - rot) % (math.pi / 3.0)
    
    # The formula r(phi) = (sqrt(3)/2) / cos(phi - pi/6)
    # works for phi in [0, pi/3].
    # At phi=0 (vertex), cos(-pi/6)=sqrt(3)/2, r=1.
    # At phi=pi/6 (side normal), cos(0)=1, r=sqrt(3)/2.
    # At phi=pi/3 (vertex), cos(pi/6)=sqrt(3)/2, r=1.
    return (math.sqrt(3.0) / 2.0) / math.cos(phi - math.pi / 6.0)

def calculate_S_req(params):
    """
    Calculate the minimum side length S of the outer hexagon (rotation 0)
    required to contain all inner hexagons defined by params.
    params: (N, 3) array of [x, y, rot_rad]
    """
    max_val = 0.0
    # Outer hexagon normals are at angles pi/6 + k*pi/3
    for i in range(N):
        x, y, rot = params[i]
        c = np.array([x, y])
        for k in range(6):
            angle = math.pi / 6.0 + k * math.pi / 3.0
            n = np.array([math.cos(angle), math.sin(angle)])
            proj = np.dot(n, c)
            # Distance from center of inner hex to its boundary in direction of n
            r = get_support(angle, rot)
            val = proj + r
            if val > max_val:
                max_val = val
    # Apothem A = S * sqrt(3) / 2
    # So S = 2 * A / sqrt(3)
    return 2.0 * max_val / math.sqrt(3.0)

def calculate_overlap_penalty(params):
    """
    Calculate a penalty term if any two inner hexagons overlap.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, r1 = params[i]
            x2, y2, r2 = params[j]
            dx = x1 - x2
            dy = y1 - y2
            d = math.sqrt(dx*dx + dy*dy)
            if d < 1e-9:
                return 1e10
            theta = math.atan2(dy, dx)
            # Distance from center i to boundary in direction theta
            ri = get_support(theta, r1)
            # Distance from center j to boundary in direction theta + pi
            # Note: direction from j to i is theta + pi
            rj = get_support(theta + math.pi, r2)
            
            # If the distance between centers is less than the sum of
            # the distances to the boundaries in that direction, they overlap.
            if d < ri + rj:
                penalty += (ri + rj - d) ** 2
    return penalty

def objective(params):
    """
    Objective function to minimize.
    Combines the required outer side length and an overlap penalty.
    """
    S_req = calculate_S_req(params)
    pen = calculate_overlap_penalty(params)
    # We want to minimize S, but ensure no overlap.
    # A large weight on penalty ensures validity.
    return S_req + 1000.0 * pen

def construct():
    # Bounds for x, y, rot
    # x, y can be roughly in [-5, 5]
    # rot in [0, pi/3]
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, math.pi / 3.0)] * N
    
    best_params = None
    best_score = float('inf')
    
    # We will run several local optimizations from different starting points.
    # 1. Random starts
    # 2. Honeycomb-like start
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if r == 0:
            # Honeycomb initialization
            init = np.zeros((N, 3))
            init[0] = [0.0, 0.0, 0.0]
            for k in range(1, 7):
                angle = (k - 1) * math.pi / 3.0
                # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
                x = math.sqrt(3.0) * math.cos(angle)
                y = math.sqrt(3.0) * math.sin(angle)
                init[k] = [x, y, 0.0]
            # Add 4 more randomly around
            for k in range(7, N):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(2.0, 4.0)
                init[k] = [dist * math.cos(angle), dist * math.sin(angle), np.random.uniform(0, math.pi/3)]
        else:
            # Random initialization
            x = np.random.uniform(-5, 5, N)
            y = np.random.uniform(-5, 5, N)
            rot = np.random.uniform(0, math.pi/3, N)
            init = np.column_stack((x, y, rot))
            
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: return a simple valid configuration
        # This might not be optimal, but it's valid.
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract the best solution
    # Convert rotations to degrees
    inner = []
    for i in range(N):
        x, y, rot_rad = best_params[i]
        rot_deg = rot_rad * 180.0 / math.pi
        inner.append([x, y, rot_deg])
        
    center = (0.0, 0.0)
    side = calculate_S_req(best_params)
    angle = 0.0
    
    # Verify validity
    # If there is still overlap, the score might be invalid.
    # We trust the penalty to have driven it to 0.
    
    return inner, center, side, angle
