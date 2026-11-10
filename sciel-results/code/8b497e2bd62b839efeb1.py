import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import random
import math

def compute_score(P):
    """
    Computes the score Delta/n^n for 8 points P.
    Returns the score value.
    """
    P = np.array(P)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return -np.inf
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return -np.inf
        
    sum_log_dists = np.sum(np.log(dists))
    
    # Score is exp(sum_log_dists - 28 * log(D/2)) / 8^8
    # log_score = sum_log_dists - 28 * log(D/2) - 8 * log(8)
    log_score = sum_log_dists - 28 * np.log(D / 2.0) - 8 * np.log(8)
    
    # Return actual score, but for optimization we often work with log.
    # However, the task asks for the score value.
    # To avoid overflow/underflow issues in return, we return the value.
    # If log_score is very negative, score is 0.
    if log_score < -700:
        return 0.0
    if log_score > 700:
        return np.inf
    return np.exp(log_score)

def score_from_params_5d(params):
    """
    params: [r1, theta1, r2, theta2, offset] 
    Actually, let's use a more flexible 5-param structure:
    Two squares, potentially rotated and scaled.
    Square 1: 4 points at radius r1, angles theta1 + k*pi/2
    Square 2: 4 points at radius r2, angles theta2 + k*pi/2
    
    To allow general 2D shifts and rotations, we can fix the center at (0,0) 
    and rotation of the whole set (which doesn't change distances) by fixing one angle.
    However, the relative rotation between the two squares matters.
    
    Let's define:
    r1: radius of square 1
    r2: radius of square 2
    theta_rel: relative rotation of square 2 w.r.t square 1
    x_center, y_center: center of the configuration (can be set to 0,0 WLOG due to translation invariance of distances)
    
    Wait, the problem is invariant under translation and rotation.
    So we can fix the center at (0,0) and one square's rotation to 0.
    Parameters: r1, r2, theta_rel.
    That's only 3 parameters.
    
    Is that enough? 
    If the optimal solution is two concentric squares, yes.
    What if they are not concentric?
    The "two squares" structure in the literature for n=8 often refers to two squares centered at the origin but rotated relative to each other.
    
    Let's expand to 5 parameters to be safe:
    r1, r2, theta1, theta2, and maybe a radial offset? No, let's stick to the manifold of two concentric squares first, 
    but allow independent rotations.
    
    Actually, let's look at the "inspiration" note: "5-parameter two-squares manifold".
    Likely: r1, r2, theta1, theta2, and maybe a scale? Or just 4 params + 1 noise?
    
    Let's use:
    r1: radius of first square
    r2: radius of second square
    theta1: rotation of first square
    theta2: rotation of second square
    x_shift: horizontal shift of the whole config? No, WLOG center is 0.
    
    Perhaps the 5th parameter is a "tilt" or something?
    Let's try: r1, r2, theta1, theta2, and alpha (where alpha scales the distance between centers if we allow non-concentric?).
    
    Let's try a slightly more general 5-param model:
    Points 0-3: Square 1 centered at (0,0), radius r1, angle theta1
    Points 4-7: Square 2 centered at (c, 0), radius r2, angle theta2
    Params: r1, r2, theta1, theta2, c
    This allows non-concentric squares.
    """
    r1, r2, theta1, theta2, c = params
    
    # Square 1
    angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    P1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2
    angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    P2 = np.column_stack([r2 * np.cos(angles2) + c, r2 * np.sin(angles2)])
    
    P = np.vstack([P1, P2])
    
    return compute_score(P)

def objective_5d(params):
    """
    Negative log score for optimization.
    """
    r1, r2, theta1, theta2, c = params
    
    angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    P1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    P2 = np.column_stack([r2 * np.cos(angles2) + c, r2 * np.sin(angles2)])
    
    P = np.vstack([P1, P2])
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    
    # We want to maximize sum_log_dists - 28*log(D/2)
    # Minimize negative of that.
    return -(sum_log_dists - 28 * np.log(D / 2.0))

def objective_16d(x):
    """
    x: flattened 16 coordinates.
    Returns negative log score (unnormalized part).
    """
    P = x.reshape(8, 2)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    
    return -(sum_log_dists - 28 * np.log(D / 2.0))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_val = -1e100
    best_points = None
    
    # 1. Global search on 5-parameter manifold
    # Bounds:
    # r1, r2: [0.1, 2.0] (diameter will be scaled later, so relative sizes matter)
    # theta1, theta2: [0, 2*pi]
    # c: [-1.0, 1.0] (shift)
    
    bounds = [
        (0.1, 2.0), # r1
        (0.1, 2.0), # r2
        (0.0, 2*np.pi), # theta1
        (0.0, 2*np.pi), # theta2
        (-1.0, 1.0)  # c
    ]
    
    # Differential Evolution
    try:
        de_res = differential_evolution(
            objective_5d, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=False
        )
        
        if time.time() < deadline - 10:
            # Local polish on 5D
            res_5d = minimize(objective_5d, de_res.x, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res_5d.fun > best_val:
                best_val = -res_5d.fun
                # Reconstruct points
                r1, r2, theta1, theta2, c = res_5d.x
                angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                P1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                P2 = np.column_stack([r2 * np.cos(angles2) + c, r2 * np.sin(angles2)])
                best_points = np.vstack([P1, P2])
    except Exception:
        pass
    
    if best_points is None:
        # Fallback to regular octagon
        angles = np.arange(8) * (2 * np.pi / 8)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_val = -objective_16d(best_points.flatten())
        
    if time.time() > deadline - 20:
        # Scale and return
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        D = np.sqrt(np.max(dists_sq))
        if D > 0:
            best_points = best_points * (2.0 / D)
        return best_points.tolist()

    # 2. Local polish in full 16D space starting from the best manifold point
    x_start = best_points.flatten()
    
    # Try multiple local optimizations from the manifold result
    for i in range(10):
        if time.time() > deadline - 5:
            break
        
        # Add small noise to start points to escape local minima
        if i == 0:
            x_curr = x_start
        else:
            x_curr = x_start + np.random.normal(0, 0.01, 16)
        
        try:
            res_16d = minimize(objective_16d, x_curr, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
            val_16d = -res_16d.fun
            
            if val_16d > best_val:
                best_val = val_16d
                best_points = res_16d.x.reshape(8, 2)
        except Exception:
            pass

    # Final scaling to diameter 2
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    D = np.sqrt(np.max(dists_sq))
    
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    print(compute_score(pts))
