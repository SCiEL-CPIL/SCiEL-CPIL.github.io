import numpy as np
import itertools
import random
from scipy.optimize import minimize

def construct():
    N = 16
    
    def expand_points(coords):
        """
        coords: array of shape (4, 2) representing 4 points in the first quadrant.
        Returns: array of shape (16, 2) with C4 symmetry.
        """
        pts = []
        for i in range(4):
            x, y = coords[i]
            # Rotate 90, 180, 270
            pts.append([x, y])
            pts.append([-y, x])
            pts.append([-x, -y])
            pts.append([y, -x])
        return np.array(pts)

    def objective(coords):
        P = expand_points(coords)
        # Calculate pairwise distances
        # Use broadcasting for efficiency
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d_min = np.min(dists[mask])
        d_max = np.max(dists[mask])
        
        if d_min < 1e-9:
            return 1e10
            
        ratio = d_max / d_min
        return ratio**2

    best_score = float('inf')
    best_coords = None
    
    # Initial guesses
    # 1. Square Grid approximation
    # Points in first quadrant for 4x4 grid centered at 0:
    # (0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)
    guess1 = np.array([
        [0.5, 0.5],
        [0.5, 1.5],
        [1.5, 0.5],
        [1.5, 1.5]
    ])
    
    # 2. Regular 16-gon approximation projected to C4
    # Angles 0, 22.5, 45, 67.5 degrees? No, 16 points -> 22.5 deg spacing.
    # C4 orbits: 4 points per orbit.
    # Orbit 1: angles 0, 90, 180, 270 -> (1,0)
    # Orbit 2: angles 22.5, 112.5, ...
    # Orbit 3: angles 45, ...
    # Orbit 4: angles 67.5, ...
    # We need 4 points in the first quadrant.
    # Angles: 0, 22.5, 45, 67.5? No, 0 is on axis.
    # Let's just use angles 11.25, 33.75, 56.25, 78.75? 
    # Actually, just pick 4 angles in [0, 90].
    angles = np.array([11.25, 33.75, 56.25, 78.75]) * np.pi / 180.0
    guess2 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 3. Random guesses
    random.seed(42)
    guesses = [guess1, guess2]
    for _ in range(5):
        rand_pts = random.random((4, 2)) * 2.0
        guesses.append(rand_pts)
        
    for guess in guesses:
        # Use Powell method for derivative-free optimization
        res = minimize(objective, guess, method='Powell', options={'maxiter': 2000, 'ftol': 1e-8, 'xtol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_coords = res.x.reshape(4, 2)
            
    # Final check and return
    P = expand_points(best_coords)
    
    # Optional: Local refinement with higher precision or different method
    # Try L-BFGS-B for a final polish
    res2 = minimize(objective, best_coords, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12})
    if res2.fun < best_score:
        best_score = res2.fun
        best_coords = res2.x.reshape(4, 2)
        P = expand_points(best_coords)
        
    return P
