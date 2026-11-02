import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Normals for unit hexagon (side 1)
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Define the "Star" topology: 1 center, 6 inner, 5 outer
    # Inner ring angles: 0, 60, 120, 180, 240, 300
    # Outer ring angles: 0, 60, 120, 180, 240 (missing 300)
    
    inner_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    outer_angles = np.deg2rad([0, 60, 120, 180, 240])
    
    def get_centers(params):
        r1, r2, rot = params
        rot_rad = np.deg2rad(rot)
        
        centers = np.zeros((N, 2))
        # Center
        centers[0] = [0.0, 0.0]
        
        # Inner ring (6)
        for i in range(6):
            ang = inner_angles[i] + rot_rad
            centers[i+1] = [r1 * np.cos(ang), r1 * np.sin(ang)]
            
        # Outer ring (5)
        for i in range(5):
            ang = outer_angles[i] + rot_rad
            centers[i+7] = [r2 * np.cos(ang), r2 * np.sin(ang)]
            
        return centers

    def objective(params):
        centers = get_centers(params)
        
        # Check collisions
        min_dist = np.sqrt(3.0)
        viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    viol += (min_dist - dist) ** 2
        
        # Support function
        dots = centers @ normals.T
        max_dot = np.max(dots)
        L_req = (2.0 / np.sqrt(3.0)) * (max_dot + np.sqrt(3.0)/2.0)
        
        return L_req + 100.0 * viol

    # Initial guess: r1 = sqrt(3), r2 = 2*sqrt(3), rot = 0
    x0 = [np.sqrt(3.0), 2.0 * np.sqrt(3.0), 0.0]
    
    best_L = float('inf')
    best_params = None
    
    # Try multiple rotations to escape local minima
    for rot_start in [0, 30, 60, 90, 120, 150]:
        x0_try = [np.sqrt(3.0), 2.0 * np.sqrt(3.0), float(rot_start)]
        try:
            result = minimize(objective, x0_try, method='L-BFGS-B', bounds=[(0.5, 5.0), (0.5, 5.0), (0, 180)])
            if result.success or result.fun < best_L:
                # Verify
                centers = get_centers(result.x)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(centers[i]-centers[j]) < np.sqrt(3.0) - 1e-5:
                            valid = False
                            break
                if valid:
                    dots = centers @ normals.T
                    max_dot = np.max(dots)
                    L_final = (2.0 / np.sqrt(3.0)) * (max_dot + np.sqrt(3.0)/2.0)
                    if L_final < best_L:
                        best_L = L_final
                        best_params = result.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to seed logic or simple grid
        # Re-use seed fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = get_centers(best_params)
    inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_L, 0.0
