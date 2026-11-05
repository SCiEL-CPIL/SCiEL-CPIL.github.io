import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n_hex = 11
    
    # Base vertices of unit hexagon (pointy-right, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    # Pre-compute rotation matrices? No, rotations are variables.
    
    def compute_L(params):
        """
        params: 
        [0:2] center 0
        [2:4] center 1
        ...
        [20:22] center 10
        [22:33] rotations phi_0 ... phi_10
        [33] outer rotation theta_out
        """
        centers = params[:2*n_hex].reshape(n_hex, 2)
        phis = params[2*n_hex:2*n_hex+n_hex]
        theta_out = params[-1]
        
        # Outer hexagon normals
        # Normals at 30, 90, 150, 210, 270, 330 degrees + theta_out
        # Wait, a regular hexagon has normals at 30 + k*60.
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = 0.0
        
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(np.abs(dots)))
            
        # Side length L = 2 * max_proj / sqrt(3)
        L = 2.0 * max_proj / np.sqrt(3)
        
        # Check overlaps
        # Min distance between centers of two unit hexagons is sqrt(3)
        # Penalty for violation
        penalty = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
                    
        return L + 1000.0 * penalty

    # Define bounds for Differential Evolution
    # Centers: assume they fit in a box of side ~12 (since L~10, radius ~5)
    # Let's use a generous bound
    center_bound = 6.0
    angle_bound = np.pi/3 # 60 degrees, due to symmetry of hexagon
    
    bounds = []
    for _ in range(n_hex):
        bounds.append((-center_bound, center_bound))
        bounds.append((-center_bound, center_bound))
    for _ in range(n_hex):
        bounds.append((0, angle_bound))
    bounds.append((0, angle_bound)) # outer rotation
    
    # Initial guess: Flower pattern
    # 1 center at (0,0)
    # 6 at distance sqrt(3)
    # 4 at distance 3?
    # Let's just use a random start or the flower pattern as a seed for DE?
    # DE generates its own population.
    
    # Time check
    if time.time() > deadline - 10:
        # Fallback to simple configuration
        centers = np.zeros((n_hex, 2))
        phis = np.zeros(n_hex)
        theta_out = 0
        params = np.concatenate([centers.ravel(), phis, [theta_out]])
        L = compute_L(params)
        # Extract result
        inner = []
        for i in range(n_hex):
            inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
        return inner, (0.0, 0.0), L, np.rad2deg(theta_out)

    # Run Differential Evolution
    try:
        result = differential_evolution(compute_L, bounds, seed=42, maxiter=100, tol=1e-6, polish=True, workers=1)
    except Exception:
        # Fallback
        centers = np.zeros((n_hex, 2))
        phis = np.zeros(n_hex)
        theta_out = 0
        params = np.concatenate([centers.ravel(), phis, [theta_out]])
        L = compute_L(params)
        inner = []
        for i in range(n_hex):
            inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
        return inner, (0.0, 0.0), L, np.rad2deg(theta_out)
        
    if result.success and result.fun < 100:
        best_params = result.x
    else:
        # If DE failed, try a simple local optimization from a good guess
        # Flower pattern guess
        centers_guess = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3)
        for k in range(6):
            angle = np.deg2rad(k * 60)
            centers_guess = np.vstack([centers_guess, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0
        for k in range(4):
            angle = np.deg2rad(30 + k * 60)
            centers_guess = np.vstack([centers_guess, [r2 * np.cos(angle), r2 * np.sin(angle)]])
            
        phis_guess = np.zeros(n_hex)
        theta_out_guess = 0.0
        best_params = np.concatenate([centers_guess.ravel(), phis_guess, [theta_out_guess]])
        
        # Local refine
        try:
            res = minimize(compute_L, best_params, method='L-BFGS-B', options={'maxiter': 500})
            if res.fun < compute_L(best_params):
                best_params = res.x
        except Exception:
            pass

    # Extract solution
    centers = best_params[:2*n_hex].reshape(n_hex, 2)
    phis = best_params[2*n_hex:2*n_hex+n_hex]
    theta_out = best_params[-1]
    
    # Calculate final L without penalty to report
    # Re-compute L strictly
    angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    max_proj = 0.0
    for i in range(n_hex):
        phi = phis[i]
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + centers[i]
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(np.abs(dots)))
    L = 2.0 * max_proj / np.sqrt(3)
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
        
    # Add tiny margin
    side = L + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
