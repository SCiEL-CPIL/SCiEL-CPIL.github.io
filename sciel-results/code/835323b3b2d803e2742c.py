import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses projection onto the 6 face normals of the outer hexagon.
        
        theta_out: rotation of outer hexagon (radians)
        phis: array of rotations of inner hexagons (radians)
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        # Note: The "angle" parameter in the interface usually defines the orientation.
        # For a regular hexagon, the distance from center to side (apothem) is S * sqrt(3) / 2.
        # The normals are perpendicular to the sides.
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            # We need the maximum absolute projection to ensure containment
            # Since the outer hexagon is centered at origin and symmetric, 
            # and inner hexagons are also centered, we check max(v.n) and min(v.n)
            # Actually, for a regular hexagon centered at origin, the condition is |v.n| <= apothem.
            # So we take max(abs(dots))
            max_proj = max(max_proj, np.max(np.abs(dots)))
        
        # S = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    # All inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    # In radians: 30 deg = pi/6
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)
    
    best_score = objective(x0)
    best_params = x0.copy()
    
    # Phase 1: Multi-start Powell optimization with various starting points
    # Powell is good for this smooth, low-dimensional problem
    n_starts = 15
    # Generate diverse starting points
    np.random.seed(42)
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),  # Analytical
        np.array([0.0] + [0.0] * n_hex),  # All zero
        np.array([np.pi/6] + [0.0] * n_hex),  # Outer at 30, inner at 0
        np.array([0.0] + [np.pi/6] * n_hex),  # Outer at 0, inner at 30
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
    ]
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            # Random perturbations around the best found so far
            x_start = best_params + np.random.uniform(-0.1, 0.1, n_hex + 1)
            # Wrap angles to [0, pi/3) for consistency, though objective is periodic
            # Actually, let's just use random angles in [0, pi/3]
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)
        
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Coordinate descent polish
    # Refine each parameter individually with a fine grid search
    if time.time() < deadline - 15:
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 10:
                    break
                
                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                
                # Golden section search around current value with decreasing step sizes
                # Since the function is periodic with period pi/3, we can search locally
                # Try small steps
                for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4,
                             1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-16:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue
                
                if best_local < best_score - 1e-16:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True
            
            if not improved:
                break

    # Phase 3: Final Nelder-Mead polish on the best parameters
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    # Add a tiny margin to the side length to ensure floating point safety
    # The evaluator checks if inner hexagons are strictly inside.
    # If we report exactly the minimum, floating point errors might cause a violation.
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
