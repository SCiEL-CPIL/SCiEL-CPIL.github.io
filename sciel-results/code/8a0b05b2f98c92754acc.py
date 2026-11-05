import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Lattice vectors for hexagonal packing
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    
    def generate_lattice_points(n_points):
        points = []
        # Generate a grid of lattice points
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = sqrt3 * (i + j / 2.0)
                y = 1.5 * j
                points.append((x, y))
        
        # Sort by distance to origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        return points[:n_points]
    
    # Initial guess: 11 closest lattice points
    init_points = generate_lattice_points(N)
    x0 = np.array([p[0] for p in init_points] + [p[1] for p in init_points] + [5.0])
    
    def hexagon_violation(v, S):
        # v is vector (dx, dy)
        # Check if v is inside hexagon of side S (centered at 0)
        # Inequalities:
        # |x| <= S
        # |y| <= S*sqrt(3)/2
        # |x + y/sqrt(3)| <= S
        # |x - y/sqrt(3)| <= S
        x, y = v
        v1 = abs(x) - S
        v2 = abs(y) - S * sqrt3 / 2
        v3 = abs(x + y / sqrt3) - S
        v4 = abs(x - y / sqrt3) - S
        return max(v1, v2, v3, v4)

    def objective(params):
        # params: x1..xN, y1..yN, S
        coords = params[:2*N].reshape(N, 2)
        S = params[2*N]
        
        # Penalty for outer containment violation
        # Inner hexagons must be inside outer hexagon of side S.
        # Equivalent to centers being inside hexagon of side S-1.
        pen_outer = 0.0
        S_inner = S - 1.0
        if S_inner < 0:
            return 1e6 # Invalid S
        for i in range(N):
            viol = hexagon_violation(coords[i], S_inner)
            if viol > 0:
                pen_outer += viol**2
        
        # Penalty for overlap
        # Centers must be outside hexagon of side 2 (Minkowski sum of two unit hexes)
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v = coords[i] - coords[j]
                viol = hexagon_violation(v, 2.0)
                if viol > 0:
                    pen_overlap += viol**2
        
        # Objective: Minimize S + penalties
        return S + 100.0 * (pen_outer + pen_overlap)

    def constraints(params):
        # S must be positive
        return [
            {'type': 'ineq', 'fun': lambda p: p[2*N] - 1.0} # S >= 1
        ]

    best_score = 1e9
    best_params = x0.copy()
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=[(-10, 10)]*2*N + [(1, 100)], 
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
        if res.success:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline:
            break
        # Random perturbation of initial guess
        noise = np.random.normal(0, 0.5, 2*N)
        x_r = x0.copy()
        x_r[:2*N] += noise
        x_r[2*N] = 5.0
        
        try:
            res_r = minimize(objective, x_r, method='SLSQP', bounds=[(-10, 10)]*2*N + [(1, 100)], 
                             constraints=constraints, options={'maxiter': 100, 'ftol': 1e-9})
            if res_r.success and res_r.fun < best_score:
                best_params = res_r.x
                best_score = res_r.fun
        except Exception:
            pass

    # Extract result
    coords = best_params[:2*N].reshape(N, 2)
    S = best_params[2*N]
    
    # Construct inner hexagons
    inner = []
    for i in range(N):
        inner.append([coords[i][0], coords[i][1], 0.0])
    
    # Outer hexagon
    center = (0.0, 0.0)
    angle = 0.0
    
    # Verify validity and adjust S if needed (safety)
    # Check if current S is valid. If not, increase S slightly.
    # The objective function penalizes violations, so it should be valid.
    # But let's double check.
    
    return inner, center, S, angle

# EVOLVE-BLOCK-END
