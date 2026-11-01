"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute constants
    SQRT3 = np.sqrt(3)
    
    # Helper to check if a point is inside a hexagon of side L centered at origin (flat-top)
    def point_in_hexagon(x, y, L):
        # Flat top hexagon inequalities:
        # |x| <= L
        # |y| <= sqrt(3)/2 * L
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L
        if abs(x) > L: return False
        if abs(y) > SQRT3/2 * L: return False
        if abs(SQRT3/2 * x + 0.5 * y) > SQRT3/2 * L: return False
        return True

    def hexagon_containment_penalty(x, y, L):
        """
        Returns penalty for a unit hexagon at (x,y) being outside a hexagon of side L at (0,0).
        We check all 6 vertices of the inner hexagon.
        """
        # Vertices of unit hexagon at (0,0)
        verts = [
            (1.0, 0.0),
            (0.5, SQRT3/2),
            (-0.5, SQRT3/2),
            (-1.0, 0.0),
            (-0.5, -SQRT3/2),
            (0.5, -SQRT3/2)
        ]
        
        penalty = 0.0
        for vx, vy in verts:
            px = x + vx
            py = y + vy
            # Calculate violation depth
            v1 = abs(px) - L
            v2 = abs(py) - SQRT3/2 * L
            v3 = abs(SQRT3/2 * px + 0.5 * py) - SQRT3/2 * L
            
            depth = max(0, v1, v2, v3)
            penalty += depth**2
        return penalty

    def hexagon_overlap_penalty(x1, y1, x2, y2):
        """
        Returns penalty for two unit hexagons at (x1,y1) and (x2,y2) overlapping.
        Uses Minkowski sum logic.
        The difference vector d = (x1-x2, y1-y2) must be outside the hexagon of side 2.
        """
        dx = x1 - x2
        dy = y1 - y2
        
        s1 = 2.0 - abs(dx)
        s2 = SQRT3 - abs(dy)
        s3 = SQRT3 - abs(SQRT3/2 * dx + 0.5 * dy)
        
        depth = min(s1, s2, s3)
        
        if depth > 0:
            return depth**2
        return 0.0

    def objective(params, L):
        """
        params: array of 2*N floats (x1, y1, ..., xN, yN)
        L: outer hexagon side length
        """
        penalty = 0.0
        # Containment
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            penalty += hexagon_containment_penalty(x, y, L)
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                x1 = params[2*i]
                y1 = params[2*i+1]
                x2 = params[2*j]
                y2 = params[2*j+1]
                penalty += hexagon_overlap_penalty(x1, y1, x2, y2)
                
        return penalty

    def is_valid_config(inner, L):
        """
        Hard check for validity.
        inner: list of (x, y, angle)
        L: side length
        """
        # Check containment
        for i in range(N):
            x, y, _ = inner[i]
            verts = [
                (1.0, 0.0), (0.5, SQRT3/2), (-0.5, SQRT3/2),
                (-1.0, 0.0), (-0.5, -SQRT3/2), (0.5, -SQRT3/2)
            ]
            for vx, vy in verts:
                if not point_in_hexagon(x+vx, y+vy, L):
                    return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, _ = inner[i]
                x2, y2, _ = inner[j]
                dx = x1 - x2
                dy = y1 - y2
                if (abs(dx) <= 2.0 and 
                    abs(dy) <= SQRT3 and 
                    abs(SQRT3/2 * dx + 0.5 * dy) <= SQRT3):
                    return False
        return True

    # Generate initial lattice configuration
    def get_lattice_positions():
        positions = []
        positions.append((0, 0))
        for i in range(6):
            coords = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
            q, r = coords[i]
            x = SQRT3 * q + SQRT3/2 * r
            y = 1.5 * r
            positions.append((x, y))
            
        ring2_coords = [(2,0), (2,-1), (1,-2), (0,-2), (-1,-2)]
        for q, r in ring2_coords:
            x = SQRT3 * q + SQRT3/2 * r
            y = 1.5 * r
            positions.append((x, y))
            
        return positions

    # New approach: Joint optimization of positions AND outer hexagon side length
    # This allows the optimizer to find the minimal L that works
    
    def joint_objective(params):
        """
        params: array of 2*N + 1 floats (x1, y1, ..., xN, yN, L)
        L is the outer hexagon side length (last parameter)
        """
        L = max(params[-1], 2.5)  # Ensure L is at least minimum possible
        pos_params = params[:-1]
        
        penalty = 0.0
        # Containment
        for i in range(N):
            x = pos_params[2*i]
            y = pos_params[2*i+1]
            penalty += hexagon_containment_penalty(x, y, L)
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                x1 = pos_params[2*i]
                y1 = pos_params[2*i+1]
                x2 = pos_params[2*j]
                y2 = pos_params[2*j+1]
                penalty += hexagon_overlap_penalty(x1, y1, x2, y2)
                
        # Add a small penalty for large L to encourage minimization
        penalty += 0.001 * max(0, L - 4.0)**2
        
        return penalty

    # Try multiple initial configurations
    initial_configs = []
    
    # Configuration 1: Standard hexagonal lattice
    initial_configs.append(get_lattice_positions())
    
    # Configuration 2: Tighter packing - shift ring 2 inward
    def get_tight_packing():
        positions = []
        positions.append((0, 0))
        for i in range(6):
            coords = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
            q, r = coords[i]
            x = SQRT3 * q + SQRT3/2 * r
            y = 1.5 * r
            positions.append((x, y))
        
        # Tighter ring 2
        ring2_tight = [(1.8,0), (1.8,-0.9), (0.9,-1.8), (0,-1.8), (-0.9,-1.8)]
        for x, y in ring2_tight:
            positions.append((x, y))
        return positions
    
    initial_configs.append(get_tight_packing())
    
    # Configuration 3: Random perturbation of lattice
    def get_random_perturbed():
        base = get_lattice_positions()
        positions = []
        for x, y in base:
            positions.append((x + random.gauss(0, 0.1), y + random.gauss(0, 0.1)))
        return positions
    
    initial_configs.append(get_random_perturbed())
    
    best_L = 10.0
    best_params = None
    best_score = float('inf')
    
    # Try each initial configuration with L-BFGS-B
    for config_idx, initial_pos in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        params0 = np.array([p[0] for p in initial_pos] + [p[1] for p in initial_pos] + [5.0])
        
        # Set bounds: positions in [-6, 6], L in [2.5, 10]
        bounds = [(-6, 6)] * (2*N) + [(2.5, 10)]
        
        try:
            result = minimize(joint_objective, params0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                best_L = result.x[-1]
                
        except Exception as e:
            pass
    
    # Also try Nelder-Mead on the joint problem for robustness
    for config_idx, initial_pos in enumerate(initial_configs[:2]):
        if time.time() > deadline - 10:
            break
        
        params0 = np.array([p[0] for p in initial_pos] + [p[1] for p in initial_pos] + [5.0])
        
        try:
            result = minimize(joint_objective, params0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                best_L = result.x[-1]
                
        except Exception as e:
            pass
    
    # If we still don't have a good result, use the lattice with a reasonable L
    if best_params is None:
        initial_pos = get_lattice_positions()
        best_params = np.array([p[0] for p in initial_pos] + [p[1] for p in initial_pos] + [5.0])
        best_L = 5.0

    inner = [[float(best_params[2*i]), float(best_params[2*i+1]), 0.0] for i in range(N)]
    center = (0.0, 0.0)
    side = float(best_L)
    angle = 0.0
    
    # Final validation - if invalid, increase L until valid
    if not is_valid_config(inner, side):
        # Try increasing L
        for test_L in [side + 0.5, side + 1.0, side + 2.0, 8.0, 10.0]:
            if is_valid_config(inner, test_L):
                side = test_L
                break
        else:
            # Fallback: spread out more
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            side = 40.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
