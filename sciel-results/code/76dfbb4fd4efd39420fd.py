import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _separation_violations(centers, normals):
    """Return a list of violation values for each pair.
    Violation > 0 means overlap.
    """
    sqrt3 = np.sqrt(3.0)
    n = len(centers)
    violations = []
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            d_dots = np.abs(diff @ normals.T)
            max_d_dot = np.max(d_dots)
            viol = sqrt3 - max_d_dot
            violations.append(viol)
    return violations

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # Radial Slicing: 1 center, 5 inner ring, 6 outer ring
    # Parameters: [r_inner, r_outer, rot_inner, rot_outer]
    
    def build_centers(params):
        r_in, r_out, rot_in, rot_out = params
        rot_in_rad = np.deg2rad(rot_in)
        rot_out_rad = np.deg2rad(rot_out)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring: 5 hexagons
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_in_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
            
        # Outer ring: 6 hexagons
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_out_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
            
        return centers
    
    def objective(params):
        centers = build_centers(params)
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        
        # Calculate max violation
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                d_dots = np.abs(diff @ normals.T)
                max_d_dot = np.max(d_dots)
                viol = sqrt3 - max_d_dot
                if viol > max_viol:
                    max_viol = viol
                    
        # Penalize overlap linearly with a very large factor to force feasibility
        penalty = 1000.0 * max_viol
        return L_req + penalty
    
    # Try multiple random starts with tighter bounds
    # We will use a grid of rotations to ensure we cover the symmetry
    # and randomize the radii slightly around the expected touching distance.
    
    for seed in range(500):
        np.random.seed(seed)
        # Tighter initial guesses
        # r_in should be ~1.732 (touching center)
        # r_out should be ~3.464 (touching inner ring, roughly)
        r_in = np.random.uniform(1.6, 2.0)
        r_out = np.random.uniform(3.2, 4.0)
        rot_in = np.random.uniform(0, 72)
        rot_out = np.random.uniform(0, 60)
        
        x0 = np.array([r_in, r_out, rot_in, rot_out])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            params = result.x
            
            # Verify
            centers = build_centers(params)
            max_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    diff = centers[i] - centers[j]
                    d_dots = np.abs(diff @ normals.T)
                    max_d_dot = np.max(d_dots)
                    viol = sqrt3 - max_d_dot
                    if viol > max_viol:
                        max_viol = viol
                        
            if max_viol > 1e-6:
                continue
            
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
                
        except Exception:
            continue
            
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_inner, (0.0, 0.0), best_L, 0.0
