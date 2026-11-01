import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    if pts is None or len(pts) != N:
        return np.inf
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_d3d_points(params):
    """
    Generate 14 points using D3d symmetry.
    params: [r1, r2, z2, r3, z3]
    r1: z-coordinate of the 2 points on the C3 axis.
    r2, z2: x, z coordinates of the seed for the first 6-point orbit (in sigma_d plane).
    r3, z3: x, z coordinates of the seed for the second 6-point orbit (in sigma_d plane).
    
    D3d operations on (x, y, z):
    The group D3d consists of:
    E: (x, y, z)
    C3: (x cos120 - y sin120, x sin120 + y cos120, z)
    C3^2: (x cos240 - y sin240, x sin240 + y cos240, z)
    C2' (3 axes): 
       One C2 is x-axis? No, in D3d, C2 axes are perpendicular to C3 (z).
       Let's align one C2' with the x-axis.
       C2_x: (-x, -y, z) ? No, C2 rotation around x: (-x, -y, z) is 180 deg? 
       Rotation 180 around x: (x, -y, -z).
       Wait, D3d has 3 C2 axes.
       Let's use the standard matrices.
       
       D3d is isomorphic to D3 x {E, i}? No.
       D3d = D3 + i*D3.
       D3 operations on (x,y,z):
       E: (x,y,z)
       C3: (-0.5x - 0.866y, 0.866x - 0.5y, z)
       C3^2: (-0.5x + 0.866y, -0.866x - 0.5y, z)
       C2_1: (-x, y, -z) ? (Rotation 180 around axis in xy plane)
       
       Let's explicitly construct the 12 rotation matrices for D3d.
       Axis C3 is z.
       C2 axes are in xy plane, at 0, 120, 240 degrees?
       Let C2_1 be along x-axis.
       R_C2_1 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
       R_C2_2 = rotation of R_C2_1 by 120 deg around z.
       R_C2_3 = rotation of R_C2_1 by 240 deg around z.
       
       Plus inversion i: (-x, -y, -z).
       Plus S6: i * C3.
       
       Let's generate the group elements.
    """
    r1, r2, z2, r3, z3 = params
    
    # Define D3d group matrices
    # C3 rotation around z
    c3 = np.array([
        [-0.5, -np.sqrt(3)/2, 0],
        [np.sqrt(3)/2, -0.5, 0],
        [0, 0, 1]
    ])
    c3_2 = c3 @ c3
    I3 = np.eye(3)
    
    # C2 rotation around x-axis
    c2_1 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Generate all 6 proper rotations (D3)
    proper = [I3, c3, c3_2, c2_1, c3 @ c2_1, c3_2 @ c2_1]
    
    # Generate all 12 operations (D3d = D3 union i*D3)
    inv = -I3
    ops = []
    for R in proper:
        ops.append(R)
        ops.append(inv @ R)
        
    # Verify unique ops
    # ops should have 12 unique matrices
    
    points = []
    
    # Orbit 1: Size 2. Seed (0, 0, r1)
    # Stabilizer is large, orbit size should be 2.
    # Let's just apply all ops to seed and unique them.
    seed1 = np.array([0.0, 0.0, r1])
    orb1 = []
    for R in ops:
        p = R @ seed1
        # Check if p is in orb1 (with tolerance)
        is_dup = False
        for q in orb1:
            if np.linalg.norm(p - q) < 1e-9:
                is_dup = True
                break
        if not is_dup:
            orb1.append(p)
            
    # Orbit 2: Size 6. Seed (r2, 0, z2)
    # This seed is in the xz plane (sigma_d plane?).
    # If it's generic in the plane, orbit size is 6.
    seed2 = np.array([r2, 0.0, z2])
    orb2 = []
    for R in ops:
        p = R @ seed2
        is_dup = False
        for q in orb2:
            if np.linalg.norm(p - q) < 1e-9:
                is_dup = True
                break
        # Also check against orb1 to ensure distinctness if they overlap?
        # The problem requires 14 DISTINCT points.
        # If orbits overlap, we have fewer than 14 points.
        # We must ensure seeds are chosen such that orbits are disjoint.
        # For now, add to list, we will check distinctness later.
        if not is_dup:
            orb2.append(p)
            
    # Orbit 3: Size 6. Seed (r3, 0, z3)
    seed3 = np.array([r3, 0.0, z3])
    orb3 = []
    for R in ops:
        p = R @ seed3
        is_dup = False
        for q in orb3:
            if np.linalg.norm(p - q) < 1e-9:
                is_dup = True
                break
        if not is_dup:
            orb3.append(p)
            
    all_pts = np.array(orb1 + orb2 + orb3)
    
    # Check distinctness
    if len(all_pts) < N:
        return None
        
    # Ensure exactly 14 points
    # If orbits overlapped, we might have < 14.
    # If they didn't, we have 2 + 6 + 6 = 14.
    
    return all_pts

def _objective(params):
    """
    Objective function for optimization.
    """
    pts = _generate_d3d_points(params)
    if pts is None or len(pts) != N:
        return 1e10
        
    # Check for duplicate points
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        return 1e10 + d.min()
        
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return 1e10
        
    ratio = mx / mn
    return ratio ** 2

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
        
    # Parameters: r1, r2, z2, r3, z3
    # Scale invariance: fix r1 = 1.0 to reduce dimensionality?
    # Actually, let's keep r1 variable but add a penalty or just let optimizer find scale.
    # Since objective is scale invariant, the landscape is flat along the scaling direction.
    # This can cause issues with gradient methods.
    # Better to fix one parameter, e.g., r1 = 1.0.
    
    # If we fix r1=1, we have 4 parameters: r2, z2, r3, z3.
    # Let's try that.
    
    def obj_4d(params_4):
        r2, z2, r3, z3 = params_4
        full_params = [1.0, r2, z2, r3, z3]
        return _objective(full_params)
        
    # Initial guesses
    # We want points to be well spread out.
    # r1 = 1.
    # r2, z2, r3, z3 should be of similar magnitude.
    
    rng = np.random.default_rng(42)
    
    best_score = np.inf
    best_params = None
    best_pts = None
    
    # Grid search / Multi-start Nelder-Mead
    # Since the space is small (4D), we can afford many starts.
    
    starts = []
    # Heuristic starts
    starts.append([1.0, 0.0, 1.0, 1.0, 0.0]) # r1=1, r2=1, z2=0, r3=1, z3=0 (might collide)
    starts.append([1.0, 1.5, 0.5, 2.0, 1.0])
    starts.append([1.0, 0.5, 1.5, 1.5, 0.5])
    starts.append([1.0, 2.0, -1.0, 1.0, 2.0])
    
    # Random starts
    for _ in range(50):
        r2 = rng.uniform(0.5, 3.0)
        z2 = rng.uniform(-2.0, 2.0)
        r3 = rng.uniform(0.5, 3.0)
        z3 = rng.uniform(-2.0, 2.0)
        starts.append([1.0, r2, z2, r3, z3])
        
    time_limit_per_opt = 1.5 # seconds
    
    for start in starts:
        if time.time() - _t0 > DEADLINE - 5:
            break
            
        r2, z2, r3, z3 = start[1:]
        x0 = np.array([r2, z2, r3, z3])
        
        try:
            res = minimize(
                obj_4d, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.success or res.fun < best_score:
                params_4 = res.x
                full_params = [1.0] + params_4.tolist()
                pts = _generate_d3d_points(full_params)
                
                if pts is not None and len(pts) == N:
                    d = cdist(pts, pts)
                    np.fill_diagonal(d, np.inf)
                    if d.min() > 1e-6:
                        score = _true_score(pts)
                        if score < best_score:
                            best_score = score
                            best_params = full_params
                            best_pts = pts.copy()
                            
        except Exception:
            continue
            
    # If Nelder-Mead failed or didn't find anything, try a coarse grid
    if best_pts is None:
        grid_r2 = np.linspace(0.5, 2.5, 5)
        grid_z2 = np.linspace(-1.5, 1.5, 5)
        grid_r3 = np.linspace(0.5, 2.5, 5)
        grid_z3 = np.linspace(-1.5, 1.5, 5)
        
        for r2 in grid_r2:
            if time.time() - _t0 > DEADLINE - 5:
                break
            for z2 in grid_z2:
                for r3 in grid_r3:
                    for z3 in grid_z3:
                        params = [1.0, r2, z2, r3, z3]
                        pts = _generate_d3d_points(params)
                        if pts is not None and len(pts) == N:
                            d = cdist(pts, pts)
                            np.fill_diagonal(d, np.inf)
                            if d.min() > 1e-6:
                                score = _true_score(pts)
                                if score < best_score:
                                    best_score = score
                                    best_params = params
                                    best_pts = pts.copy()
                                    
    # Fallback to Fibonacci sphere if symmetry approach fails
    if best_pts is None:
        i = np.arange(N)
        phi = np.arccos(1 - 2*(i+0.5)/N)
        theta = np.pi*(1+5**0.5)*i
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        best_pts -= best_pts.mean(axis=0)
        
    # Final sanity check
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        # Perturb to make distinct
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
