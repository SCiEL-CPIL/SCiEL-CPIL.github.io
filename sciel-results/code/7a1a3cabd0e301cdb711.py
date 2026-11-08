import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)

def get_hex_vertices(center):
    # Returns 6 vertices of a unit hexagon centered at `center`, aligned with axes (pointy-top)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    vs = np.column_stack([np.cos(angles), np.sin(angles)])
    return center + vs

def calculate_L(positions):
    # positions: (N, 2) array
    # Calculate max projection onto the 6 normals of a pointy-top hexagon
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    norm_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    
    max_proj = 0.0
    for i in range(N):
        c = positions[i]
        verts = get_hex_vertices(c)
        # Project all vertices onto all normals
        # verts: (6, 2), normals: (6, 2)
        # We need max of n . v for all v, n
        # Since hexagon is convex and symmetric, max projection is max over vertices
        dots = np.dot(verts, normals.T) # (6, 6)
        max_abs = np.max(np.abs(dots))
        if max_abs > max_proj:
            max_proj = max_abs
            
    L = 2.0 * max_proj / SQRT3
    return L

def distance_constraint(x):
    # x is flattened (N, 2)
    pos = x.reshape(N, 2)
    # Constraints: ||p_i - p_j||^2 >= 3 for all i < j
    # To use SLSQP, we define c_k >= 0
    # c_k = ||p_i - p_j||^2 - 3
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            d2 = np.sum((pos[i] - pos[j])**2)
            cons.append(d2 - 3.0)
    return cons

def construct():
    # Initial guess: Hexagonal lattice
    # Generate a cluster of 11 hexagons
    # Center (0,0)
    # 6 neighbors
    # 4 more around
    init_pos = []
    # Center
    init_pos.append([0.0, 0.0])
    # 6 neighbors (distance sqrt(3))
    for k in range(6):
        angle = np.deg2rad(30 + 60*k) # Flat-to-flat contact angle
        # Actually, neighbors in hex lattice are at distance sqrt(3)
        # Directions: 0, 60, 120... or 30, 90...?
        # Vertices at 0, 60...
        # Side midpoints at 30, 90...
        # Neighbor centers are at distance sqrt(3) in direction of side midpoints?
        # Yes, if aligned, the distance between centers of touching hexes is sqrt(3).
        # The direction is perpendicular to the shared side.
        # Shared side is horizontal? No, if pointy top, sides are at 30, 90...
        # So normals are 30, 90...
        # So neighbors are at angles 30, 90, 150, 210, 270, 330.
        x = SQRT3 * np.cos(angle)
        y = SQRT3 * np.sin(angle)
        init_pos.append([x, y])
        
    # We have 7 hexagons. Need 4 more.
    # Place them further out.
    # Distance from center for 2nd shell is roughly 2*sqrt(3) or more.
    # Let's just add 4 random points further out or specific lattice points.
    # Lattice points at distance 2*sqrt(3)?
    # Let's use a simpler grid and let the optimizer fix it.
    # Or use known tight packing for 11.
    # A 1-2-3-2-1 pyramid?
    # Let's just append 4 points at distance 2*sqrt(3) * 1.1
    for k in range(4):
        angle = np.deg2rad(30 + 90*k) # Spread out
        x = 2.2 * SQRT3 * np.cos(angle)
        y = 2.2 * SQRT3 * np.sin(angle)
        init_pos.append([x, y])
        
    x0 = np.array(init_pos).flatten()
    
    # Constraints
    # SLSQP requires constraints as functions returning >= 0
    # We have N(N-1)/2 distance constraints
    # This might be slow for SLSQP with many constraints.
    # N=11 -> 55 constraints. Acceptable.
    
    # Objective function
    def objective(x):
        pos = x.reshape(N, 2)
        return calculate_L(pos)
        
    # Gradient (optional, speeds up)
    def jac(x):
        # Numerical gradient is fine for small N
        return None # Let SLSQP use finite diff if None
        
    # Define constraints
    # SLSQP takes a list of dicts {'type': 'ineq', 'fun': func}
    # func(x) >= 0
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def con(x):
                    pos = x.reshape(N, 2)
                    d2 = np.sum((pos[i] - pos[j])**2)
                    return d2 - 3.0
                return con
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Bounds
    # Keep positions within a reasonable range, e.g., [-10, 10]
    bounds = [(-10, 10)] * (2 * N)
    
    # Run optimization
    # Use multiple restarts if possible?
    # Time limit is 150s.
    # SLSQP on 22 vars with 55 constraints should be fast.
    
    best_x = x0
    best_val = calculate_L(x0.reshape(N, 2))
    
    start_time = time.time()
    
    # Try one main run
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success:
            val = calculate_L(res.x.reshape(N, 2))
            if val < best_val:
                best_x = res.x
                best_val = val
    except Exception:
        pass
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Try a few random restarts with perturbed initial guesses
        for _ in range(10):
            x0_rand = x0 + np.random.normal(0, 0.5, size=2*N)
            try:
                res = minimize(objective, x0_rand, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
                if res.success:
                    val = calculate_L(res.x.reshape(N, 2))
                    if val < best_val:
                        best_x = res.x
                        best_val = val
            except Exception:
                pass
            if time.time() - start_time > 110:
                break

    best_pos = best_x.reshape(N, 2)
    L = calculate_L(best_pos)
    
    # Ensure L is not NaN
    if np.isnan(L):
        L = 40.0
        best_pos = np.array(init_pos)
        
    # Construct inner list
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(L), 0.0

if __name__ == "__main__":
    construct()
