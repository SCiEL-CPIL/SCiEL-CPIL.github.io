import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices_local(angle_deg):
    """Vertices of a unit hexagon rotated by angle_deg."""
    angles = np.deg2rad(angle_deg) + np.arange(0, 360, 60) * np.pi / 180
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _point_in_polygon(p, poly):
    """Check if point p is inside convex polygon poly (N x 2)."""
    # poly is Nx2
    # Check cross products
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Normal points inward if polygon is CCW
        # Let's assume CCW.
        # Check if p is on the left of edge
        v = p - p1
        cross = edge[0]*v[1] - edge[1]*v[0]
        if cross < -1e-9:
            return False, 0.0 # Outside
    return True, 0.0

def _dist_point_to_polygon(p, poly):
    """Distance from point p to convex polygon poly. Negative if inside."""
    inside, _ = _point_in_polygon(p, poly)
    if inside:
        # Find min distance to edges
        min_d = np.inf
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            # Project p onto segment p1-p2
            v = p2 - p1
            w = p - p1
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p - p1)
            else:
                c2 = np.dot(w, v2) if False else np.dot(v, v)
                if c2 <= 0:
                    d = np.linalg.norm(p - p2)
                else:
                    t = c1 / c2
                    proj = p1 + t * v
                    d = np.linalg.norm(p - proj)
            min_d = min(min_d, d)
        return -min_d # Negative distance indicates depth
    else:
        min_d = np.inf
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            v = p2 - p1
            w = p - p1
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p - p1)
            else:
                c2 = np.dot(v, v)
                if c2 <= 0:
                    d = np.linalg.norm(p - p2)
                else:
                    t = c1 / c2
                    proj = p1 + t * v
                    d = np.linalg.norm(p - proj)
            min_d = min(min_d, d)
        return min_d

def _pairwise_penalty(hex1, hex2):
    """Penalty for overlap between two hexagons."""
    # Check vertices of hex1 against hex2
    min_depth = 0.0
    for v in hex1:
        d = _dist_point_to_polygon(v, hex2)
        if d < min_depth:
            min_depth = d
    for v in hex2:
        d = _dist_point_to_polygon(v, hex1)
        if d < min_depth:
            min_depth = d
    if min_depth < 0:
        return min_depth**2
    return 0.0

def construct():
    start_time = time.time()
    best_side = np.inf
    best_params = None
    
    # Initial guesses
    # Mode 0: Hex lattice
    # Mode 1: Linear
    # Mode 2: Circle
    
    def run_optimization(x0, max_iter=200):
        nonlocal best_side, best_params
        
        def objective(params):
            # Unpack
            # 0..32: xi, yi, theta_i (3 per hex)
            # 33: X, 34: Y, 35: Phi, 36: S
            xys = params[:33].reshape(11, 3)
            X, Y, Phi, S = params[33], params[34], params[35], params[36]
            
            if S <= 0:
                return 1e6
            
            # Outer hexagon
            outer_angles = np.deg2rad(Phi) + np.arange(0, 360, 60) * np.pi / 180
            outer_poly = np.column_stack((S * np.cos(outer_angles), S * np.sin(outer_angles))) + np.array([X, Y])
            
            penalty = 0.0
            inner_polys = []
            
            for i in range(11):
                xi, yi, theta_i = xys[i]
                inner_angles = np.deg2rad(theta_i) + np.arange(0, 360, 60) * np.pi / 180
                inner_poly = np.column_stack((np.cos(inner_angles), np.sin(inner_angles))) + np.array([xi, yi])
                inner_polys.append(inner_poly)
                
                # Boundary check
                for v in inner_poly:
                    d = _dist_point_to_polygon(v, outer_poly)
                    if d < 0:
                        penalty += d**2
            
            # Overlap check
            for i in range(11):
                for j in range(i+1, 11):
                    penalty += _pairwise_penalty(inner_polys[i], inner_polys[j])
            
            return S + 100.0 * penalty
        
        # Bounds
        # x, y can be anywhere reasonable, say [-20, 20]
        # theta in [0, 360]
        # X, Y in [-20, 20]
        # Phi in [0, 360]
        # S in [1, 20]
        bounds = []
        for _ in range(11):
            bounds.append((-20, 20))
            bounds.append((-20, 20))
            bounds.append((0, 360))
        bounds.append((-20, 20)) # X
        bounds.append((-20, 20)) # Y
        bounds.append((0, 360))  # Phi
        bounds.append((1, 20))   # S
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-9})
        
        if res.success or res.fun < 1e6:
            S_val = res.x[36]
            pen = (res.fun - S_val) / 100.0
            if pen < 1e-6: # Valid solution
                if S_val < best_side:
                    best_side = S_val
                    best_params = res.x.copy()
            
        # Polish with SLSQP if time permits
        if time.time() - start_time < 100:
            try:
                def slsqp_objective(params):
                    # Same as above
                    xys = params[:33].reshape(11, 3)
                    X, Y, Phi, S = params[33], params[34], params[35], params[36]
                    if S <= 0: return 1e6
                    outer_angles = np.deg2rad(Phi) + np.arange(0, 360, 60) * np.pi / 180
                    outer_poly = np.column_stack((S * np.cos(outer_angles), S * np.sin(outer_angles))) + np.array([X, Y])
                    penalty = 0.0
                    inner_polys = []
                    for i in range(11):
                        xi, yi, theta_i = xys[i]
                        inner_angles = np.deg2rad(theta_i) + np.arange(0, 360, 60) * np.pi / 180
                        inner_poly = np.column_stack((np.cos(inner_angles), np.sin(inner_angles))) + np.array([xi, yi])
                        inner_polys.append(inner_poly)
                        for v in inner_poly:
                            d = _dist_point_to_polygon(v, outer_poly)
                            if d < 0: penalty += d**2
                    for i in range(11):
                        for j in range(i+1, 11):
                            penalty += _pairwise_penalty(inner_polys[i], inner_polys[j])
                    return S + 100.0 * penalty
                
                res2 = minimize(slsqp_objective, res.x, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
                if res2.success:
                    S_val = res2.x[36]
                    pen = (res2.fun - S_val) / 100.0
                    if pen < 1e-6:
                        if S_val < best_side:
                            best_side = S_val
                            best_params = res2.x.copy()
            except Exception:
                pass

    # Generate initial guesses
    inits = []
    
    # Mode 0: Hex lattice
    # Center at 0,0
    # 1 center, 6 around, 4 more?
    # Just use a grid
    x0 = np.zeros(37)
    x0[36] = 10.0 # Initial S
    # Place in a line for simplicity in init, optimizer will move them
    for i in range(11):
        x0[i*3] = i * 2.0
        x0[i*3+1] = 0.0
        x0[i*3+2] = 0.0
    inits.append(x0)
    
    # Mode 1: Hexagonal cluster
    # 1 at center, 6 around, 4 more
    # Center (0,0)
    # Radius 2 for neighbors
    pos = [(0,0)]
    for k in range(6):
        ang = k * np.pi / 3
        pos.append((2*np.cos(ang), 2*np.sin(ang)))
    # Add 4 more further out
    for k in range(4):
        ang = (k + 0.5) * np.pi / 3
        pos.append((4*np.cos(ang), 4*np.sin(ang)))
        
    x0 = np.zeros(37)
    x0[36] = 10.0
    for i in range(11):
        x0[i*3] = pos[i][0]
        x0[i*3+1] = pos[i][1]
        x0[i*3+2] = 0.0
    inits.append(x0)
    
    # Mode 2: Random
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, 37)
    x0[36] = 10.0
    inits.append(x0)
    
    # Try different outer rotations
    rotations = [0, 30, 60]
    
    for rot in rotations:
        for x0 in inits:
            x0[35] = rot # Phi
            if time.time() - start_time > 110:
                break
            run_optimization(x0)
        if time.time() - start_time > 110:
            break
            
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Extract results
    xys = best_params[:33].reshape(11, 3)
    X, Y, Phi, S = best_params[33], best_params[34], best_params[35], best_params[36]
    
    inner = []
    for i in range(11):
        inner.append([xys[i][0], xys[i][1], xys[i][2]])
        
    return inner, (X, Y), S, Phi

