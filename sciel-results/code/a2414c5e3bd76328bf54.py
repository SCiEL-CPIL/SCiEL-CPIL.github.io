"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 11
TIME_LIMIT = 120.0  # safety margin before 150s kill

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    angles = theta + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _softmin(x, beta=10.0):
    """Approximate minimum using log-sum-exp trick."""
    # Handle potential overflow by subtracting max
    x = np.asarray(x)
    max_x = np.max(x)
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * (x - max_x)))) - max_x / beta * 0 # Wait, standard formula: -1/beta * log(sum(exp(-beta*x)))
    # Let's use the numerically stable version:
    # min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # To avoid overflow: let m = max(x). sum(exp(-beta*(x-m))) = exp(-beta*m) * sum(exp(-beta*x)).
    # log(sum(exp(-beta*x))) = -beta*m + log(sum(exp(-beta*(x-m))))
    # So -1/beta * log(...) = m - 1/beta * log(sum(exp(-beta*(x-m))))
    # This approximates m (max), not min.
    # For min, we use: min(x) approx 1/beta * log(sum(exp(beta*x)))? No.
    # min(x) = -max(-x).
    # So softmin(x) = -softmax(-x).
    # softmax(y) = y_max + 1/beta * log(sum(exp(beta*(y-y_max))))
    # So softmin(x) = - ( -x_max + 1/beta * log(sum(exp(beta*(-x - (-x_max))))))
    #              = x_max - 1/beta * log(sum(exp(beta*(x_max - x))))
    
    x_max = np.max(x)
    exp_terms = np.exp(beta * (x_max - x))
    return x_max - (1.0 / beta) * np.log(np.sum(exp_terms))

def _signed_distances_to_hex(p, hex_verts):
    """
    Calculate signed distances from point p to the 6 edges of the hexagon defined by hex_verts.
    Positive if p is inside.
    """
    distances = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal pointing inward
        # For CCW polygon, inward normal is (-dy, dx) normalized?
        # Let's check. Edge (1,0)->(0,1). dx=-1, dy=1. Normal (-1, -1)?
        # Center is (0,0). Midpoint (0.5, 0.5). Vector to center (-0.5, -0.5).
        # Edge vector (-1, 1). Cross product z: (-1)(1) - (1)(-1) = 0? No.
        # Let's use the property that for a regular hexagon, the inward normal at edge i
        # is opposite to the direction of the vertex midpoint.
        
        # Simpler: Use the cross product of edge and vector from p1 to p.
        # If CCW, points to the left of edge are inside.
        # cross = edge[1]*(p[1]-p1[1]) - edge[0]*(p[0]-p1[0])
        # If cross > 0, p is to the left (inside) for CCW.
        # We need the actual distance, so normalize the edge.
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        # Check if normal points inward
        # Vector from edge midpoint to center (0,0) of this local hex?
        # We don't know the center of the local hex easily here, but we can use the fact
        # that the hexagon is convex.
        # Let's just compute the value. If the hexagon is CCW, the normal (-dy, dx) points left.
        # Is left inward?
        # For a standard CCW polygon, yes.
        dist = np.dot(normal, p - p1)
        distances.append(dist)
    return np.array(distances)

def _loss(params):
    """
    params: [S, x1, y1, th1, ..., x11, y11, th11]
    """
    S = params[0]
    if S < 1.0:
        S = 1.0
    
    # Outer hexagon centered at (0,0), angle 0
    # Apothem a = S * sqrt(3) / 2
    a = S * np.sqrt(3) / 2.0
    # Normals for outer hexagon (angle 0)
    # Vertices at pi/6 + k*pi/3.
    # Edge 0: between v0 and v1. Midpoint angle pi/6 + pi/6 = pi/3?
    # Let's just generate outer verts and use the same distance logic, but we want the "inside" distance.
    # For outer hex, we want to penalize if inner points are OUTSIDE.
    # So we want to penalize if signed distance < 0.
    
    outer_verts = _hex_verts(0.0, 0.0, S, 0.0)
    
    total_penalty = 0.0
    inner_verts_list = []
    
    # Extract inner hexagons
    for i in range(N):
        x = params[1 + i*3]
        y = params[2 + i*3]
        th_deg = params[3 + i*3]
        th_rad = np.deg2rad(th_deg)
        verts = _hex_verts(x, y, 1.0, th_deg)
        inner_verts_list.append(verts)
        
        # Containment Penalty
        # For each vertex of this inner hex, check distance to outer hex edges.
        for v in verts:
            d_out = _signed_distances_to_hex(v, outer_verts)
            # If any distance is negative, it's outside.
            # Penalty: sum of squares of negative distances.
            neg_d = np.minimum(0.0, d_out)
            total_penalty += np.sum(neg_d**2)
            
    # Overlap Penalty
    for i in range(N):
        for j in range(i + 1, N):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            # Check vertices of i against edges of j
            for p in vi:
                d_to_j = _signed_distances_to_hex(p, vj)
                # If p is inside j, min(d_to_j) > 0.
                # We want to penalize this.
                # Approximate min(d_to_j) with softmin.
                m = _softmin(d_to_j, beta=20.0)
                if m > 0:
                    total_penalty += m**2
            
            # Check vertices of j against edges of i
            for p in vj:
                d_to_i = _signed_distances_to_hex(p, vi)
                m = _softmin(d_to_i, beta=20.0)
                if m > 0:
                    total_penalty += m**2
                    
    # Objective: Minimize S (or S^2) + Penalty
    # Using S^2 helps with scaling
    obj = S**2 + 100.0 * total_penalty
    return obj

def _loss_grad(params):
    # L-BFGS-B can use finite differences if gradient is None, but providing it is faster.
    # Given the complexity, let's rely on finite differences for the first pass to ensure correctness,
    # or use a simple gradient estimation.
    # Actually, scipy.optimize.minimize with 'L-BFGS-B' accepts jac=None and uses finite differences.
    # With 34 variables, finite differences might be slow but acceptable for 120s.
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Initial Guess: Hexagonal lattice
    # Center hexagon at (0,0). 6 around it. 4 more?
    # Let's place them in a roughly circular pattern.
    x0 = [0.0, 1.732, 0.866, -0.866, -1.732, -0.866, 0.866, 1.732, 2.6, 0.0, -2.6]
    y0 = [0.0, 0.0, 1.5, 1.5, 0.0, -1.5, -1.5, 0.0, 0.0, 2.6, 0.0]
    th0 = [0.0]*11
    
    # Flatten: S, x1, y1, th1, ...
    # Initial S estimate: max extent + margin
    max_r = max(np.sqrt(x**2 + y**2) for x, y in zip(x0, y0)) + 2.0
    S0 = max(3.0, max_r)
    
    x0_flat = [S0]
    for i in range(N):
        x0_flat.extend([x0[i], y0[i], th0[i]])
        
    bounds = [(1.0, 10.0)] # S
    for i in range(N):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 60.0)])
        
    try:
        res = scipy.optimize.minimize(
            _loss, 
            x0_flat, 
            method='L-BFGS-B', 
            jac=None, 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > TIME_LIMIT:
            # If it takes too long, return the initial guess? No, res might be partial.
            # L-BFGS-B doesn't easily interrupt mid-iteration in a simple way without callbacks.
            # But 1000 iterations of 34 vars should be fast.
            pass
            
        best_params = res.x
        best_S = best_params[0]
        
        # Extract inner positions
        inner = []
        for i in range(N):
            x = best_params[1 + i*3]
            y = best_params[2 + i*3]
            th = best_params[3 + i*3]
            inner.append([x, y, th])
            
        return inner, (0.0, 0.0), best_S, 0.0
        
    except Exception as e:
        # Fallback to a simple valid packing if optimization fails
        # Linear chain
        inner = [[2.1 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        side = 2.1 * (N // 2) + 2.0
        return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
