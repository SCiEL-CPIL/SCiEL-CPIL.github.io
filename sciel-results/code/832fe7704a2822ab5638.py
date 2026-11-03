import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11

def get_hex_normals(center, angle, side):
    """
    Returns normals n (6x2) and offsets c (6,) for the half-plane representation.
    n_k . x <= c_k
    """
    # Normals point from center to edge midpoints.
    # If vertices are at pi/6 + k*pi/3, normals are at pi/3 + k*pi/3.
    # Let's align with the vertex definition in the seed:
    # Seed: base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    # This is pointy-top.
    # Normals for pointy-top (vertices at 30, 90...):
    # Edge 1 (30 to 90): Mid 60. Normal 60.
    # Edge 2 (90 to 150): Mid 120. Normal 120.
    # ...
    # So normals are at pi/3 + k*pi/3.
    normal_angles = np.pi / 3 + np.pi / 3 * np.arange(6) + angle
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Inradius
    r = side * np.sqrt(3) / 2
    
    # c_k = n_k . center + r
    c = normals @ center + r
    
    return normals, c

def get_hex_vertices(cx, cy, angle, side=1.0):
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def compute_violations(verts, normals, c):
    """
    verts: (6, 2)
    normals: (6, 2)
    c: (6,)
    Returns (6,) array of violations V_m = max_k (n_k . v_m - c_k)
    """
    # Dots: (6, 2) @ (2, 6) -> (6, 6)
    dots = verts @ normals.T
    # Subtract c: (6, 6) - (6,) -> (6, 6)
    diffs = dots - c
    # Max along axis 1 (normals)
    violations = np.max(diffs, axis=1)
    return violations

def smooth_relu(x, beta=100.0):
    """
    Smooth approximation of max(0, x).
    """
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow, use scipy.special.softplus or manual
    # We want smooth_relu(x) ~ max(0, x)
    # Using softplus(beta * x) / beta
    # If x > 0, softplus(beta x) ~ beta x. Result ~ x.
    # If x < 0, softplus(beta x) ~ exp(beta x). Result ~ exp(beta x)/beta -> 0.
    # Use logsumexp for stability?
    # log(1 + exp(z)) = logsumexp([0, z])
    z = beta * x
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -50, 50)
    val = np.logaddexp(0, z_clipped)
    return val / beta

def construct():
    # Variables: x1, y1, a1, ..., xN, yN, aN, S, phi_outer
    # Indices:
    # 0..32: 3*N
    # 33: S
    # 34: phi_outer
    
    def objective(params):
        # Extract params
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles_inner = params[2:2*N+2:3]
        S = params[3*N]
        phi_outer = params[3*N + 1]
        
        if S < 1.0:
            S = 1.0
            
        # Calculate outer hexagon
        outer_center = np.array([0.0, 0.0])
        outer_normals, outer_c = get_hex_normals(outer_center, phi_outer, S)
        
        penalty = 0.0
        
        # Precompute inner hex data
        inner_verts = []
        inner_normals = []
        inner_c = []
        
        for i in range(N):
            c_i = np.array([centers_x[i], centers_y[i]])
            v_i = get_hex_vertices(centers_x[i], centers_y[i], angles_inner[i])
            n_i, c_i_off = get_hex_normals(c_i, angles_inner[i], 1.0)
            
            inner_verts.append(v_i)
            inner_normals.append(n_i)
            inner_c.append(c_i_off)
            
            # Containment penalty: Vertices of i must be inside outer
            # V = max_k (n_outer_k . v_i_m - c_outer_k)
            # We want V <= 0
            viol_i = compute_violations(v_i, outer_normals, outer_c)
            # viol_i is (6,)
            # Penalty: sum of smooth_relu(viol_i)^2
            sr = smooth_relu(viol_i)
            penalty += np.sum(sr * sr)
            
        # Separation penalty
        for i in range(N):
            for j in range(i + 1, N):
                # Vertices of j inside i?
                # V_j_in_i = max_k (n_i_k . v_j_m - c_i_k)
                viol_j_in_i = compute_violations(inner_verts[j], inner_normals[i], inner_c[i])
                sr1 = smooth_relu(viol_j_in_i)
                penalty += np.sum(sr1 * sr1)
                
                # Vertices of i inside j?
                viol_i_in_j = compute_violations(inner_verts[i], inner_normals[j], inner_c[j])
                sr2 = smooth_relu(viol_i_in_j)
                penalty += np.sum(sr2 * sr2)
                
        # Objective: S + lambda * penalty
        lambda_val = 100.0
        return S + lambda_val * penalty

    best_S = float('inf')
    best_params = None
    
    # Multi-start
    num_starts = 10
    
    for start_idx in range(num_starts):
        np.random.seed(start_idx + 42)
        x0 = np.zeros(3 * N + 2)
        
        if start_idx == 0:
            # Hexagonal lattice initialization
            d = np.sqrt(3) + 0.01 # Touching distance
            
            # 1 center
            x0[0] = 0.0
            x0[1] = 0.0
            x0[2] = 0.0
            
            # 6 neighbors
            for k in range(6):
                ang = k * np.pi / 3
                idx = 3 * (1 + k)
                x0[idx] = d * np.cos(ang)
                x0[idx+1] = d * np.sin(ang)
                x0[idx+2] = 0.0
                
            # 4 more from second ring
            # Second ring positions are at distance 2*d? No, lattice spacing is d.
            # Distance to second ring is 2*d.
            # Pick 4 positions from the 12 in the second ring.
            # To be compact, pick them adjacent to the first ring?
            # Or spread them?
            # Let's pick 4 positions at angles: 0, 60, 120, 180?
            # Positions in 2nd ring:
            # (2d, 0), (d, d*sqrt(3)), (0, 2d*sqrt(3))...
            # Let's just place them at radius 2*d with some angle offsets.
            for k in range(4):
                ang = (k * 90 + 45) * np.pi / 180 # 45, 135, 225, 315
                idx = 3 * (7 + k)
                x0[idx] = 2.0 * d * np.cos(ang)
                x0[idx+1] = 2.0 * d * np.sin(ang)
                x0[idx+2] = 0.0
                
            x0[3*N] = 5.0
            x0[3*N+1] = 0.0
        else:
            # Random start near the lattice
            for i in range(N):
                idx = 3 * i
                # Jitter the lattice points
                # Get base lattice point
                if i == 0:
                    bx, by = 0.0, 0.0
                elif i <= 6:
                    ang = (i-1) * np.pi / 3
                    bx, by = d * np.cos(ang), d * np.sin(ang)
                else:
                    k = i - 7
                    ang = (k * 90 + 45) * np.pi / 180
                    bx, by = 2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang)
                
                x0[idx] = bx + np.random.normal(0, 0.2)
                x0[idx+1] = by + np.random.normal(0, 0.2)
                x0[idx+2] = np.random.normal(0, 0.1)
                
            x0[3*N] = 5.0
            x0[3*N+1] = np.random.uniform(0, np.pi/3)

        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Check validity
            S_val = res.x[3*N]
            # Re-evaluate penalty to be sure
            # We can't easily get penalty from res.fun without re-calc, but res.fun = S + lambda * P
            # So P = (res.fun - S_val) / lambda
            P_val = (res.fun - S_val) / 100.0
            
            if P_val < 1e-4:
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, np.degrees(ang)])
    
    outer_center = (0.0, 0.0)
    outer_side = best_params[3*N]
    outer_angle = np.degrees(best_params[3*N+1])
    
    return inner, outer_center, outer_side, outer_angle
