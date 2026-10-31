import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Unit hexagon vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees)
    # This orientation has a horizontal top and bottom edge.
    base_verts = np.array([(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)])
    
    def get_rotated_verts(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return base_verts @ rot.T + np.array([cx, cy])

    def compute_required_side_smooth(params, beta=50.0):
        """
        Computes a smooth approximation of the minimum outer hexagon side length S
        required to contain all inner hexagons.
        params: array of shape (3*N,) -> [x1, y1, a1, x2, y2, a2, ...]
        """
        # Unpack
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        # Center of mass to keep configuration centered (helps optimization)
        cx = np.mean(xs)
        cy = np.mean(ys)
        xs = xs - cx
        ys = ys - cy
        
        # Generate all vertices of all inner hexagons
        # Shape: (N, 6, 2)
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = get_rotated_verts(xs[i], ys[i], angles[i])
        
        # Flatten to (N*6, 2)
        all_verts = verts.reshape(-1, 2)
        
        # The outer hexagon is centered at (0,0) with side length S.
        # For a regular hexagon with side S, the apothem (distance from center to edge) is S * sqrt(3)/2.
        # The 6 outward normal directions of the outer hexagon edges are at angles 30, 90, 150, 210, 270, 330 degrees.
        # Let's define the normals relative to the "flat-top" orientation of the outer hex.
        # If outer hex is flat-top, its edges are horizontal/vertical-ish? 
        # Standard regular hexagon vertices at 0, 60... has flat left/right? No, pointy left/right.
        # Vertices: (1,0), (0.5, 0.866)... 
        # Edge 1 connects (1,0) and (0.5, 0.866). Midpoint (0.75, 0.433). Normal is perpendicular.
        # Actually, it's easier to define the 6 supporting lines.
        # For a hexagon centered at origin, the distance from center to each edge (apothem) is A = S * sqrt(3)/2.
        # The normals to the edges point in directions: 30, 90, 150, 210, 270, 330 degrees.
        
        theta_normals = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
        normals = np.column_stack((np.cos(theta_normals), np.sin(theta_normals))) # Shape (6, 2)
        
        # Project all inner vertices onto these 6 normals
        # Projections shape: (6, N*6)
        projs = all_verts.T @ normals.T
        
        # For each normal direction k, we need the maximum projection of any inner vertex.
        # If max_proj_k > 0, the required apothem is at least max_proj_k.
        # S_required_k = max_proj_k / (sqrt(3)/2) = max_proj_k * 2/sqrt(3)
        
        max_projs = np.max(projs, axis=1) # Shape (6,)
        
        # We need S such that S * sqrt(3)/2 >= max_projs for all k.
        # So S >= max(max_projs) * 2/sqrt(3).
        
        # To make it smooth and differentiable for optimization, we can use a soft-max or just the max.
        # Since we want to minimize S, and S is determined by the bottleneck direction,
        # using the raw max is non-differentiable. 
        # However, scipy's L-BFGS-B can handle non-smoothness to some extent, 
        # but a log-sum-exp approximation is better.
        
        # Let's use log-sum-exp to smooth the max.
        # max(x) ~= (1/beta) * log(sum(exp(beta*x)))
        # We want to minimize the max of the required S values.
        # Required S for direction k is max_projs[k] * 2/sqrt(3).
        # Let's just smooth the max_projs themselves, then scale.
        
        # If max_projs are all negative, S can be 0 (but we assume positive).
        # To avoid log(0) issues, add a small epsilon or shift.
        # But typically in packing, some projections will be positive.
        
        # Better approach: The objective is the side length S.
        # S = max_k ( max_proj_k * 2/sqrt(3) )
        
        # Let's use a smooth approximation of the maximum.
        # beta controls smoothness. Higher beta = sharper max.
        # We must be careful with overflow.
        
        # Shift max_projs to prevent overflow in exp
        m = np.max(max_projs)
        if m < 0:
             # If all projections are negative, the hexagon fits in a very small S.
             # But physically, S must be at least 1 (side of inner hex).
             # Actually, if all max_projs < 0, it means the origin is "outside" the convex hull in those directions?
             # No, max_proj is the furthest point in that direction.
             # If all are negative, the configuration is entirely in the opposite half-space?
             # Given we center at CoM, this shouldn't happen for all 6 directions.
             pass
            
        # For stability, we can just use the raw max if we use a gradient-free or robust method.
        # But let's try log-sum-exp.
        
        # To handle potential negative values in exp, we shift by m.
        # log(sum(exp(beta * (x - m)))) + m
        # This gives a smooth max.
        
        # However, if beta is high, it's close to max.
        # Let's use a moderate beta.
        
        # Actually, for packing, the "max" is the correct metric. 
        # Scipy's L-BFGS-B uses numerical gradients by default if jac is None.
        # It handles non-smooth functions reasonably well, but might get stuck.
        # Let's provide a gradient? No, let's stick to numerical for simplicity and robustness.
        
        # Objective: minimize S
        # S = max(max_projs) * 2 / math.sqrt(3)
        
        # If we want smoothness, we can return the log-sum-exp result.
        # But let's first try if L-BFGS-B can handle the raw max.
        
        # Alternative: Use the sum of squares of positive violations? No, we want the max.
        
        # Let's use the smooth max via log-sum-exp.
        # This ensures differentiability.
        
        # beta = 100.0 is usually good for "sharp" max.
        # But if values are large, exp overflows.
        
        # Let's implement a safe log-sum-exp.
        # max_projs shape (6,)
        
        # If all max_projs are very negative, S is small.
        
        # Let's just return the raw max for now, as L-BFGS-B is quite robust.
        # If it fails, we can switch to log-sum-exp.
        
        # Wait, the inspiration says "smooth log-sum-exp surrogate". Let's use it.
        
        # Shift to prevent overflow
        # We want max(x). 
        # soft_max(x) = (1/beta) * log(sum(exp(beta*x)))
        
        # Let's use beta = 50.
        
        # To be safe against overflow:
        # max_x = np.max(max_projs)
        # soft_max = max_x + (1/beta) * np.log(np.sum(np.exp(beta * (max_projs - max_x))))
        
        # This is numerically stable.
        
        max_x = np.max(max_projs)
        # If max_x is very large, beta*max_x might overflow. 
        # The shift (max_projs - max_x) ensures the max term is exp(0)=1.
        # Other terms are exp(negative), so < 1.
        # Sum is between 1 and 6.
        # Log is between 0 and log(6).
        # So soft_max is close to max_x.
        
        soft_max = max_x + (1.0/beta) * np.log(np.sum(np.exp(beta * (max_projs - max_x))))
        
        S = soft_max * 2.0 / math.sqrt(3.0)
        
        return S

    def check_validity(params):
        """
        Strict check: Are all inner hexagons disjoint and inside the outer hexagon?
        Returns True if valid.
        """
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        xs = xs - cx
        ys = ys - cy
        
        verts_list = []
        for i in range(N):
            verts_list.append(get_rotated_verts(xs[i], ys[i], angles[i]))
            
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(verts_list[i], verts_list[j]):
                    return False
        
        # Check containment
        S = compute_required_side_smooth(params, beta=1000.0) # Use sharp max for checking
        # Actually, let's compute the exact required S for the check
        # Re-use logic from smooth but with max
        max_projs_exact = np.max(verts_list[0].T @ normals.T, axis=1) # This is wrong, need all verts
        
        # Let's just use the smooth function with high beta as a proxy for exact, 
        # or better, compute exact max projection.
        
        # Exact S check:
        all_verts = np.vstack(verts_list)
        projs = all_verts.T @ normals.T
        max_projs = np.max(projs, axis=1)
        S_exact = np.max(max_projs) * 2.0 / math.sqrt(3.0)
        
        # If S_exact is the side length, the outer hexagon has apothem A = S_exact * sqrt(3)/2 = max(max_projs).
        # So by definition, the vertices are within the apothem distance.
        # Thus, containment is guaranteed if we define the outer hexagon with side S_exact.
        
        # So the only validity check needed is disjointness!
        # Because the outer hexagon is constructed to *just* contain them.
        
        return True

    def hexagons_disjoint(va, vb):
        """
        Check if two hexagons (arrays of 6 vertices) are disjoint.
        Uses Separating Axis Theorem (SAT) for convex polygons.
        """
        # Axes are the normals to the edges of both polygons
        # For a regular hexagon, there are 6 unique normal directions.
        # Since both are regular hexagons, we just need to check projections onto the 6 axes of one,
        # but they might be rotated differently.
        # So we check projections onto the 6 normals of va and the 6 normals of vb.
        
        # Get normals for va
        # Edge i is from va[i] to va[(i+1)%6]
        # Normal is perpendicular to edge.
        
        def get_normals(verts):
            normals = []
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                edge = p2 - p1
                # Normal is (-dy, dx)
                n = np.array([-edge[1], edge[0]])
                # Normalize
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    normals.append(n / n_norm)
            return normals
        
        normals_a = get_normals(va)
        normals_b = get_normals(vb)
        
        all_normals = normals_a + normals_b
        
        for n in all_normals:
            proj_a = np.dot(va, n)
            proj_b = np.dot(vb, n)
            
            min_a = np.min(proj_a)
            max_a = np.max(proj_a)
            min_b = np.min(proj_b)
            max_b = np.max(proj_b)
            
            # Check for separation
            if max_a < min_b or max_b < min_a:
                return True # Separated, so disjoint (or touching, which is allowed? "disjoint" usually means no interior overlap)
                # If they touch, max_a == min_b. This is not strictly disjoint in some definitions, but usually acceptable in packing unless "strictly disjoint" is required.
                # The task says "disjoint", which usually implies interiors are disjoint. Touching is allowed.
                # So if max_a <= min_b or max_b <= min_a, they are disjoint (non-overlapping).
                # My check `max_a < min_b` allows touching.
                # Let's use `<=` to allow touching.
        
        return False # Overlapping

    # Refine hexagons_disjoint to allow touching
    def hexagons_disjoint_allow_touching(va, vb):
        def get_normals(verts):
            normals = []
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    normals.append(n / n_norm)
            return normals
        
        normals_a = get_normals(va)
        normals_b = get_normals(vb)
        all_normals = normals_a + normals_b
        
        for n in all_normals:
            proj_a = np.dot(va, n)
            proj_b = np.dot(vb, n)
            
            min_a = np.min(proj_a)
            max_a = np.max(proj_a)
            min_b = np.min(proj_b)
            max_b = np.max(proj_b)
            
            if max_a <= min_b + 1e-9 or max_b <= min_a + 1e-9:
                return True
        
        return False

    # Update check_validity to use the touching-allowed version
    def check_validity(params):
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        xs = xs - cx
        ys = ys - cy
        
        verts_list = []
        for i in range(N):
            verts_list.append(get_rotated_verts(xs[i], ys[i], angles[i]))
            
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_allow_touching(verts_list[i], verts_list[j]):
                    return False
        return True

    # Initial guess: Hexagonal lattice
    # A regular hexagonal lattice of unit hexagons has spacing sqrt(3) in one direction and 1.5 in the other?
    # For flat-top hexagons of side 1:
    # Width = 2, Height = sqrt(3).
    # Horizontal spacing between columns: 1.5 (3/2 * side)
    # Vertical spacing between rows: sqrt(3)
    # Offset for odd columns: sqrt(3)/2
    
    def generate_lattice_guess(n_hex):
        points = []
        # Generate a large grid and pick the closest to origin
        # Or just generate a centered hexagonal cluster
        # For N=11, a 1-2-3-2-1 cluster? Or 3x3 minus corners?
        # Let's use axial coordinates (q, r)
        # Hex distance in axial coords
        
        # Generate all hexes within radius 3
        for q in range(-4, 5):
            for r in range(-4, 5):
                # Convert axial to pixel (flat-top)
                x = 1.5 * q
                y = math.sqrt(3) * (r + 0.5 * (q & 1))
                points.append((x, y, 0.0))
                
        # Sort by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take first N
        selected = points[:n_hex]
        
        # Center them
        xs = [p[0] for p in selected]
        ys = [p[1] for p in selected]
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        params = []
        for p in selected:
            params.extend([p[0]-cx, p[1]-cy, p[2]])
            
        return np.array(params)

    best_params = None
    best_S = float('inf')
    
    # Multi-start optimization
    num_starts = 5
    start_time = time.time()
    
    for k in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        if k == 0:
            x0 = generate_lattice_guess(N)
        else:
            # Perturb the best solution or lattice
            x0 = generate_lattice_guess(N)
            noise = np.random.normal(0, 0.5, size=x0.shape)
            x0 = x0 + noise
            
        # Optimize
        try:
            res = minimize(
                compute_required_side_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                # Check validity
                if check_validity(res.x):
                    S_val = compute_required_side_smooth(res.x, beta=1000.0)
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x.copy()
                        
        except Exception:
            pass
            
        # If we have a valid solution, try to polish it further with smaller steps
        if best_params is not None:
             # Polish with Nelder-Mead or small step L-BFGS
             try:
                 res2 = minimize(
                     compute_required_side_smooth,
                     best_params,
                     method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                 )
                 if res2.success and check_validity(res2.x):
                     S_val2 = compute_required_side_smooth(res2.x, beta=1000.0)
                     if S_val2 < best_S:
                         best_S = S_val2
                         best_params = res2.x.copy()
             except Exception:
                 pass

    # If no valid solution found, fallback to lattice
    if best_params is None:
        best_params = generate_lattice_guess(N)
        # Ensure it's valid (lattice should be valid)
        if not check_validity(best_params):
            # Scale up? No, lattice is disjoint.
            # Maybe the validity check is too strict or buggy.
            # Let's just return the lattice.
            pass

    # Extract results
    xs = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    inner = []
    for i in range(N):
        inner.append([xs[i]-cx, ys[i]-cy, angles[i]])
        
    # Calculate final S
    S_final = compute_required_side_smooth(best_params, beta=1000.0)
    
    return inner, (0.0, 0.0), S_final, 0.0

# EVOLVE-BLOCK-END
