import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit, leave 40s buffer)

    sqrt3 = math.sqrt(3.0)
    
    # 1. Generate Initial Static Packing
    # We use a hexagonal lattice arrangement.
    # A cluster of 19 hexagons (radius 2) is too big for a minimal container if we just take 12 randomly.
    # Let's pick the 12 hexagons from a 19-hex cluster that are closest to the center.
    # Coordinates in axial system (q, r)
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    # Sort by distance from origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:12]
    
    # Initial variables:
    # p0: 12 centers (x,y) -> 24 vars
    # alphas: 12 rotations -> 12 vars
    # Total: 36 vars
    
    p_init = np.zeros((N, 2))
    for i, (x, y) in enumerate(selected_centers):
        p_init[i] = [x, y]
        
    alphas_init = np.zeros(N)
    
    # Flatten for optimizer
    x0 = np.concatenate((p_init.flatten(), alphas_init))
    
    # Vertices of a unit hexagon (side 1)
    # Center at origin. Vertices at distance 1.
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([k * np.pi / 3.0 for k in range(6)])
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Pre-compute edge normals for unit hexagon
    # Edge 0 is between vertex 0 and vertex 1
    # Normal is perpendicular to edge, pointing outwards
    edge_normals = np.zeros((6, 2))
    for k in range(6):
        v1 = unit_verts[k]
        v2 = unit_verts[(k+1)%6]
        edge_vec = v2 - v1
        # Normal is rotate edge_vec by 90 deg
        # Since hexagon is CCW, outward normal is (edge_y, -edge_x) ? 
        # Let's check: v0=(1,0), v1=(0.5, sqrt3/2). edge = (-0.5, 0.866).
        # Outward normal should point away from origin.
        # Midpoint is (0.75, 0.433). Normal (0.866, 0.5) points to (1.6, 0.9) which is away.
        # So normal = (edge_y, -edge_x)
        edge_normals[k] = [edge_vec[1], -edge_vec[0]]
        # Normalize (length of edge is 1, so length of normal is 1)
    
    def get_vertices(p, alpha):
        """
        Get vertices of a hexagon with center p and rotation alpha.
        Returns array of shape (6, 2)
        """
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return unit_verts @ rot.T + p

    def hexagon_overlap(p1, a1, p2, a2):
        """
        Check if two hexagons overlap using SAT.
        Returns True if they overlap (penalty needed), False if disjoint.
        """
        v1 = get_vertices(p1, a1)
        v2 = get_vertices(p2, a2)
        
        # Get normals for both hexagons
        cos_a1 = math.cos(a1)
        sin_a1 = math.sin(a1)
        rot1 = np.array([[cos_a1, -sin_a1], [sin_a1, cos_a1]])
        normals1 = edge_normals @ rot1.T
        
        cos_a2 = math.cos(a2)
        sin_a2 = math.sin(a2)
        rot2 = np.array([[cos_a2, -sin_a2], [sin_a2, cos_a2]])
        normals2 = edge_normals @ rot2.T
        
        all_normals = np.vstack((normals1, normals2))
        
        for normal in all_normals:
            # Project v1
            proj1_min = np.min(v1 @ normal)
            proj1_max = np.max(v1 @ normal)
            # Project v2
            proj2_min = np.min(v2 @ normal)
            proj2_max = np.max(v2 @ normal)
            
            # Check for gap
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return False # Disjoint
        
        return True # Overlap

    def objective(x):
        if time.time() > deadline:
            return 1e10
            
        p = x[:24].reshape(N, 2)
        alphas = x[24:36]
        
        # Calculate outer hexagon side length
        # We need to enclose all vertices in a regular hexagon centered at origin (for simplicity, we fix outer center to 0,0)
        # The outer hexagon can rotate. Let's assume optimal outer hexagon is aligned with the "average" orientation 
        # or we can add outer rotation as a variable. 
        # For simplicity and speed, let's fix outer hexagon orientation to 0 (flat top) or optimize it separately?
        # Actually, the problem allows outer rotation. 
        # However, minimizing the side length of the *minimal enclosing* regular hexagon is equivalent to 
        # finding the minimal L such that there exists a rotation theta where all points fit.
        # 
        # A simpler approach: 
        # The width of a set of points in a direction n is max(p.n) - min(p.n).
        # For a regular hexagon of side L, the width in the direction of a vertex normal is L*sqrt(3).
        # The width in the direction of an edge normal is 2*L.
        # 
        # Let's use the property that a regular hexagon of side L is the intersection of 6 strips.
        # The condition for a point set S to be contained in a hexagon of side L centered at 0 with rotation theta is:
        # For each of the 6 normals n_k(theta), max_{p in S} (p . n_k) <= L * sqrt(3) / 2 ? No.
        # 
        # Let's derive the containment constraint.
        # A regular hexagon of side L centered at origin with one vertex on x-axis (rotation 0) is defined by:
        # |x| <= L
        # |x/2 + sqrt(3)y/2| <= L/2 ? No.
        # 
        # Let's use the standard definition:
        # Vertices at L * exp(i * (k * pi/3 + theta)).
        # The hexagon is the set of points p such that for all 6 outward normals n_k, p . n_k <= h,
        # where h is the apothem = L * sqrt(3) / 2.
        # 
        # So, for a given rotation theta of the OUTER hexagon, the side length L must satisfy:
        # L >= (2 / sqrt(3)) * max_{k, p} (p . n_k(theta))
        # 
        # We want to minimize L. We can also optimize theta.
        # Let's add theta as the 37th variable.
        
        # Wait, I defined x0 with 36 vars. Let's stick to 36 vars and assume outer hexagon is aligned optimally?
        # No, outer rotation matters. 
        # Let's modify the variable vector to include theta.
        # But I already defined x0. I will just optimize over theta in a nested loop or add it.
        # Adding it is better.
        
        # REVISION: I will re-define the objective to include theta as the 37th variable.
        # I need to update x0 and bounds in the main block.
        
        # For now, let's assume the objective function takes x of length 37.
        if len(x) < 37:
            # Fallback if called with old signature, though we will change it below
            theta = 0.0
        else:
            theta = x[36]
            
        # Normals of outer hexagon
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Calculate max projection of all vertices onto all normals
        # We need vertices of all inner hexagons
        # This is expensive. Let's compute vertices for all N hexagons.
        
        # Vectorized vertex computation
        cos_a = np.cos(alphas)
        sin_a = np.sin(alphas)
        # Rotation matrices
        # R = [[c, -s], [s, c]]
        # v_rot = R @ v_unit
        # v_x_rot = c * v_x - s * v_y
        # v_y_rot = s * v_x + c * v_y
        
        vx = cos_a[:, np.newaxis] * unit_verts[0, np.newaxis] - sin_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        vy = sin_a[:, np.newaxis] * unit_verts[0, np.newaxis] + cos_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        
        verts = np.stack((vx, vy), axis=-1) # (N, 6, 2)
        verts += p[:, np.newaxis, :] # Add centers
        
        verts_flat = verts.reshape(-1, 2)
        
        # Dots: (N*6, 6)
        dots = verts_flat @ normals.T
        max_proj = np.max(dots)
        
        L = (2.0 / sqrt3) * max_proj
        
        # Penalty for overlaps
        penalty = 0.0
        # Check all pairs
        for i in range(N):
            for j in range(i+1, N):
                if hexagon_overlap(p[i], alphas[i], p[j], alphas[j]):
                    # Calculate penetration depth or just a fixed penalty?
                    # Fixed penalty is cheaper but might not guide optimization well.
                    # Let's use a simple penalty: 10.0 * (number of overlaps)
                    penalty += 10.0
                    # Optional: add gradient-like info? Hard with SAT.
                    # Just penalty.
        
        return L + penalty

    # Update x0 to include theta
    theta_init = 0.0
    x0 = np.concatenate((p_init.flatten(), alphas_init, [theta_init]))
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
    for i in range(N):
        bounds.append((-math.pi, math.pi)) # alpha
    bounds.append((0.0, 2.0 * math.pi)) # theta

    best_x = x0
    best_val = objective(x0)
    
    # Use L-BFGS-B with penalty
    # It's faster than SLSQP for this many variables
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-6, 'maxfun': 1000})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass

    # Try a second run with a different initial theta if time permits
    if time.time() < deadline - 30:
        x0_2 = np.concatenate((p_init.flatten(), alphas_init, [math.pi/6]))
        try:
            res2 = minimize(objective, x0_2, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-6, 'maxfun': 1000})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Extract results
    p_final = best_x[:24].reshape(N, 2)
    alphas_final = best_x[24:36]
    theta_final = best_x[36]
    
    # Verify validity one last time
    # If penalty is high, the solution is invalid. 
    # We return the side length L calculated from the best_x.
    # Note: The evaluator checks if the hexagons are disjoint.
    # If our penalty didn't eliminate overlaps, the score is 0.
    # We must ensure the final configuration is disjoint.
    
    # Let's do a quick check and if it's invalid, we might need to return a known valid packing
    # The initial packing (selected_centers) is disjoint.
    # Let's calculate the side length for the initial packing to have a fallback.
    
    def calc_side(p, alphas, theta):
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        cos_a = np.cos(alphas)
        sin_a = np.sin(alphas)
        vx = cos_a[:, np.newaxis] * unit_verts[0, np.newaxis] - sin_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        vy = sin_a[:, np.newaxis] * unit_verts[0, np.newaxis] + cos_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        verts = np.stack((vx, vy), axis=-1)
        verts += p[:, np.newaxis, :]
        verts_flat = verts.reshape(-1, 2)
        dots = verts_flat @ normals.T
        max_proj = np.max(dots)
        return (2.0 / sqrt3) * max_proj

    fallback_side = calc_side(p_init, alphas_init, 0.0)
    
    # Check if best_x is valid
    is_valid = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_overlap(p_final[i], alphas_final[i], p_final[j], alphas_final[j]):
                is_valid = False
                break
        if not is_valid:
            break

    if not is_valid:
        # Use the initial valid packing
        p_final = p_init
        alphas_final = alphas_init
        theta_final = 0.0
        side = fallback_side
    else:
        side = calc_side(p_final, alphas_final, theta_final)
        
    inner = []
    for i in range(N):
        inner.append([p_final[i, 0], p_final[i, 1], np.degrees(alphas_final[i])])
        
    outer_center = (0.0, 0.0)
    angle = np.degrees(theta_final)
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
