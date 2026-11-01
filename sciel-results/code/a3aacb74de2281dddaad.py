"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Pre-compute normals for the outer hexagon (angle 0)
    # Normals are perpendicular to edges.
    # Vertices at angles 0, 60, 120...
    # Edges are between vertices.
    # Normal to edge between 0 and 60 deg is at 30 deg.
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
    phi_k = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack((np.cos(phi_k), np.sin(phi_k)))
    
    # Inradius of unit hexagon is sqrt(3)/2
    # Inradius of outer hexagon side S is S * sqrt(3)/2
    # Constraint: dot(v, n) <= S * sqrt(3)/2
    
    # For disjointness:
    # Forbidden region for relative center is hexagon of side 2.
    # Inradius of side 2 hexagon is 2 * sqrt(3)/2 = sqrt(3).
    # Constraint: max(dot(rel, n) - sqrt(3)) >= 0
    
    def get_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def is_valid(inner, center, side, angle):
        """Check if configuration is valid"""
        outer_center = np.array(center)
        outer_angle = angle
        
        # Check inner hexagons inside outer
        # We assume inner angles are 0 in the optimization, but check generally for safety
        for i in range(N):
            v_inner = get_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            # Translate to outer center frame? No, just check against outer boundaries
            # Calculate outer vertices
            outer_verts = get_vertices(center[0], center[1], side, angle)
            
            # Simple check: all inner vertices must be inside outer hexagon
            # Using winding number or half-plane test
            # Half-plane test against outer edges
            for i_edge in range(6):
                p1 = outer_verts[i_edge]
                p2 = outer_verts[(i_edge+1)%6]
                edge_vec = p2 - p1
                normal = np.array([-edge_vec[1], edge_vec[0]])
                # Ensure normal points outward
                # Center of outer is (center[0], center[1])
                # Vector from edge midpoint to center
                mid = (p1 + p2) / 2
                to_center = np.array(center) - mid
                if np.dot(normal, to_center) > 0:
                    normal = -normal
                
                # Check all inner vertices
                for v in v_inner:
                    # Project onto normal
                    # The distance from p1 along normal should be <= distance from center?
                    # Easier: dot(v - p1, normal) should be <= 0 (if normal points outward)
                    # Wait, if normal points outward, points inside have dot <= 0?
                    # Let's use the standard half-plane: n . (x - p1) <= 0
                    if np.dot(v - p1, normal) > 1e-9:
                        return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = get_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                
                # SAT for disjointness
                all_edges = []
                for i_edge in range(6):
                    all_edges.append(v1[i_edge] - v1[(i_edge+1)%6])
                    all_edges.append(v2[i_edge] - v2[(i_edge+1)%6])
                
                for edge in all_edges:
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) == 0:
                        continue
                    normal = normal / np.linalg.norm(normal)
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2 or max2 < min1:
                        return True # Separated on this axis, so disjoint
                # If no separation found, they overlap
                return False
        
        return True

    def objective_and_constraints(params):
        """
        Returns objective value and constraint values.
        We use SLSQP which takes constraints as functions returning arrays of inequalities (>= 0)
        """
        # Unpack
        # params[0:24] are x,y for 12 hexagons
        # params[24] is S
        coords = params[:24].reshape(12, 2)
        S = params[24]
        
        if S < 0:
            return 1e10, None
        
        # 1. Constraints for being inside outer hexagon
        # Outer hexagon center (0,0), side S, angle 0.
        # Normals: normals (precomputed)
        # Inradius R_out = S * sqrt(3)/2
        R_out = S * np.sqrt(3) / 2.0
        
        inside_constraints = []
        for i in range(N):
            cx, cy = coords[i]
            # Vertices of inner hex i (angle 0)
            # v_m = (cx + cos(m*pi/3), cy + sin(m*pi/3))
            for m in range(6):
                vx = cx + np.cos(m * np.pi / 3)
                vy = cy + np.sin(m * np.pi / 3)
                # Constraint: dot(v, n_k) <= R_out for all k
                # Rearranged: R_out - dot(v, n_k) >= 0
                for k in range(6):
                    val = R_out - (vx * normals[k,0] + vy * normals[k,1])
                    inside_constraints.append(val)
        
        # 2. Constraints for disjointness
        # For pair i,j: max_k ( dot(c_j - c_i, n_k) - sqrt(3) ) >= 0
        # This is equivalent to:
        # For each k, we don't have a single constraint.
        # The function g = max(...) is non-differentiable.
        # SLSQP can handle it if we provide the gradient? Or just the function?
        # SLSQP approximates gradients.
        # However, "max of linear functions" is convex. We want g >= 0.
        # This defines a non-convex feasible region (outside a hexagon).
        # SLSQP might struggle.
        # Alternative: Use the fact that if they are disjoint, there exists a separating axis.
        # But we need a single scalar constraint per pair for SLSQP.
        
        # Let's use the distance-based constraint instead?
        # No, the max formulation is the most direct.
        
        disjoint_constraints = []
        for i in range(N):
            for j in range(i+1, N):
                dx = coords[j,0] - coords[i,0]
                dy = coords[j,1] - coords[i,1]
                
                # Compute dot products with all 6 normals
                dots = np.dot(normals, [dx, dy])
                # g = max(dots) - sqrt(3)
                g = np.max(dots) - np.sqrt(3)
                disjoint_constraints.append(g)
        
        constraints = np.concatenate((inside_constraints, disjoint_constraints))
        
        # Objective: Minimize S
        return S, constraints

    def objective_wrapper(params):
        val, cons = objective_and_constraints(params)
        return val

    def constraint_wrapper(params):
        val, cons = objective_and_constraints(params)
        # SLSQP expects 'ineq' constraints to be >= 0
        return cons

    # Initial Guess
    def initial_guess():
        params = np.zeros(25)
        # Place in a loose grid
        # Hex grid positions
        # Ring 0: (0,0)
        # Ring 1: 6 at dist 2
        # Ring 2: 5 at dist 4?
        
        positions = []
        positions.append((0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta)))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta)))
            
        for i in range(N):
            params[2*i] = positions[i][0]
            params[2*i+1] = positions[i][1]
        
        params[24] = 6.0 # Initial S
        return params

    best_params = initial_guess()
    best_score = 1e10
    
    # Run optimization
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess
        if r == 0:
            params = initial_guess()
        else:
            params = best_params.copy()
            # Add noise to positions
            noise = np.random.normal(0, 0.5, size=24)
            params[:24] += noise
            # Don't change S much
            params[24] *= (1 + np.random.normal(0, 0.1))
            if params[24] < 3.0:
                params[24] = 3.0
        
        try:
            result = minimize(
                objective_wrapper, 
                params, 
                method='SLSQP',
                constraints=[{'type': 'ineq', 'fun': constraint_wrapper}],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.fun < best_score and result.fun < 1e9:
                # Verify validity with the robust checker
                inner = []
                for i in range(N):
                    inner.append([float(result.x[2*i]), float(result.x[2*i+1]), 0.0])
                center = (0.0, 0.0)
                side = float(result.x[24])
                angle = 0.0
                
                if is_valid(inner, center, side, angle):
                    best_score = result.fun
                    best_params = result.x
        except Exception as e:
            pass

    # Extract best result
    if best_params is not None and best_score < 1e9:
        inner = []
        for i in range(N):
            inner.append([float(best_params[2*i]), float(best_params[2*i+1]), 0.0])
        center = (0.0, 0.0)
        side = float(best_params[24])
        angle = 0.0
        
        # Final validation
        if is_valid(inner, center, side, angle):
            return inner, center, side, angle
    
    # Fallback to seed-like behavior or a known valid config
    # Use the initial guess if it was valid?
    # Let's just return a very large hexagon to be safe
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
