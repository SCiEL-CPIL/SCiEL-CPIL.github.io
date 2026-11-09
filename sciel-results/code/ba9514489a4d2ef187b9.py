# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0

    def get_hex_vertices(cx, cy, s, angle_deg):
        """Get vertices of a regular hexagon with center (cx,cy), side s, angle."""
        a = np.radians(angle_deg)
        verts = np.array([
            [cx + s * np.cos(a + np.radians(60 * i)),
             cy + s * np.sin(a + np.radians(60 * i))]
            for i in range(6)
        ])
        return verts

    def separation_violation(v1, v2):
        """
        Check if two hexagons (arrays of 6 vertices each) violate separation.
        Returns the maximum penetration depth (positive = overlap, negative = separated).
        Uses the 12 edge normals test.
        """
        # Get edge normals for both hexagons
        normals1 = []
        normals2 = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            # Outward normal (perpendicular to edge, pointing away from center)
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            # Check if it points outward
            mid = (p1 + p2) / 2
            center1 = np.mean(v1, axis=0)
            if np.dot(n, mid - center1) < 0:
                n = -n
            normals1.append(n)
            
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            center2 = np.mean(v2, axis=0)
            mid = (p1 + p2) / 2
            if np.dot(n, mid - center2) < 0:
                n = -n
            normals2.append(n)
        
        # Check all 12 normals
        max_separation = -float('inf')
        for n in normals1 + normals2:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Separation along this normal
            sep = min(max1, max2) - max(min1, min2)
            max_separation = max(max_separation, sep)
        
        return max_separation  # positive means separated, negative means overlap

    def is_pair_separated(v1, v2, tol=1e-6):
        """Check if two hexagons are separated by more than tol."""
        return separation_violation(v1, v2) > tol

    def compute_enclosing_hex_side(pos, angles, outer_angle_deg):
        """
        Compute minimal side length of enclosing hexagon with given outer angle.
        pos: (11,2) array of centers
        angles: (11,) array of inner hex angles
        outer_angle_deg: angle of outer hexagon
        """
        # For each inner hexagon, get its 6 vertices
        # The enclosing hexagon must contain all vertices of all inner hexagons
        all_verts = []
        for i in range(11):
            v = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)  # (66, 2)
        
        # Outer hexagon with center (cx, cy), side s, angle a
        # Edge normals at angles a + 30 + 60*k for k=0..5
        base_angle = np.radians(outer_angle_deg + 30)
        normals = np.array([
            [np.cos(base_angle + np.radians(60 * k)),
             np.sin(base_angle + np.radians(60 * k))]
            for k in range(6)
        ])
        
        apothem_factor = np.sqrt(3) / 2
        
        # For each vertex and each normal:
        # (vertex - center) . normal <= s * apothem_factor
        # => -center . normal - s * apothem_factor <= -vertex . normal
        
        n_verts = len(all_verts)
        A_ub = np.zeros((n_verts * 6, 3))
        b_ub = np.zeros(n_verts * 6)
        
        idx = 0
        for i in range(n_verts):
            for k in range(6):
                A_ub[idx, 0] = -normals[k, 0]
                A_ub[idx, 1] = -normals[k, 1]
                A_ub[idx, 2] = -apothem_factor
                b_ub[idx] = -(all_verts[i, 0] * normals[k, 0] + all_verts[i, 1] * normals[k, 1])
                idx += 1
        
        from scipy.optimize import linprog
        result = linprog(np.array([0, 0, 1]), A_ub=A_ub, b_ub=b_ub,
                        bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if result.success:
            return result.x[2], result.x[0], result.x[1]
        return None, None, None

    def objective_and_check(params, outer_angle_deg):
        """
        params: (33,) = [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11]
        Returns (side_length, valid)
        """
        pos = params[0::3][:11].reshape(-1, 2)
        angles = params[1::3][:11]
        
        # Check pairwise separation
        verts_list = []
        for i in range(11):
            v = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
            verts_list.append(v)
        
        for i in range(11):
            for j in range(i + 1, 11):
                if not is_pair_separated(verts_list[i], verts_list[j]):
                    return None, False
        
        # Compute enclosing hexagon
        s, cx, cy = compute_enclosing_hex_side(pos, angles, outer_angle_deg)
        if s is None:
            return None, False
        
        return s, True

    def penalty_objective(params, outer_angle_deg):
        """Objective for optimization with penalties for violations."""
        pos = params[0::3][:11].reshape(-1, 2)
        angles = params[1::3][:11]
        
        verts_list = []
        for i in range(11):
            v = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
            verts_list.append(v)
        
        # Pairwise separation penalty
        sep_penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                sep = separation_violation(verts_list[i], verts_list[j])
                if sep < 1e-6:
                    sep_penalty += (1e-6 - sep) ** 2 * 1000
        
        # Compute enclosing hex side
        s, cx, cy = compute_enclosing_hex_side(pos, angles, outer_angle_deg)
        if s is None:
            return 1e6
        
        # Add penalty for being too small
        s_penalty = 0.0
        if s < np.sqrt(11):
            s_penalty = (np.sqrt(11) - s) ** 2 * 100
        
        return s + sep_penalty + s_penalty

    # Initial guess: flower pattern
    def make_initial_guess():
        params = np.zeros(33)
        # Center
        params[0:2] = [0, 0]
        # 6 around at distance sqrt(3)
        for k in range(6):
            a = np.radians(60 * k)
            idx = 3 * (k + 1)
            params[idx:idx+2] = [np.sqrt(3) * np.cos(a), np.sqrt(3) * np.sin(a)]
        # 4 more at distance 2*sqrt(3)
        for k in range(4):
            a = np.radians(60 * (2*k+1))  # odd multiples for gaps
            idx = 3 * (7 + k)
            params[idx:idx+2] = [2*np.sqrt(3) * np.cos(a), 2*np.sqrt(3) * np.sin(a)]
        # Angles all 0
        return params

    best_result = None
    best_score = -float('inf')
    
    # Try multiple outer angles
    outer_angles = [0, 15, 30]
    
    for outer_ang in outer_angles:
        if time.time() > deadline - safety_margin:
            break
        
        # Multi-start optimization
        n_starts = 15
        rng = np.random.RandomState(42)
        
        for start_idx in range(n_starts):
            if time.time() > deadline - safety_margin:
                break
            
            if start_idx == 0:
                x0 = make_initial_guess()
            else:
                # Perturbed initial guess
                x0 = make_initial_guess()
                x0 += rng.randn(33) * 0.3
            
            # Optimize
            result = minimize(
                penalty_objective, x0, args=(outer_ang,),
                method='SLSQP',
                bounds=[(-10, 10)] * 11 + [(-10, 10)] * 11 + [(0, 360)] * 11,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < 1e5:
                s, valid = objective_and_check(result.x, outer_ang)
                if valid and s is not None and s >= np.sqrt(11) - 1e-9:
                    score = 1.0 / s
                    if score > best_score:
                        best_score = score
                        # Reconstruct
                        pos = result.x[0::3][:11].reshape(-1, 2)
                        angles = result.x[1::3][:11]
                        # Get center of enclosing hex
                        s2, cx, cy = compute_enclosing_hex_side(pos, angles, outer_ang)
                        if s2 is not None:
                            best_result = (pos - np.array([cx, cy]), angles, outer_ang, s2)
    
    # If we found a result, use it
    if best_result is not None:
        pos, angles, outer_ang, s = best_result
        
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i, 0] = pos[i, 0]
            inner_hex_data[i, 1] = pos[i, 1]
            inner_hex_data[i, 2] = angles[i]
        
        outer_hex_data = np.array([0.0, 0.0, float(outer_ang)])
        outer_hex_side_length = float(s)
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    else:
        # Fallback: known valid configuration
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
