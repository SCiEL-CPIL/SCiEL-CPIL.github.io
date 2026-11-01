import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def smooth_max(vals, k=10.0):
        """Log-sum-exp approximation of max for differentiability."""
        max_val = np.max(vals)
        if np.all(vals == max_val):
            return max_val
        return max_val + np.log(np.sum(np.exp(k * (vals - max_val)))) / k

    def compute_penetration_depth(p1_verts, p2_verts):
        """
        Estimate penetration depth between two convex polygons.
        Returns 0 if disjoint, negative if overlapping (approx).
        Uses SAT to find minimum separation axis.
        """
        min_sep = float('inf')
        
        # Check edges of p1
        for i in range(6):
            j = (i + 1) % 6
            edge = p1_verts[j] - p1_verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            proj1 = p1_verts.dot(normal)
            proj2 = p2_verts.dot(normal)
            sep = np.min(proj2) - np.max(proj1)
            min_sep = min(min_sep, sep)
            if sep > 0:
                return 0.0 # Disjoint along this axis
        
        # Check edges of p2
        for i in range(6):
            j = (i + 1) % 6
            edge = p2_verts[j] - p2_verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            proj1 = p1_verts.dot(normal)
            proj2 = p2_verts.dot(normal)
            sep = np.min(proj2) - np.max(proj1)
            min_sep = min(min_sep, sep)
            if sep > 0:
                return 0.0 # Disjoint along this axis
        
        return min_sep # Negative if overlapping

    def objective(params):
        """
        Objective function: Outer Side Length + Penalty for Overlaps.
        params: (2N,) array of [x1, y1, x2, y2, ...]
        """
        positions = params.reshape(N, 2)
        
        # 1. Outer side length
        side = required_outer_side(positions)
        
        # 2. Overlap penalty
        penalty = 0.0
        verts_list = []
        for i in range(N):
            verts_list.append(hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0))
        
        for i in range(N):
            for j in range(i + 1, N):
                # Quick bounding box check to skip far apart hexagons
                # (Optional optimization, but SAT is fast enough for N=12)
                sep = compute_penetration_depth(verts_list[i], verts_list[j])
                if sep < 0:
                    # Overlap
                    penalty += sep * sep
        
        return side + 100.0 * penalty

    def gradient(params):
        """
        Numerical gradient via central differences.
        """
        eps = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            old = params[i]
            params[i] = old + eps
            f_plus = objective(params)
            params[i] = old - eps
            f_minus = objective(params)
            params[i] = old
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def solve_from_init(init_positions, time_budget=10.0):
        """
        Run L-BFGS-B optimization from a given initial position.
        """
        init_params = init_positions.flatten()
        
        # Check if initial is already valid and good
        init_side = required_outer_side(init_positions)
        
        # Optimize
        try:
            res = minimize(
                objective, 
                init_params, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception as e:
            return init_positions, init_side
        
        opt_positions = res.x.reshape(N, 2)
        opt_side = required_outer_side(opt_positions)
        
        # Verify validity of the optimized solution
        # If penalty is still high, it might be invalid
        verts_list = [hexagon_vertices(opt_positions[i, 0], opt_positions[i, 1], 1.0, 0.0) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                sep = compute_penetration_depth(verts_list[i], verts_list[j])
                if sep < -1e-4: # Significant overlap
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return opt_positions, opt_side
        else:
            # If optimization failed to find a valid config, return best effort or init
            # Actually, if penalty is high, side length is unreliable as a "valid" side.
            # We should only accept if valid.
            return init_positions, init_side

    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random seeds
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    np.random.seed(123)
    init4 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init4)
    
    # 4. Perturbed Lattice
    init5 = init1 + np.random.uniform(-0.5, 0.5, (N, 2))
    configs.append(init5)

    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        verts_list = [hexagon_vertices(init[i, 0], init[i, 1], 1.0, 0.0) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                sep = compute_penetration_depth(verts_list[i], verts_list[j])
                if sep < 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to fix by scaling
            fixed = False
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                verts_list_s = [hexagon_vertices(scaled[i, 0], scaled[i, 1], 1.0, 0.0) for i in range(N)]
                valid_s = True
                for i in range(N):
                    for j in range(i + 1, N):
                        sep = compute_penetration_depth(verts_list_s[i], verts_list_s[j])
                        if sep < 0:
                            valid_s = False
                            break
                    if not valid_s:
                        break
                if valid_s:
                    init = scaled
                    fixed = True
                    break
            if not fixed:
                continue

        # Run optimization
        pos, side = solve_from_init(init)
        if pos is not None and side < best_side:
            best_side = side
            best_positions = pos

    # Fallback
    if best_positions is None:
        # Use the first valid config
        for init in configs:
            verts_list = [hexagon_vertices(init[i, 0], init[i, 1], 1.0, 0.0) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    sep = compute_penetration_depth(verts_list[i], verts_list[j])
                    if sep < 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_positions = init
                best_side = required_outer_side(init)
                break
        
        if best_positions is None:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
