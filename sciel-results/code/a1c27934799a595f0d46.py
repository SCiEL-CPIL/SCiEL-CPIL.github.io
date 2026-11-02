import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
EPS = 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles_deg):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            all_verts.append(verts)
        
        verts_flat = np.vstack(all_verts)
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

    def check_validity(positions, angles_deg, tol=1e-5):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                # Check SAT for p1 edges
                for k in range(6):
                    edge = p1[(k + 1) % 6] - p1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check SAT for p2 edges
                    for k in range(6):
                        edge = p2[(k + 1) % 6] - p2[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    
                    if not disjoint:
                        return False
        return True

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles_deg = params[2*N:3*N]
        return required_outer_side(positions, angles_deg)

    def make_constraint(i, j):
        def constraint(params):
            positions = params[:2*N].reshape(N, 2)
            angles_deg = params[2*N:3*N]
            
            v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles_deg[j])
            
            # SAT Check: If any separating axis exists, they are disjoint.
            # We want to enforce that NO separating axis exists?
            # No, we want to enforce that they are disjoint.
            # SAT says: Polygons are disjoint iff there exists an axis where projections are disjoint.
            # So, constraint: MAX over all axes of (min(projections) - max(projections)) >= 0.
            # Let D_axis = min(proj1) - max(proj2) OR min(proj2) - max(proj1).
            # We want max(D_axis) >= 0.
            
            max_sep = -np.inf
            
            # Axes from v1
            for k in range(6):
                edge = v1[(k + 1) % 6] - v1[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10: continue
                normal /= norm
                proj1 = v1.dot(normal)
                proj2 = v2.dot(normal)
                sep = min(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
                if sep > max_sep:
                    max_sep = sep
            
            # Axes from v2
            for k in range(6):
                edge = v2[(k + 1) % 6] - v2[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10: continue
                normal /= norm
                proj1 = v1.dot(normal)
                proj2 = v2.dot(normal)
                sep = min(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
                if sep > max_sep:
                    max_sep = sep
            
            # We require max_sep >= -tol (allowing slight numerical overlap or touching)
            # Actually, for strict non-overlap, we need max_sep >= 0.
            # Using a small negative tolerance helps with numerical stability in SLSQP.
            return max_sep + 1e-4
        
        constraint.name = f"nonoverlap_{i}_{j}"
        return constraint

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1_pos = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1_pos.append([x, y])
    init1_pos = np.array(init1_pos)
    init1_angles = np.zeros(N)
    configs.append(np.concatenate([init1_pos.flatten(), init1_angles]))
    
    # 2. Circular arrangement
    init2_pos = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2_pos[i, 0] = r * np.cos(angle)
        init2_pos[i, 1] = r * np.sin(angle)
    init2_angles = np.zeros(N)
    configs.append(np.concatenate([init2_pos.flatten(), init2_angles]))
    
    # 3. Random
    np.random.seed(42)
    init3_pos = np.random.uniform(-5, 5, (N, 2))
    init3_angles = np.random.uniform(0, 60, N)
    configs.append(np.concatenate([init3_pos.flatten(), init3_angles]))
    
    best_params = None
    best_side = float('inf')
    
    constraints = []
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(make_constraint(i, j))
            
    # Bounds for parameters
    # Positions: -10 to 10
    # Angles: -180 to 180
    bounds = [(-10, 10)] * (2 * N) + [(-180, 180)] * N

    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, scale up
        init_pos = init[:2*N].reshape(N, 2)
        init_angles = init[2*N:3*N]
        
        valid = check_validity(init_pos, init_angles)
        if not valid:
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled_pos = init_pos * scale
                if check_validity(scaled_pos, init_angles):
                    init_pos = scaled_pos
                    init = np.concatenate([init_pos.flatten(), init_angles])
                    break
            else:
                continue
        
        try:
            res = minimize(
                objective,
                init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_side:
                pos = res.x[:2*N].reshape(N, 2)
                angles = res.x[2*N:3*N]
                
                # Final validation
                if check_validity(pos, angles):
                    side = required_outer_side(pos, angles)
                    if side < best_side:
                        best_side = side
                        best_params = res.x
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = configs[0]
        best_side = required_outer_side(best_params[:2*N].reshape(N,2), best_params[2*N:3*N])
    
    # Final validation
    final_pos = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:3*N]
    
    if not check_validity(final_pos, final_angles):
        # Try to repair by scaling
        for scale in [1.1, 1.2, 1.5, 2.0]:
            scaled_pos = final_pos * scale
            if check_validity(scaled_pos, final_angles):
                final_pos = scaled_pos
                break
        else:
            # Last resort: linear arrangement
            final_pos = np.array([[i * 3.0, 0.0] for i in range(N)])
            final_angles = np.zeros(N)
            
    final_side = required_outer_side(final_pos, final_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_angles[i])])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
