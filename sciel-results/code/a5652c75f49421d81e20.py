import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """
        Returns True if v1 and v2 are disjoint (including touching).
        Uses Separating Axis Theorem.
        """
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = np.min(np.dot(v1, [nx, ny])), np.max(np.dot(v1, [nx, ny]))
            min2, max2 = np.min(np.dot(v2, [nx, ny])), np.max(np.dot(v2, [nx, ny]))
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = np.min(np.dot(v1, [nx, ny])), np.max(np.dot(v1, [nx, ny]))
            min2, max2 = np.min(np.dot(v2, [nx, ny])), np.max(np.dot(v2, [nx, ny]))
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        return False

    def compute_outer_side(centers, angles, ocx=0.0, ocy=0.0, otheta=0.0):
        """
        Compute the minimum side length of a regular hexagon centered at (ocx, ocy) 
        with orientation otheta that contains all inner hexagons.
        """
        otheta_rad = math.radians(otheta)
        max_proj = 0.0
        # Normals for the 6 sides of the outer hexagon
        # For a regular hexagon, the normals to the sides are at angles 
        # otheta + 30, otheta + 90, ...
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                # Vertices of inner hexagon
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        
        # The distance from center to side is max_proj.
        # For a regular hexagon of side S, distance to side is S * sqrt(3) / 2.
        # So S = max_proj * 2 / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles):
        """
        Check if all hexagons are disjoint.
        """
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(N):
            for j in range(i + 1, N):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    # Initial configuration: Hexagonal lattice
    # This is a valid starting point
    d = math.sqrt(3)
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    for i in range(5):
        centers_init.append(ring2_candidates[i].copy())
        
    assert len(centers_init) == N
    
    # State representation: list of [x, y, angle_deg]
    current_state = [[c[0], c[1], 0.0] for c in centers_init]
    current_side = compute_outer_side([s[:2] for s in current_state], [s[2] for s in current_state])
    
    best_state = [s.copy() for s in current_state]
    best_side = current_side
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    step_size = 0.5
    max_iters = 50000
    
    start_time = time.time()
    
    for i_iter in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        old_x, old_y, old_angle = current_state[idx]
        
        # Perturbation
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        d_angle = random.gauss(0, step_size)
        
        new_x = old_x + dx
        new_y = old_y + dy
        new_angle = old_angle + d_angle
        
        # Check validity (overlap)
        # Temporarily update state
        current_state[idx] = [new_x, new_y, new_angle]
        
        if not is_valid([s[:2] for s in current_state], [s[2] for s in current_state]):
            # Revert
            current_state[idx] = [old_x, old_y, old_angle]
            continue
        
        # Calculate new side length
        new_side = compute_outer_side([s[:2] for s in current_state], [s[2] for s in current_state])
        
        # Acceptance criterion
        delta = new_side - current_side
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_state = [s.copy() for s in current_state]
        else:
            # Revert
            current_state[idx] = [old_x, old_y, old_angle]
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optionally restart with slightly larger step or different seed
            # For now, just continue with small steps

    # Final Polish: Try to optimize outer hexagon orientation and center
    # We can try a few orientations for the outer hexagon
    best_final_side = best_side
    best_final_state = best_state
    best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0
    
    for otheta in [0, 15, 30, 45, 60]:
        # Try to center the outer hexagon at the centroid of the inner hexagons
        cx_avg = sum(s[0] for s in best_state) / N
        cy_avg = sum(s[1] for s in best_state) / N
        
        side = compute_outer_side([s[:2] for s in best_state], [s[2] for s in best_state], cx_avg, cy_avg, otheta)
        if side < best_final_side:
            best_final_side = side
            best_ocx, best_ocy, best_otheta = cx_avg, cy_avg, otheta

    # Return best configuration
    inner = [[s[0], s[1], s[2]] for s in best_state]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_final_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
