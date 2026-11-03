import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def hexagon_radius_in_direction(cx, cy, angle_deg, theta):
        """Maximum projection of hexagon vertices in direction theta."""
        angle = math.radians(angle_deg)
        max_proj = 0.0
        for i in range(6):
            vert_angle = angle + i * math.pi / 3
            # Vertex position relative to center
            vx = math.cos(vert_angle)
            vy = math.sin(vert_angle)
            proj = vx * math.cos(theta) + vy * math.sin(theta)
            if proj > max_proj:
                max_proj = proj
        return max_proj

    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.001):
        """Check if all hexagons are disjoint (center distance >= sqrt(3))."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimum outer hexagon side for given configuration."""
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                # For each hexagon, find max projection of its vertices
                cx, cy = centers[i]
                angle = angles[i]
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_config(centers, angles, max_iters=200, step_size=0.05):
        """Local optimization to minimize outer hexagon side."""
        n = len(centers)
        best_centers = [c.copy() for c in centers]
        best_angles = angles.copy()
        best_side = compute_outer_side(centers, angles, 0, 0, 0)
        
        # Optimize outer hexagon center and angle
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        for outer_iter in range(100):
            if time.time() > deadline - 30:
                break
            improved = False
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    for da_deg in [-15, 0, 15]:
                        da = math.radians(da_deg)
                        side = compute_outer_side(centers, angles, ocx+dx, ocy+dy, otheta+da)
                        if side < best_side - 1e-10:
                            best_side = side
                            ocx += dx
                            ocy += dy
                            otheta += da
                            improved = True
            if not improved:
                break
        
        # Now optimize inner hexagon positions and angles
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(n):
                # Try position perturbations
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        new_centers = [c.copy() for c in centers]
                        new_centers[i][0] += dx
                        new_centers[i][1] += dy
                        if check_disjoint(new_centers, angles):
                            side = compute_outer_side(new_centers, angles, ocx, ocy, otheta)
                            if side < best_side - 1e-10:
                                best_side = side
                                best_centers = [c.copy() for c in new_centers]
                                centers = new_centers
                                improved = True
                # Try angle perturbations
                for da_deg in [-15, 0, 15]:
                    new_angles = angles.copy()
                    new_angles[i] += da_deg
                    if check_disjoint(centers, new_angles):
                        side = compute_outer_side(centers, new_angles, ocx, ocy, otheta)
                        if side < best_side - 1e-10:
                            best_side = side
                            best_angles = new_angles.copy()
                            angles = new_angles
                            improved = True
            if not improved:
                break
        
        return best_centers, best_angles, best_side, ocx, ocy, otheta

    # Lattice vectors for hexagonal packing
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    
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
        centers.append(ring2_candidates[i].copy())
        
    assert len(centers) == 12
    
    # Convert to list format
    centers_list = [[c[0], c[1]] for c in centers]
    angles_list = [30.0] * 12
    
    # Run local optimization
    opt_centers, opt_angles, opt_side, opt_ocx, opt_ocy, opt_otheta = optimize_config(centers_list, angles_list)
    
    # Final high-precision polish: fine-tune outer hexagon parameters
    best_outer_side = opt_side
    best_ocx, best_ocy, best_otheta_deg = opt_ocx, opt_ocy, math.degrees(opt_otheta)
    
    # Fine grid search around the optimized outer parameters
    for otheta_deg in np.linspace(best_otheta_deg - 10, best_otheta_deg + 10, 41):
        otheta = math.radians(otheta_deg)
        for ocx in np.linspace(best_ocx - 0.5, best_ocx + 0.5, 11):
            for ocy in np.linspace(best_ocy - 0.5, best_ocy + 0.5, 11):
                side = compute_outer_side(opt_centers, opt_angles, ocx, ocy, otheta)
                if side < best_outer_side - 1e-12:
                    best_outer_side = side
                    best_ocx, best_ocy, best_otheta_deg = ocx, ocy, otheta_deg
    
    # Use optimized configuration
    inner = [[c[0], c[1], a] for c, a in zip(opt_centers, opt_angles)]
    
    outer_center = (best_ocx, best_ocy)
    outer_side = best_outer_side
    outer_angle = best_otheta_deg
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
