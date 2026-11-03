"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        """
        Returns the maximum gap between two hexagons using SAT.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
            
            for n in [n1, n2]:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    overlap_amount = min(max1, max2) - max(min1, min2)
                    gap = -overlap_amount
                    
                if gap > max_gap:
                    max_gap = gap
                    
        return max_gap

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(centers, angles):
        """Check if configuration is valid (no overlaps)."""
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    def objective(params):
        """
        Objective function for optimization.
        params: 2N floats, x1, y1, x2, y2, ...
        Returns: outer side length + penalty for overlaps
        """
        centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        angles = [0.0] * N # Keep angles fixed at 0 for simplicity in this structural change
        
        # Check overlaps
        penalty = 0.0
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += -gap * 10.0 # Strong penalty for overlap
        
        if time.time() > deadline - 5.0:
            return 1e10
            
        # Center of mass for outer hexagon
        cx = sum(c[0] for c in centers) / N
        cy = sum(c[1] for c in centers) / N
        
        side = compute_outer_side(centers, angles, cx, cy, 0.0)
        
        return side + penalty

    # Generate initial configurations using correct honeycomb lattice
    # Lattice vectors: (2, 0) and (1, sqrt(3))
    # We will try a few rectangular blocks of this lattice
    
    best_side = float('inf')
    best_config = None
    
    def get_lattice_block(rows, cols):
        """
        Generate a block of hexagons in a honeycomb lattice.
        rows: number of rows
        cols: number of columns in the first row
        """
        centers = []
        for r in range(rows):
            for c in range(cols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * math.sqrt(3)
                centers.append((x, y))
        
        # Center the block
        cx_mean = sum(c[0] for c in centers) / len(centers)
        cy_mean = sum(c[1] for c in centers) / len(centers)
        return [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Candidate shapes that sum to 12
    # 2x6, 3x4, 4x3, 6x2
    # Also try "hexagonal" shapes like 1-3-3-3-2 (12) or similar
    # Let's stick to rectangular blocks first as they are easy to generate validly.
    
    candidate_shapes = [
        (2, 6),
        (6, 2),
        (3, 4),
        (4, 3)
    ]
    
    for r, c in candidate_shapes:
        if time.time() > deadline - 20.0:
            break
            
        init_centers = get_lattice_block(r, c)
        init_params = []
        for x, y in init_centers:
            init_params.extend([x, y])
        
        # Optimize positions
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success or res.fun < best_side:
            opt_params = res.x
            opt_centers = [(opt_params[2*i], opt_params[2*i+1]) for i in range(N)]
            angles = [0.0] * N
            
            if check_validity(opt_centers, angles):
                cx = sum(c[0] for c in opt_centers) / N
                cy = sum(c[1] for c in opt_centers) / N
                side = compute_outer_side(opt_centers, angles, cx, cy, 0.0)
                
                if side < best_side:
                    best_side = side
                    best_config = (opt_centers, angles, cx, cy, 0.0)
    
    if best_config:
        centers, angles, cx, cy, angle = best_config
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
        
        final_side = best_side * 1.001 + 1e-5
        return inner, (float(cx), float(cy)), float(final_side), float(angle)
    else:
        # Fallback: Return a large valid configuration
        centers = get_lattice_block(3, 4)
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
            
        side = compute_outer_side(centers, [0.0]*N, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), float(side * 1.1), 0.0

# EVOLVE-BLOCK-END
