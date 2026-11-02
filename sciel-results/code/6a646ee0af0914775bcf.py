import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
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
        sqrt3 = math.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions, tol=1e-6):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        Optimized: precompute normals.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                # Check edges of p1
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
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
                    # Check edges of p2
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
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

    def lattice_positions():
        """
        Generate a hexagonal lattice configuration of 12 hexagons.
        Center + 6 neighbors + 5 more from the next ring to make 12.
        This guarantees disjointness initially.
        """
        sqrt3 = math.sqrt(3)
        # Hex grid coordinates (q, r)
        # Ring 0: (0,0) -> 1
        # Ring 1: 6 neighbors
        # Ring 2: 12 neighbors, pick 5
        coords = []
        # Center
        coords.append((0, 0))
        # Ring 1
        for q, r in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
            coords.append((q, r))
        # Ring 2 (partial)
        # Neighbors of Ring 1 that are not in Ring 0/1
        # (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2)?? No.
        # Let's just pick 5 specific ones from Ring 2 to break symmetry and allow compacting.
        ring2_candidates = [
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
            (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)
        ]
        # Pick 5
        for c in ring2_candidates[:5]:
            coords.append(c)
            
        positions = []
        for q, r in coords:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            positions.append([x, y])
        return np.array(positions)

    def local_search(positions, max_iters=5000):
        """
        Local search: try to move each hexagon slightly towards the center or randomly,
        accepting only if valid. This effectively "compacts" the cluster.
        """
        current = positions.copy()
        current_side = required_outer_side(current)
        if not check_validity(current):
            return current, current_side

        best = current.copy()
        best_side = current_side
        
        # Direction vectors: 6 axes + random
        directions = []
        for i in range(6):
            angle = i * math.pi / 3
            directions.append([math.cos(angle), math.sin(angle)])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random hexagon
            idx = np.random.randint(N)
            
            # Try a few small steps
            improved = False
            for _ in range(5):
                # Random direction and small step
                angle = np.random.uniform(0, 2 * math.pi)
                step = np.random.uniform(0.05, 0.2)
                dx = step * math.cos(angle)
                dy = step * math.sin(angle)
                
                new_pos = current.copy()
                new_pos[idx] += np.array([dx, dy])
                
                # Quick validity check
                if check_validity(new_pos):
                    new_side = required_outer_side(new_pos)
                    if new_side < current_side:
                        current = new_pos
                        current_side = new_side
                        improved = True
                        if current_side < best_side:
                            best = current.copy()
                            best_side = current_side
                        break # Accept and break to next iteration
        
        return best, best_side

    def rotate_cluster(positions, angle_rad):
        """Rotate all positions around the origin."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return positions @ rot.T

    def optimize_rotation(positions, max_iters=100):
        """
        Find the best rotation angle for the given cluster positions.
        """
        best_angle = 0.0
        best_side = required_outer_side(positions)
        
        current_angle = 0.0
        current_side = best_side
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try small rotations
            delta_angle = np.random.uniform(-0.1, 0.1)
            new_angle = current_angle + delta_angle
            rotated = rotate_cluster(positions, new_angle)
            new_side = required_outer_side(rotated)
            
            if new_side < current_side:
                current_angle = new_angle
                current_side = new_side
                if current_side < best_side:
                    best_angle = current_angle
                    best_side = current_side
        
        return rotate_cluster(positions, best_angle), best_side

    # Main logic
    best_positions = None
    best_side = float('inf')
    
    # 1. Start with lattice
    init_pos = lattice_positions()
    
    # 2. Local search to compact
    compacted_pos, side1 = local_search(init_pos, max_iters=20000)
    
    # 3. Optimize rotation
    rotated_pos, side2 = optimize_rotation(compacted_pos, max_iters=200)
    
    if side2 < best_side:
        best_side = side2
        best_positions = rotated_pos
    
    # 4. Try a few random restarts of local search from slightly perturbed lattice
    for restart in range(3):
        if time.time() > deadline:
            break
        perturbed = init_pos + np.random.normal(0, 0.1, (N, 2))
        # Ensure validity
        if not check_validity(perturbed):
            # Scale out slightly
            scale = 1.1
            while not check_validity(perturbed * scale) and scale < 2.0:
                scale += 0.1
            perturbed = perturbed * scale
        
        comp_pos, s1 = local_search(perturbed, max_iters=10000)
        rot_pos, s2 = optimize_rotation(comp_pos, max_iters=100)
        
        if s2 < best_side:
            best_side = s2
            best_positions = rot_pos

    # Fallback
    if best_positions is None:
        best_positions = init_pos
        best_side = required_outer_side(best_positions)

    # Final validation
    if not check_validity(best_positions):
        # Should not happen if local search is correct, but just in case
        best_positions = init_pos
        best_side = required_outer_side(best_positions)

    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
