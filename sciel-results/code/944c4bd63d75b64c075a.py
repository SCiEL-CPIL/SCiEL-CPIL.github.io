"""Rewritten program using physics-based relaxation for hexagon packing."""
import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def min_distance_between_hexagons(v1, v2):
        """
        Compute the minimum distance between two convex polygons (hexagons).
        Returns 0 if they overlap, positive if separated.
        Uses SAT to find minimum separation.
        """
        min_dist = float('inf')
        for poly in [v1, v2]:
            for i in range(len(poly)):
                edge = poly[(i + 1) % len(poly)] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                # Gap in this direction
                gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                if gap > 0:
                    min_dist = min(min_dist, gap)
                else:
                    # Overlapping in this direction, but check other axes
                    pass
        return max(0.0, min_dist)

    def polygons_overlap_sat(v1, v2):
        """SAT overlap test for two convex polygons."""
        for poly in [v1, v2]:
            for i in range(len(poly)):
                edge = poly[(i + 1) % len(poly)] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def compute_outer_side(positions, angles, outer_angle_deg):
        """
        Compute the minimum side length of a regular hexagon (centered at origin,
        with given rotation) that contains all inner hexagons.
        """
        all_verts = np.vstack([hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)])
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        # Apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def physics_relax(positions, angles, outer_angle, max_iters=20000, dt=0.01):
        """
        Physics-based relaxation: use repulsive forces between overlapping hexagons
        and attractive forces toward the center to find a compact, valid packing.
        """
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        
        # Adaptive step size
        current_dt = dt
        prev_energy = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Compute forces
            forces = np.zeros_like(pos)
            
            # Repulsive forces between overlapping hexagons
            verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
            
            for i in range(N):
                for j in range(i + 1, N):
                    if polygons_overlap_sat(verts[i], verts[j]):
                        # Compute separation direction
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        if dist < 1e-6:
                            d = np.array([0.1, 0.1])
                            dist = np.linalg.norm(d)
                        d_norm = d / dist
                        # Strong repulsive force
                        force_mag = 1.0
                        forces[i] -= force_mag * d_norm
                        forces[j] += force_mag * d_norm
            
            # Attractive force toward center (to keep packing compact)
            for i in range(N):
                to_center = -pos[i]
                dist_to_center = np.linalg.norm(to_center)
                if dist_to_center > 1e-6:
                    forces[i] += 0.01 * to_center / dist_to_center
            
            # Update positions
            new_pos = pos + current_dt * forces
            
            # Check validity
            new_verts = [hex_vertices(new_pos[i][0], new_pos[i][1], ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if polygons_overlap_sat(new_verts[i], new_verts[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                pos = new_pos
                # Compute energy (outer side length)
                score = compute_outer_side(pos, ang, outer_angle)
                if score < prev_energy:
                    prev_energy = score
                    current_dt = min(current_dt * 1.1, 0.05)  # Increase step
                else:
                    current_dt = max(current_dt * 0.9, 0.001)  # Decrease step
            else:
                # Reject and reduce step
                current_dt = max(current_dt * 0.5, 0.001)
        
        return pos, ang, outer_angle

    def refine_with_local_search(positions, angles, outer_angle, max_iters=5000):
        """
        After physics relaxation, do fine-tuning local search to minimize outer side.
        """
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        best_score = compute_outer_side(pos, ang, outer_angle)
        best_pos = pos.copy()
        
        if best_score == float('inf'):
            return pos, ang, outer_angle, best_score
        
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for i in range(N):
                for dx in [-0.01, 0.0, 0.01]:
                    for dy in [-0.01, 0.0, 0.01]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = pos.copy()
                        new_pos[i] += np.array([dx, dy])
                        
                        # Check validity
                        new_verts = [hex_vertices(new_pos[k][0], new_pos[k][1], ang[k]) for k in range(N)]
                        valid = True
                        for a in range(N):
                            for b in range(a + 1, N):
                                if polygons_overlap_sat(new_verts[a], new_verts[b]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            new_score = compute_outer_side(new_pos, ang, outer_angle)
                            if new_score < best_score - 1e-9:
                                pos = new_pos
                                best_score = new_score
                                best_pos = pos.copy()
                                improved = True
            
            if not improved:
                break
        
        return best_pos, ang, outer_angle, best_score

    def make_honeycomb(rows, cols):
        """Generate a honeycomb lattice of hexagons."""
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * math.sqrt(3)
                y = r * 1.5
                if r % 2 == 1:
                    x += math.sqrt(3) / 2
                positions.append((x, y))
                angles.append(0.0)
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    def make_hex_ring(center_count):
        """Place hexagons in a hexagonal ring pattern."""
        positions = []
        angles = []
        # Center cluster
        cluster_positions = [
            (0.0, 0.0),
            (math.sqrt(3), 0.0),
            (math.sqrt(3) / 2, 1.5),
            (-math.sqrt(3) / 2, 1.5),
            (-math.sqrt(3), 0.0),
            (-math.sqrt(3) / 2, -1.5),
            (math.sqrt(3) / 2, -1.5),
        ]
        for i in range(min(center_count, 7)):
            positions.append(cluster_positions[i])
            angles.append(0.0)
        
        remaining = N - len(positions)
        if remaining > 0:
            ring_radius = 3.0 * math.sqrt(3) / 2
            for i in range(remaining):
                a = 2 * math.pi * i / max(remaining, 1)
                x = ring_radius * math.cos(a)
                y = ring_radius * math.sin(a)
                positions.append((x, y))
                angles.append(0.0)
        
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    def random_config(seed):
        """Generate a random initial configuration."""
        np.random.seed(seed)
        positions = []
        angles = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 360)
            positions.append((x, y))
            angles.append(a)
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # Generate diverse initial configurations
    candidates = []
    
    # Honeycomb patterns
    candidates.append(make_honeycomb(4, 3))
    candidates.append(make_honeycomb(3, 4))
    candidates.append(make_honeycomb(2, 6))
    candidates.append(make_honeycomb(6, 2))
    
    # Hex ring patterns
    candidates.append(make_hex_ring(7))
    candidates.append(make_hex_ring(4))
    candidates.append(make_hex_ring(1))
    
    # Random configs
    for seed in [42, 123, 456, 789, 1011]:
        candidates.append(random_config(seed))
    
    # Try different outer angles
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    for idx, (pos, ang) in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        
        for oa in outer_angles:
            if time.time() > deadline - 20:
                break
            
            # First check if initial config is valid
            verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if polygons_overlap_sat(verts[i], verts[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = compute_outer_side(pos, ang, oa)
                if score < best_score:
                    best_score = score
                    best_positions = pos
                    best_angles = ang
                    best_outer_angle = oa
            
            # Physics relaxation
            if time.time() < deadline - 25:
                phys_pos, phys_ang, phys_oa = physics_relax(pos, ang, oa, max_iters=15000)
                
                # Check validity after physics
                phys_verts = [hex_vertices(phys_pos[i][0], phys_pos[i][1], phys_ang[i]) for i in range(N)]
                phys_valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if polygons_overlap_sat(phys_verts[i], phys_verts[j]):
                            phys_valid = False
                            break
                    if not phys_valid:
                        break
                
                if phys_valid:
                    phys_score = compute_outer_side(phys_pos, phys_ang, phys_oa)
                    if phys_score < best_score:
                        best_score = phys_score
                        best_positions = phys_pos
                        best_angles = phys_ang
                        best_outer_angle = phys_oa
                    
                    # Refine with local search
                    if time.time() < deadline - 15:
                        ref_pos, ref_ang, ref_oa, ref_score = refine_with_local_search(
                            phys_pos, phys_ang, phys_oa, max_iters=3000)
                        if ref_score < best_score:
                            best_score = ref_score
                            best_positions = ref_pos
                            best_angles = ref_ang
                            best_outer_angle = ref_oa

    # Fallback
    if best_positions is None or best_score == float('inf'):
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 3.0, r * 3.0))
            angles.append(0.0)
        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_outer_side(positions, angles, 0.0)

    # Final centering
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0.0, 0.0)] * N
        best_angles = [0.0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
