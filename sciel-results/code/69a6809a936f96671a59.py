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
        Returns the maximum gap between two hexagons.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        # Use 6 axes for robustness, though 3 are enough for parallel edges
        # We check normals of all edges of both hexagons
        axes = []
        for i in range(6):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                axes.append(n1 / norm1)
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                axes.append(n2 / norm2)
        
        for n in axes:
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

    def simulate_shrink(initial_centers, initial_angles, initial_outer_side, outer_angle_deg=0.0):
        """
        Simulates the packing by shrinking the container.
        Returns the best valid configuration found.
        """
        # State: x, y, vx, vy for each hexagon
        # Angles are fixed for now to simplify dynamics (translation only)
        # This is a valid subset of the solution space and often sufficient for hexagons
        # If we need rotation, we'd add theta, omega. Let's stick to translation first for speed/stability.
        
        positions = np.array(initial_centers, dtype=float)
        velocities = np.zeros_like(positions)
        
        outer_center = np.array([0.0, 0.0])
        
        # Start with a large container and shrink
        current_side = initial_outer_side
        min_side = 3.0 # Theoretical minimum is roughly sqrt(3)*2 + something? 
                       # For 12 hexes, a 2x3 grid has side ~ 3*sqrt(3)/2 + 1.5? 
                       # Let's start from a safe upper bound and shrink.
                       # A single hex has side 1. 12 hexes in a line: 12 units long -> side ~ 7-8.
                       # Compact cluster: ~ 4-5.
                       
        dt = 0.005
        damping = 0.95
        k_repel = 500.0
        k_wall = 1000.0
        margin = 1e-4 # Safety margin for validity check
        
        best_config = None
        best_side = float('inf')
        
        # Simulation loop
        # We shrink the side length gradually
        # We use a "quasi-static" approach: apply forces, integrate, damp.
        
        num_steps_per_shrink = 200
        shrink_steps = 100
        total_steps = num_steps_per_shrink * shrink_steps
        
        # Initial side estimate
        # Calculate bounding box of initial centers to set initial side
        max_x = np.max(positions[:, 0])
        min_x = np.min(positions[:, 0])
        max_y = np.max(positions[:, 1])
        min_y = np.min(positions[:, 1])
        width = max_x - min_x + 2.0
        height = max_y - min_y + 2.0
        # Approximate side length needed to contain width/height in a hexagon
        # For a hexagon of side s, width (flat-to-flat) is sqrt(3)*s, height (point-to-point) is 2*s
        # So s >= max(width/sqrt(3), height/2)
        init_side = max(width / math.sqrt(3), height / 2) * 1.2 + 2.0
        
        current_side = init_side
        
        # Precompute vertices for inner hexagons (angles are 0)
        # We only need to update vertices if we allow rotation. Here angles are fixed to 0.
        # But wait, the problem allows rotation. 
        # Fixed orientation (angle=0) is a valid constraint. We will search for best translation.
        # If we want to allow rotation, we need to add it to the state.
        # Let's stick to translation only first as it's more stable and faster.
        # If this doesn't improve, we can try adding rotation.
        
        angles = np.zeros(N)
        
        for step in range(total_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Shrink container
            if step % num_steps_per_shrink == 0 and step > 0:
                # Shrink towards a target
                # Target side: We don't know the exact minimum, so we shrink aggressively
                # But we must ensure we don't shrink below the physical limit of the current config
                # Actually, we just shrink a fixed amount or ratio.
                current_side *= 0.95
                if current_side < 4.0:
                    current_side = 4.0 # Don't go too small, let's see
            
            # Calculate forces
            forces = np.zeros_like(positions)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    v_i = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
                    v_j = hex_vertices(positions[j, 0], positions[j, 1], angles[j])
                    
                    gap = get_gap(v_i, v_j)
                    
                    if gap < 0:
                        # Overlap
                        # Force direction: push apart along the axis of maximum penetration
                        # We need to recover the axis. get_gap doesn't return it.
                        # For simplicity, use center-to-center vector as approximation for direction
                        # This is less precise but robust.
                        delta = positions[i] - positions[j]
                        dist = np.linalg.norm(delta)
                        if dist < 1e-6:
                            delta = np.array([1.0, 0.0])
                            dist = 1.0
                        else:
                            delta = delta / dist
                        
                        # Strength proportional to overlap depth
                        force_mag = k_repel * (-gap)
                        forces[i] += delta * force_mag
                        forces[j] -= delta * force_mag
            
            # Wall repulsion
            # Outer hexagon normals
            outer_angle_rad = math.radians(outer_angle_deg)
            for i in range(N):
                v_i = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
                rel_verts = v_i - outer_center
                
                for k in range(6):
                    normal_angle = outer_angle_rad + k * math.pi / 3
                    normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                    
                    # Projection of vertices onto normal
                    proj = np.dot(rel_verts, normal)
                    max_proj = np.max(proj)
                    
                    # Distance from wall:
                    # The wall is at distance (side * sqrt(3) / 2) from center along the normal
                    # For a hexagon of side S, the apothem is S * sqrt(3) / 2
                    apothem = current_side * math.sqrt(3) / 2.0
                    
                    # If max_proj > apothem, we are outside
                    if max_proj > apothem:
                        penetration = max_proj - apothem
                        # Force inward
                        forces[i] -= normal * k_wall * penetration
            
            # Integrate
            # F = ma, m=1
            velocities += forces * dt
            velocities *= damping
            positions += velocities * dt
            
            # Check validity and record best
            # We check validity frequently to avoid wasting time on invalid states
            if step % 10 == 0:
                # Check if all hexagons are inside the container
                valid = True
                for i in range(N):
                    v_i = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
                    rel_verts = v_i - outer_center
                    for k in range(6):
                        normal_angle = outer_angle_rad + k * math.pi / 3
                        normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                        proj = np.dot(rel_verts, normal)
                        max_proj = np.max(proj)
                        apothem = current_side * math.sqrt(3) / 2.0
                        if max_proj > apothem + 1e-6: # Small tolerance
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Check pairwise overlap
                    for i in range(N):
                        for j in range(i+1, N):
                            v_i = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
                            v_j = hex_vertices(positions[j, 0], positions[j, 1], angles[j])
                            gap = get_gap(v_i, v_j)
                            if gap < -1e-5:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        # This configuration is valid with side = current_side
                        # We can try to shrink further in the next iterations
                        # Record this as a candidate
                        # Note: The actual minimum side for this configuration might be slightly smaller
                        # but current_side is a valid upper bound.
                        if current_side < best_side:
                            best_side = current_side
                            best_config = (positions.copy(), angles.copy(), outer_center.copy(), current_side)

        if best_config is None:
            return None
            
        return best_config

    # Initial configuration: Hexagonal grid
    # 12 hexagons.
    # Arrangement: 3 rows of 4? Or 2 rows of 6? Or a flower pattern?
    # Let's try a compact cluster.
    # Centers in a hexagonal lattice.
    # dx = sqrt(3), dy = 1.5
    # Row 0: 0, 1, 2, 3 (4 hexes)
    # Row 1: 0.5, 1.5, 2.5, 3.5 (shifted by sqrt(3)/2)
    # Row 2: 0, 1, 2, 3
    # Total 12.
    
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3)
            y = r * 1.5
            if r % 2 == 1:
                x += math.sqrt(3) / 2
            centers.append((x, y))
            
    # Center the cluster
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N
    initial_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Run simulation
    result = simulate_shrink(initial_centers, [0.0]*N, 10.0)
    
    if result:
        positions, angles, outer_center, side = result
        inner = []
        for i in range(N):
            inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
        
        # Add safety margin
        final_side = side * 1.01 + 1e-4
        
        return inner, (float(outer_center[0]), float(outer_center[1])), float(final_side), 0.0
    else:
        # Fallback: Return the initial grid if simulation failed
        inner = [[float(x), float(y), 0.0] for x, y in initial_centers]
        # Calculate side for this grid
        side = compute_outer_side(initial_centers, [0.0]*N, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), float(side * 1.01 + 1e-4), 0.0

# EVOLVE-BLOCK-END
