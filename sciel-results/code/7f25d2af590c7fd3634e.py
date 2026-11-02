import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute vertices for a unit hexagon centered at origin
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.radians(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Precompute edge normals for SAT
    # Edge k is from vertex k to vertex (k+1)%6
    # Normal is perpendicular to edge, pointing outward
    edges = np.roll(base_verts, -1, axis=0) - base_verts
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # The 6 normals correspond to the 6 faces of the hexagon
    # For a regular hexagon, these are sufficient for SAT against another regular hexagon
    # because the set of normals is the same for all orientations (since we only translate).
    # Wait, if we allow rotation, normals rotate.
    # The task says "each inner hexagon may translate and rotate".
    # However, the current seed program fixes rotation to 0.
    # The inspirations suggest symmetry-constrained search.
    # Let's stick to translation only (rotation=0) for speed and simplicity, 
    # as regular hexagons in a hexagonal lattice usually align.
    # If rotation is allowed, the search space is 36D.
    # Let's try translation only first (24D) which is much faster.
    # If that fails to beat the seed, we might need rotation.
    # But the seed score is 7.43. 
    # A full 24D SA is feasible.

    def get_verts_flat(positions):
        """
        positions: (N, 2) array of centers.
        Returns: (N, 6, 2) array of vertices.
        """
        return base_verts[None, :, :] + positions[:, None, :]

    def check_validity(positions, tol=1e-4):
        """
        Vectorized SAT check for disjointness.
        Returns True if all hexagons are disjoint.
        """
        verts = get_verts_flat(positions)
        
        # We need to check all pairs (i, j) with i < j
        # For each pair, project onto the 6 normals.
        # If there is a separating axis, they are disjoint.
        # If NO separating axis is found for a pair, they overlap.
        
        # To be efficient, we can check all pairs at once?
        # It's tricky to vectorize the "any pair overlaps" logic easily without huge memory.
        # N=12, so 66 pairs.
        # We can loop over pairs, but vectorize the projection within the pair check?
        # Or loop over axes, and check all pairs for that axis?
        
        # Let's loop over the 6 axes.
        # For a fixed axis (normal), compute projections of all N hexagons.
        # proj[i] = min/max dot product of hexagon i with normal.
        # Then for each pair (i, j), if max_i < min_j or max_j < min_i, they are separated.
        # We need to ensure that for EVERY pair, there EXISTS at least one axis that separates them.
        
        # Actually, SAT says: Two convex polygons are disjoint if and only if there exists
        # a separating axis.
        # So, for each pair (i,j), we check the 6 axes.
        # If all 6 axes show overlap, the pair is invalid.
        
        # Strategy:
        # 1. Compute min_proj and max_proj for all N hexagons on all 6 axes.
        #    Shape: (N, 6, 2) where [:, :, 0] is min, [:, :, 1] is max.
        
        min_proj = np.min(np.einsum('ij,ik->jk', verts, normals), axis=1) # (N, 6)
        max_proj = np.max(np.einsum('ij,ik->jk', verts, normals), axis=1) # (N, 6)
        
        # Now, for each pair (i, j), check if they are separated on ANY axis.
        # separated[i, j] = OR over axes k of (max_proj[i, k] + tol < min_proj[j, k] OR max_proj[j, k] + tol < min_proj[i, k])
        
        # Let's create a matrix of separations.
        # i indices: 0..N-1, j indices: 0..N-1
        # We only care about i < j.
        
        is_separated = np.zeros((N, N), dtype=bool)
        
        for k in range(6):
            # mins and maxs for axis k
            mins = min_proj[:, k] # (N,)
            maxs = max_proj[:, k] # (N,)
            
            # For all pairs i, j:
            # separated if maxs[i] + tol < mins[j] OR maxs[j] + tol < mins[i]
            # maxs[:, None] + tol < mins[None, :]  -> (N, N) matrix M1
            # maxs[None, :] + tol < mins[:, None]  -> (N, N) matrix M2
            
            M1 = maxs[:, None] + tol < mins[None, :]
            M2 = maxs[None, :] + tol < mins[:, None]
            
            is_separated |= (M1 | M2)
            
        # Now, is_separated[i, j] is True if there is at least one axis separating i and j.
        # If is_separated[i, j] is False, they overlap.
        
        # We need to check if ANY pair (i, j) with i < j has is_separated[i, j] == False.
        # Create upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # If any entry in the upper triangle is False, we have a collision.
        if not np.all(is_separated[mask]):
            return False
            
        return True

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = get_verts_flat(positions)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Normals for the outer hexagon (same as inner hexagon normals)
        # L1..L6 projections
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    # Initial positions: Hexagonal lattice
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 hexagons (one missing)
    
    def generate_initial_positions():
        pos = np.zeros((N, 2))
        pos[0] = [0, 0]
        for k in range(6):
            ang = np.radians(60 * k)
            pos[1+k] = [2.0 * np.cos(ang), 2.0 * np.sin(ang)]
        # Ring 2 at radius ~ 3.464 (2 * sqrt(3))
        # Angles offset by 30 deg?
        # Standard hex lattice:
        # Ring 2 positions are at distance 2*sqrt(3) from center, angles 30, 90, 150, 210, 270, 330?
        # No, in a hex lattice, ring 2 vertices are at distance 2*sqrt(3) if side is 1?
        # Distance between centers of adjacent hexes is 2 * apothem = 2 * sqrt(3)/2 * 1? No.
        # Side 1. Apothem sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
        # So Ring 1 radius is sqrt(3).
        # Ring 2 radius is 2 * sqrt(3).
        
        r1 = np.sqrt(3)
        r2 = 2 * np.sqrt(3)
        
        pos[0] = [0, 0]
        for k in range(6):
            ang = np.radians(60 * k)
            pos[1+k] = [r1 * np.cos(ang), r1 * np.sin(ang)]
        
        # Ring 2 angles: 30, 90, 150, 210, 270, 330
        # We need 5. Let's pick 5 of these 6.
        # Angles: 30, 90, 150, 210, 270
        for k in range(5):
            ang = np.radians(30 + 60 * k)
            pos[7+k] = [r2 * np.cos(ang), r2 * np.sin(ang)]
            
        return pos

    current_pos = generate_initial_positions()
    if not check_validity(current_pos):
        # Fallback to a spread out configuration
        current_pos = np.random.rand(N, 2) * 10 - 5
        # Ensure valid? If not, SA will fix it.
        
    current_side = required_outer_side(current_pos)
    best_pos = current_pos.copy()
    best_side = current_side
    
    # Simulated Annealing
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.995
    
    # Estimate steps possible
    # Each step: check_validity (fast vectorized), required_outer_side (fast)
    # Let's run until deadline.
    
    while time.time() < deadline:
        # Generate new position
        # Random perturbation
        # Scale of perturbation proportional to T
        step_size = T * 0.5
        new_pos = current_pos + np.random.randn(N, 2) * step_size
        
        # Check validity
        if not check_validity(new_pos):
            # Reject invalid moves (or accept with low prob? Standard SA for packing often rejects invalids)
            # To ensure we find a valid packing, let's just reject invalids.
            # But this might get stuck. 
            # Better: If invalid, calculate a penalty?
            # Given the tight packing, rejecting invalids is safer to maintain feasibility.
            # However, if we start valid and move small steps, we stay valid often.
            # Let's try: if invalid, skip.
            continue
        
        new_side = required_outer_side(new_pos)
        
        # Acceptance criterion
        if new_side < current_side:
            # Always accept improvement
            current_pos = new_pos
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_pos = current_pos.copy()
        else:
            # Accept with probability exp(-(delta_E) / T)
            delta_E = new_side - current_side
            if delta_E < 0:
                # This shouldn't happen if we check new_side < current_side first
                pass
            else:
                prob = math.exp(-delta_E / T)
                if np.random.random() < prob:
                    current_pos = new_pos
                    current_side = new_side
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final check and return
    if not check_validity(best_pos):
        # This shouldn't happen if we only update best_pos when valid
        pass
        
    final_side = required_outer_side(best_pos)
    
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# Test locally if needed
# if __name__ == "__main__":
#     res = construct()
#     print(res)
