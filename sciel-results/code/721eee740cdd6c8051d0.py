import math
import time
import numpy as np

N = 11
# Distance between centers of touching aligned unit hexagons
MIN_DIST = math.sqrt(3)

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Normals for the 6 sides of a regular hexagon aligned with flat top/bottom?
    # Let's assume vertices at 0, 60, 120... degrees.
    # Then sides are perpendicular to 30, 90, 150, 210, 270, 330 degrees.
    angles = [math.radians(30 + 60*k) for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angles])
    
    # Initialize positions in a honeycomb lattice
    # Row 0: 3 points
    # Row 1: 4 points
    # Row 2: 4 points
    # This is a rough guess. A better init is a tight cluster.
    # Let's use a simple spiral or lattice.
    # Lattice vectors:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2) ? No, distance between rows is 1.5?
    # In a hex lattice of unit hexagons:
    # x spacing: sqrt(3)
    # y spacing: 1.5
    # offset: sqrt(3)/2
    
    pos = []
    # Generate a 3x4 block of lattice points and pick 11 closest to center
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * math.sqrt(3) + (j % 2) * math.sqrt(3)/2
            y = j * 1.5
            candidates.append([x, y])
            
    # Sort by distance to origin and take 11
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    pos = np.array(candidates[:N])
    
    # Optimization: Simulated Annealing / Gradient Descent
    # Objective: Minimize M = max_k (n_k . p_i)
    # Constraint: |p_i - p_j| >= MIN_DIST
    
    def get_M(p):
        # p is (N, 2)
        # project p onto normals
        projections = p @ normals.T # (N, 6)
        # max projection for each point
        max_projs = np.max(projections, axis=1)
        # global max
        return np.max(max_projs)
    
    def get_overlap_penalty(p):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                if d < MIN_DIST:
                    # Quadratic penalty for overlap
                    penalty += (MIN_DIST - d)**2
        return penalty

    # Annealing
    T = 0.1
    T_min = 1e-4
    alpha = 0.999
    
    best_M = get_M(pos)
    best_pos = pos.copy()
    
    while time.time() < deadline and T > T_min:
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Attraction to center / minimization of M
        # Gradient of M is roughly the normal corresponding to the max projection
        # For simplicity, use a heuristic: pull points towards origin if they are far
        # Or use sub-gradient: for each point, if it determines the max M, push it in opposite of normal
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < MIN_DIST and d > 1e-6:
                    # Repulsive force
                    f = (MIN_DIST - d) / d
                    forces[i] += f * d_vec
                    forces[j] -= f * d_vec
                elif d <= 1e-6:
                    # Jitter
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1
        
        # Boundary force: Push points in if they exceed current M
        # We want to minimize M, so we push points inward
        # Approximate gradient:
        # If point i is at the boundary (max projection), push it opposite to that normal
        projections = pos @ normals.T
        max_idx = np.argmax(projections, axis=1)
        
        for i in range(N):
            # If this point is on the boundary of the bounding hexagon
            # We can push it towards the center
            # Simple heuristic: pull all points towards center slightly
            # But this might break symmetry.
            # Better: only push points that are "active" in the max
            if projections[i, max_idx[i]] > best_M - 0.01:
                n = normals[max_idx[i]]
                forces[i] -= 0.1 * n # Push inward
        
        # Update positions
        pos = pos + forces * T + np.random.randn(N, 2) * T * 0.1
        
        # Check overlap penalty to reject bad moves (Metropolis)
        current_M = get_M(pos)
        current_pen = get_overlap_penalty(pos)
        
        # We want to minimize M, but allow overlaps temporarily if it helps M?
        # No, overlaps are invalid. We should penalize them heavily.
        # Score = M + 10 * current_pen
        score = current_M + 10.0 * current_pen
        best_score = best_M + 10.0 * get_overlap_penalty(best_pos)
        
        if score < best_score or np.random.rand() < math.exp(-(score - best_score) / T):
            if current_pen < 1e-6: # Only accept if no overlap
                best_M = current_M
                best_pos = pos.copy()
            elif score < best_score: # Accept if better even with overlap? No, keep best valid
                pass
        
        T *= alpha
        
    # Final polish: if best_pos has overlaps, try to fix them
    # For simplicity, if overlaps exist, we might need a more robust solver.
    # But the lattice init is usually overlap-free.
    
    # Calculate S
    # M = r_in - apothem
    # r_in = S * sqrt(3) / 2
    # apothem = sqrt(3) / 2
    # M = S * sqrt(3) / 2 - sqrt(3) / 2
    # S = (M + sqrt(3)/2) * 2 / sqrt(3)
    
    # Use best_M which corresponds to best_pos
    # Ensure best_pos is valid (no overlaps)
    # If best_pos has overlaps, we might need to use a valid fallback.
    # Given the init is valid and forces are repulsive, it should stay valid.
    
    S = (best_M + math.sqrt(3)/2) * 2 / math.sqrt(3)
    
    # Construct inner list
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (0.0, 0.0), float(S), 0.0
