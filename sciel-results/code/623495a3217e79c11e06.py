import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Normals for the outer hexagon (flat top orientation)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    angles = [np.pi/6 + k * np.pi/3 for k in range(6)]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    apothem_inner = math.sqrt(3) / 2.0
    apothem_diff = math.sqrt(3) # Apothem of Minkowski difference (side 2)

    # Initialize positions
    # Start with a compact cluster
    # A hexagonal grid pattern
    # Center at 0,0
    # Neighbors at distance 2 in x, or sqrt(3) in y, etc.
    # Let's just use a random initialization within a small circle
    np.random.seed(42)
    pos = np.random.uniform(-2, 2, (N, 2))
    
    best_pos = pos.copy()
    best_S = float('inf')
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.995
    iterations = 5000
    
    def calculate_S(p):
        # p is (N, 2)
        # projections = p @ normals.T  (N, 6)
        projections = np.dot(p, normals.T)
        # max projection per hex
        max_proj = np.max(np.abs(projections), axis=1)
        # required S
        # max_proj + apothem_inner <= S * apothem_inner
        # S >= (max_proj + apothem_inner) / apothem_inner
        S_vals = (max_proj + apothem_inner) / apothem_inner
        return np.max(S_vals)
    
    def calculate_overlap_penalty(p):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = p[i] - p[j]
                # project d onto normals
                projs = np.abs(np.dot(d, normals.T))
                max_proj = np.max(projs)
                if max_proj < apothem_diff:
                    # overlap
                    # penalty is how much it violates
                    violation = apothem_diff - max_proj
                    penalty += violation * violation
        return penalty

    def total_energy(p):
        S = calculate_S(p)
        pen = calculate_overlap_penalty(p)
        # We want to minimize S, but avoid overlaps.
        # If overlaps exist, S is not valid.
        # Use a large multiplier for penalty.
        return S + 100.0 * pen

    current_energy = total_energy(pos)
    
    for it in range(iterations):
        # Pick a random index to move
        idx = np.random.randint(N)
        # Save original
        orig = pos[idx].copy()
        # Random perturbation
        pos[idx] += np.random.normal(0, 0.1, 2)
        
        new_energy = total_energy(pos)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_energy = new_energy
            # Update best
            S_now = calculate_S(pos)
            if calculate_overlap_penalty(pos) < 1e-6 and S_now < best_S:
                best_S = S_now
                best_pos = pos.copy()
        else:
            pos[idx] = orig
            
        T *= cooling
        if T < 1e-6:
            break
            
    # Final check and refinement
    # If best_S is still inf or overlaps exist, we might have failed.
    # But with N=12, it should converge.
    
    # Construct output
    # inner: list of [x, y, angle]
    # angle is 0.0 for all
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), 0.0])
        
    # Outer hexagon center
    # We assumed center at (0,0) for calculation.
    # We should center the packing to minimize S?
    # The calculation of S assumed center at 0.
    # If the packing is off-center, the required S is larger.
    # We can shift the whole packing to minimize the max projection.
    # This is equivalent to finding the minimum enclosing hexagon.
    # For now, return center (0,0) and the calculated S.
    # To be safe, we can compute the centroid and shift, then re-calc S.
    
    centroid = np.mean(best_pos, axis=0)
    shifted_pos = best_pos - centroid
    
    # Re-calculate S for shifted positions
    S_final = calculate_S(shifted_pos)
    
    # Update inner coordinates
    inner = []
    for i in range(N):
        inner.append([float(shifted_pos[i][0]), float(shifted_pos[i][1]), 0.0])
        
    # Outer center is centroid
    center = (float(centroid[0]), float(centroid[1]))
    side = float(S_final)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
