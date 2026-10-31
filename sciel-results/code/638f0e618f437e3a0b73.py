import numpy as np
import random
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Normals for the hexagon sides
    # A regular hexagon centered at origin with flat-top orientation
    # Vertices at (S, 0), (S/2, S*sqrt(3)/2), etc.
    # Normals are perpendicular to sides.
    # Side 1: vertical? No, flat-top means horizontal top and bottom.
    # Let's use pointy-top for simplicity? 
    # Actually, orientation of container doesn't matter due to symmetry, 
    # but we must be consistent with the normals.
    # Let's define the container as a regular hexagon with vertices at angles 0, 60, 120...
    # Then the sides are at angles 30, 90, 150...
    # The normals to the sides point in directions 30, 90, 150, 210, 270, 330 degrees.
    # n_j = (cos(theta_j), sin(theta_j))
    
    angles = [math.pi/6 + i * math.pi/3 for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angles])
    
    def get_M(centers):
        # centers: (N, 2) array
        # Project centers onto normals
        # proj[i, j] = dot(centers[i], normals[j])
        projs = centers @ normals.T
        # We need max |proj|
        return np.max(np.abs(projs))
    
    def check_overlap(centers, i, new_pos, eps=1e-6):
        # Check if new_pos overlaps with any center j != i
        # Distance must be >= sqrt(3)
        for j in range(N):
            if i == j: continue
            diff = new_pos - centers[j]
            dist = np.linalg.norm(diff)
            if dist < math.sqrt(3) - eps:
                return False
        return True

    # Initialize positions
    # Use a hexagonal lattice
    # Basis vectors for triangular lattice with spacing sqrt(3)
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    v1 = np.array([math.sqrt(3), 0.0])
    v2 = np.array([math.sqrt(3)/2, 1.5]) # 3/2 = 1.5
    
    candidates = []
    # Generate a grid of candidates around origin
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * v1 + j * v2
            candidates.append(p)
            
    # Sort by distance from origin
    candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Take first N
    centers = np.array(candidates[:N])
    
    # Local Search / Simulated Annealing
    best_centers = centers.copy()
    best_M = get_M(best_centers)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    steps = 5000
    step_size = 0.5
    
    current_M = best_M
    current_centers = best_centers.copy()
    
    import time
    start_time = time.time()
    timeout = 10.0 # Keep well under 120s
    
    for step in range(steps):
        if time.time() - start_time > timeout:
            break
            
        # Pick random index
        i = random.randint(0, N-1)
        
        # Propose move
        # Random direction and magnitude
        angle = random.uniform(0, 2*math.pi)
        mag = random.uniform(0, step_size)
        delta = np.array([math.cos(angle), math.sin(angle)]) * mag
        new_pos = current_centers[i] + delta
        
        # Check overlap
        if check_overlap(current_centers, i, new_pos):
            # Calculate new M
            old_pos = current_centers[i].copy()
            current_centers[i] = new_pos
            new_M = get_M(current_centers)
            
            # Accept/Reject
            if new_M < current_M or random.random() < math.exp(-(new_M - current_M) / T):
                current_M = new_M
                if new_M < best_M:
                    best_M = new_M
                    best_centers = current_centers.copy()
            else:
                current_centers[i] = old_pos
        else:
            # If overlap, try to push away? Or just reject.
            # Rejecting is simpler and safer.
            pass
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Optionally restart with smaller step size or break
            step_size *= 0.95

    # Final calculation
    # M is max |n . c|
    # S * sqrt(3)/2 - sqrt(3)/2 >= M
    # S >= 1 + 2*M/sqrt(3)
    side_length = 1.0 + 2.0 * best_M / math.sqrt(3)
    
    # Convert centers to list of [x, y, angle]
    inner = []
    for c in best_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(side_length), 0.0

# EVOLVE-BLOCK-END
