"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a packing of N circles in a unit square using force-directed relaxation.
    Strategy:
    1. Initialize with a hexagonal lattice pattern (optimal for infinite plane, good for finite).
    2. Apply repulsive forces between circles and attractive forces towards the center/bounds.
    3. Iterate until time budget is exhausted or convergence is reached.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # 1. Initialization: Hexagonal lattice
    # Approximate number of columns to fit N points
    cols = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
    rows = int(np.ceil(N / cols))
    
    # Hexagonal spacing
    # For a hexagonal packing, horizontal distance is d, vertical is d*sqrt(3)/2
    # We want to fit this into [-0.5, 0.5]^2
    # Let's start with a rough scale and let relaxation adjust it.
    # Initial scale based on area: Area per circle ~ 1/N. Radius ~ 0.5/sqrt(N)
    # Hex lattice area per point = d * d * sqrt(3)/2
    # d^2 * sqrt(3)/2 = 1/N => d = sqrt(2 / (N * sqrt(3)))
    d_init = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= N:
                break
            x = (j - cols / 2.0 + 0.5) * d_init
            y = (i - rows / 2.0 + 0.5) * d_init * np.sqrt(3.0)
            # Shift odd rows by d_init / 2 for hexagonal offset
            if i % 2 == 1:
                x += d_init / 2.0
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # 2. Force-Directed Relaxation
    # Objective: Maximize minimum distance.
    # We use a repulsive force model: F = 1/r^2 for pairs, and a soft boundary constraint.
    # We want to maximize the minimum distance, which is equivalent to minimizing the sum of inverse distances or similar.
    # However, simple gradient descent on min-distance is non-smooth.
    # Instead, we simulate repulsion:
    # - Repulsion between all pairs (strong at short range)
    # - Attraction to center (to keep them in)
    # - Hard clipping to bounds (or soft penalty)
    
    # Parameters
    learning_rate = 0.01
    decay = 0.999
    min_dist_target = 0.01 # Avoid division by zero
    
    # Precompute indices for efficiency? N=200 is small enough for O(N^2) per step.
    # 200^2 = 40,000 operations per step. 
    # If we can do 10,000 steps in 145s, that's 1.45us per step. 
    # 40k ops in 1.45us is ~27M ops/sec. NumPy vectorization helps.
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask self-distances
        np.fill_diagonal(dists, 1e9)
        
        # Repulsive force magnitude: 1 / dist^2 (capped)
        # To avoid explosion, cap the force
        force_mag = 1.0 / (dists ** 2)
        force_mag = np.minimum(force_mag, 100.0) # Cap force
        
        # Force vector
        forces = diff * force_mag[:, :, np.newaxis]
        
        # Sum forces for each particle
        net_forces = np.sum(forces, axis=1)
        
        # Boundary forces: Push towards center if too close to edge
        # Soft boundary: If x > 0.5 - r_est, push left. 
        # Since r is not known exactly, we use a soft potential.
        # Let's use a simple spring to center to keep them compact, 
        # but the repulsion will expand them.
        # A better boundary handling: 
        # If a particle is outside [-0.5, 0.5], pull it back strongly.
        # Or, use a penalty in the force.
        
        # Let's add a centering force to prevent drift and keep them inside.
        # F_center = -k * center
        k_center = 0.01
        center_forces = -k_center * centers
        
        # Update velocities/positions (simple Euler integration)
        # We don't track velocity, just use gradient descent style update
        centers = centers + learning_rate * (net_forces + center_forces)
        
        # Hard clipping to ensure validity (centers must be in [-0.5, 0.5])
        # Note: Clipping can cause stacking, but repulsion will separate them.
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay learning rate to allow convergence
        learning_rate *= decay
        
        # Check for early stopping if learning rate is too small
        if learning_rate < 1e-6:
            break
            
    # Final cleanup: Ensure strict validity
    # The clipping ensures they are inside. 
    # We return the centers.
    return centers

# EVOLVE-BLOCK-END
