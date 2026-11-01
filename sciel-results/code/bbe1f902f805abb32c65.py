"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_radius(centers):
    """Compute the maximum valid radius for a given set of centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance from origin to each center
    dists = np.sqrt(np.sum(centers**2, axis=1))
    # Minimum distance from center to boundary of unit circle
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    # Minimum pairwise distance between centers
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_pair:
                min_pair = d
    min_pair = min_pair / 2.0
    return min(min_boundary, min_pair)

def _initial_hexagonal():
    """Generate initial positions using a hexagonal lattice pattern."""
    # Use a hexagonal packing pattern
    # Start with a reasonable spacing
    best_centers = None
    best_radius = 0.0
    
    # Try different spacings for hexagonal lattice
    for spacing in np.linspace(0.2, 0.35, 15):
        centers = []
        # Generate hexagonal lattice points
        row_spacing = spacing * np.sqrt(3) / 2
        for row in range(-20, 21):
            y = row * row_spacing
            offset = (row % 2) * spacing / 2
            for col in range(-20, 21):
                x = col * spacing + offset
                if np.sqrt(x**2 + y**2) <= 1.0 - spacing/2:
                    centers.append([x, y])
        
        centers = np.array(centers)
        if len(centers) >= N:
            # Select N points that maximize the minimum distance
            # Simple approach: take first N after sorting by radius
            radii = np.sqrt(np.sum(centers**2, axis=1))
            idx = np.argsort(radii)[:N]
            selected = centers[idx]
            r = _compute_radius(selected)
            if r > best_radius:
                best_radius = r
                best_centers = selected.copy()
    
    if best_centers is None:
        # Fallback: random positions
        best_centers = np.random.rand(N, 2) * 0.8 - 0.4
        best_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius

def _simulated_annealing(centers, initial_temp=0.01, final_temp=0.0001, 
                          iterations_per_temp=100, n_temps=1000):
    """Simulated annealing to improve the packing."""
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    temps = np.geomspace(initial_temp, final_temp, n_temps)
    
    for temp in temps:
        for _ in range(iterations_per_temp):
            # Random perturbation
            new_centers = current_centers.copy()
            i = np.random.randint(N)
            dx = np.random.normal(0, temp)
            dy = np.random.normal(0, temp)
            new_centers[i] += np.array([dx, dy])
            
            # Ensure points stay in unit circle
            dist = np.sqrt(np.sum(new_centers**2, axis=1))
            if np.max(dist) > 1.0:
                # Scale back
                scale = 0.99 / np.max(dist)
                new_centers = new_centers * scale
            
            new_radius = _compute_radius(new_centers)
            
            # Accept or reject
            if new_radius >= current_radius:
                current_centers = new_centers
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
            else:
                # Accept with probability exp((new - current) / temp)
                prob = np.exp((new_radius - current_radius) / temp)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_radius = new_radius
    
    return best_centers, best_radius

def _gradient_descent(centers, n_steps=10000, lr=0.001):
    """Gradient descent on the objective (maximize minimum radius)."""
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    for step in range(n_steps):
        # Compute gradient of the radius with respect to each center
        # The radius is min(min_pair_dist/2, min_boundary)
        # We need to find which constraint is active
        
        dists = np.sqrt(np.sum(current_centers**2, axis=1))
        min_boundary = 1.0 - np.max(dists)
        
        # Find the closest pair
        min_pair_dist = np.inf
        min_pair_i, min_pair_j = 0, 1
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair_i, min_pair_j = i, j
        
        min_pair = min_pair_dist / 2.0
        
        # Determine which constraint is active
        if min_boundary < min_pair:
            # Boundary constraint is active
            # Push the point with max distance inward
            i_max = np.argmax(dists)
            grad = np.zeros((N, 2))
            if dists[i_max] > 0:
                grad[i_max] = -current_centers[i_max] / dists[i_max]
        else:
            # Pair constraint is active
            # Push the two closest points apart
            grad = np.zeros((N, 2))
            diff = current_centers[min_pair_i] - current_centers[min_pair_j]
            d = np.linalg.norm(diff)
            if d > 0:
                grad[min_pair_i] = diff / d
                grad[min_pair_j] = -diff / d
        
        # Update
        new_centers = current_centers + lr * grad
        # Keep in unit circle
        new_dists = np.sqrt(np.sum(new_centers**2, axis=1))
        mask = new_dists > 1.0
        if np.any(mask):
            new_centers[mask] *= (0.99 / new_dists[mask])[:, None]
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            # Decay learning rate
            lr *= 0.999
    
    return best_centers, best_radius

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with hexagonal lattice initialization
    centers, current_radius = _initial_hexagonal()
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Phase 1: Simulated annealing
    while time.time() < deadline - 30:
        centers, radius = _simulated_annealing(
            best_centers, 
            initial_temp=0.005, 
            final_temp=0.0001,
            iterations_per_temp=50,
            n_temps=500
        )
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        else:
            break
    
    # Phase 2: Gradient descent refinement
    while time.time() < deadline - 10:
        centers, radius = _gradient_descent(best_centers, n_steps=5000, lr=0.0005)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        else:
            break
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centers
# EVOLVE-BLOCK-END
