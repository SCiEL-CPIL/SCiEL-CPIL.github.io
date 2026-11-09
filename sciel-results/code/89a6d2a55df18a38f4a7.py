"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START

def _score(centres):
    """Compute the minimum radius: min of half-pair-distances and wall distances."""
    # Wall distances
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    
    # Pair distances
    n = len(centres)
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Exclude self-distances
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)


def _construct_grid():
    """Create initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_honeycomb():
    """Create a honeycomb-like initial configuration for better packing."""
    # Use a hexagonal lattice pattern
    centres = []
    # Try different offsets to find a good configuration
    for row in range(8):
        for col in range(8):
            if len(centres) >= N:
                break
            # Hexagonal offset
            x = (col - 3.5) * 0.25
            y = (row - 3.5) * 0.2165  # sqrt(3)/2 * 0.25
            if col % 2 == 1:
                y += 0.10825
            # Check if within bounds
            if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                centres.append([x, y])
    
    if len(centres) < N:
        # Fill remaining with random positions
        while len(centres) < N:
            centres.append([random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45)])
    
    return np.array(centres[:N])


def _anneal(centres, max_time=100.0, seed=42):
    """Simulated annealing optimization."""
    np.random.seed(seed)
    random.seed(seed)
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Temperature schedule
    T = 0.01
    T_min = 1e-8
    cooling = 0.999
    
    n = len(centres)
    
    while time.time() < deadline:
        # Choose a random circle to move
        i = np.random.randint(n)
        
        # Generate a random displacement
        scale = T * 10
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        new_centre = centres[i].copy()
        new_centre[0] += dx
        new_centre[1] += dy
        
        # Keep within bounds (with margin)
        new_centre[0] = np.clip(new_centre[0], -0.49, 0.49)
        new_centre[1] = np.clip(new_centre[1], -0.49, 0.49)
        
        old_centre = centres[i].copy()
        centres[i] = new_centre
        
        new_score = _score(centres)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centres = centres.copy()
        else:
            centres[i] = old_centre
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_centres, best_score


def _polish(centres, max_time=20.0):
    """Local optimization pass to refine the solution."""
    np.random.seed(123)
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(centres)
    
    # Use a smaller step size for polishing
    step_size = 0.005
    
    while time.time() < deadline:
        # Try to improve each circle
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_centre = centres[i].copy()
            improved = False
            
            # Try small moves in multiple directions
            for _ in range(50):
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
                
                new_centre = old_centre.copy()
                new_centre[0] += dx
                new_centre[1] += dy
                
                # Keep within bounds
                new_centre[0] = np.clip(new_centre[0], -0.49, 0.49)
                new_centre[1] = np.clip(new_centre[1], -0.49, 0.49)
                
                centres[i] = new_centre
                new_score = _score(centres)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_centres = centres.copy()
                else:
                    centres[i] = old_centre
            
            if not improved:
                # No improvement in this iteration, try smaller steps
                step_size *= 0.9
                if step_size < 1e-6:
                    break
    
    return best_centres, best_score


def construct():
    """Construct the optimal packing of 50 equal circles in a unit square."""
    start_time = time.time()
    total_deadline = start_time + 145.0  # Leave 5s margin for safety
    
    # Try multiple initial configurations
    initial_configs = [
        _construct_grid(),
        _construct_honeycomb(),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
    ]
    
    best_centres = None
    best_score = -np.inf
    
    # Phase 1: Annealing from different starts
    for config in initial_configs:
        if time.time() > total_deadline - 25.0:
            break
        
        anneal_time = min(30.0, total_deadline - time.time() - 25.0)
        if anneal_time <= 0:
            break
        
        centres, score = _anneal(config, max_time=anneal_time, seed=np.random.randint(10000))
        
        if score > best_score:
            best_score = score
            best_centres = centres
    
    if best_centres is None:
        return _construct_grid()
    
    # Phase 2: Polish the best result
    polish_time = min(15.0, total_deadline - time.time() - 5.0)
    if polish_time > 0:
        polished_centres, polished_score = _polish(best_centres, max_time=polish_time)
        if polished_score > best_score:
            best_score = polished_score
            best_centres = polished_centres
    
    # Ensure all centres are strictly inside the square
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres

# EVOLVE-BLOCK-END
