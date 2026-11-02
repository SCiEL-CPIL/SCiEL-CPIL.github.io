"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Simulated Annealing to optimize positions and radii.
    """
    best_result = None
    best_score = -np.inf
    
    # Try a few different initial configurations and aspect ratios
    # Perimeter = 4 => w + h = 2.
    # Let's try optimizing the aspect ratio as well or fixing it to optimal candidates.
    # Common optimal aspect ratios for packing are often near 1 or slightly off.
    
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for 120s limit
    
    # Define candidate aspect ratios (w/h)
    # w = 2*r/(1+r), h = 2/(1+r)
    aspects = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    for aspect in aspects:
        if time.time() - start_time > time_limit:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Initialize
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            # Add small random jitter to break symmetry
            positions[i, 0] = (col + 0.5 + random.uniform(-0.1, 0.1)) * w / side
            positions[i, 1] = (row + 0.5 + random.uniform(-0.1, 0.1)) * h / side
            # Initial radius: small fraction of cell size
            cell_w = w / side
            cell_h = h / side
            max_r = min(cell_w, cell_h) * 0.4
            radii[i] = max_r * random.uniform(0.5, 1.0)
        
        # Simulated Annealing
        result = _simulate_annealing(positions, radii, w, h, N, time_limit - (time.time() - start_time))
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback: simple grid packing in 1x1 square (perimeter 4)
        # This is a valid packing, score will be low but valid.
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * w / side
            y = (row + 0.5) * h / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

def _simulate_annealing(positions, radii, w, h, N, time_limit):
    """
    Run Simulated Annealing to optimize circle packing.
    """
    start_time = time.time()
    
    # Current state
    curr_pos = positions.copy()
    curr_rad = radii.copy()
    
    # Best state found so far
    best_pos = curr_pos.copy()
    best_rad = curr_rad.copy()
    best_score = -np.inf
    
    # Temperature
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.9999
    
    # Penalty weight
    K = 100.0
    
    def calculate_score(pos, rad):
        """
        Calculate the objective function value.
        We want to maximize sum(rad) subject to constraints.
        Score = sum(rad) - K * (violations)
        """
        score = np.sum(rad)
        
        # Boundary violations
        for i in range(N):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x < r:
                score -= K * (r - x) ** 2
            if x > w - r:
                score -= K * (x - (w - r)) ** 2
            if y < r:
                score -= K * (r - y) ** 2
            if y > h - r:
                score -= K * (y - (h - r)) ** 2
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    score -= K * (min_dist - dist) ** 2
        
        return score
    
    # Initial score
    curr_score = calculate_score(curr_pos, curr_rad)
    if curr_score > best_score:
        best_score = curr_score
        best_pos = curr_pos.copy()
        best_rad = curr_rad.copy()
        
    # Precompute pairs for efficiency? N=21 is small, so O(N^2) is fine per eval.
    # But we can speed up overlap check.
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a new state
        new_pos = curr_pos.copy()
        new_rad = curr_rad.copy()
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Move a circle
            i = random.randint(0, N - 1)
            dx = random.gauss(0, 0.02)
            dy = random.gauss(0, 0.02)
            new_pos[i, 0] += dx
            new_pos[i, 1] += dy
        elif move_type < 0.9:
            # Change radius of a circle
            i = random.randint(0, N - 1)
            dr = random.gauss(0, 0.01)
            new_rad[i] += dr
            # Ensure radius is positive
            if new_rad[i] < 0.01:
                new_rad[i] = 0.01
        else:
            # Swap positions of two circles (helps with symmetry)
            i, j = random.sample(range(N), 2)
            new_pos[i], new_pos[j] = new_pos[j].copy(), new_pos[i].copy()
            # Note: Radii stay with their "identity" or swap? 
            # Usually we swap the entire circle (pos+rad) or just pos.
            # If we swap pos only, the radii remain at their original indices.
            # Let's swap the entire state (pos and rad) to treat them as indistinguishable particles?
            # No, the objective is sum(radii), so indices don't matter for the value, 
            # but for the optimization landscape, swapping identities might help escape local minima.
            # Let's swap pos and rad together.
            new_rad[i], new_rad[j] = new_rad[j], new_rad[i]
            
        # Clamp positions to valid range roughly to avoid huge penalties
        for i in range(N):
            r = new_rad[i]
            new_pos[i, 0] = np.clip(new_pos[i, 0], r, w - r)
            new_pos[i, 1] = np.clip(new_pos[i, 1], r, h - r)
            
        new_score = calculate_score(new_pos, new_rad)
        
        delta_score = new_score - curr_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta_score > 0 or random.random() < np.exp(delta_score / T):
            curr_pos = new_pos
            curr_rad = new_rad
            curr_score = new_score
            
            if curr_score > best_score:
                best_score = curr_score
                best_pos = curr_pos.copy()
                best_rad = curr_rad.copy()
        
        # Cool down
        T *= cooling_rate
        
    # Verify best result
    # Check if best state is valid (no overlaps, inside bounds)
    valid = True
    for i in range(N):
        x, y, r = best_pos[i, 0], best_pos[i, 1], best_rad[i]
        if r <= 0:
            valid = False
            break
        if x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                dx = best_pos[i, 0] - best_pos[j, 0]
                dy = best_pos[i, 1] - best_pos[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < best_rad[i] + best_rad[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
    if valid:
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [best_pos[i, 0], best_pos[i, 1], best_rad[i]]
        return result
    else:
        return None

# EVOLVE-BLOCK-END
