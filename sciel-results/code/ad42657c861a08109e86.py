"""Circle packing in perimeter-4 rectangle: maximize sum of 21 circle radii."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Initial configuration: grid-based
    def initial_config():
        # Start with a grid arrangement, aspect ratio ~1.0
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            centers[i, 0] = (col + 0.5) * w / side
            centers[i, 1] = (row + 0.5) * h / side
            radii[i] = 0.5 * w / side - 0.005
        return centers, radii, w, h
    
    def energy(centers, radii, w, h):
        """Compute penalty: overlap + out-of-bounds + negative radius."""
        penalty = 0.0
        
        # Check bounds
        for i in range(N):
            r = radii[i]
            if r < 0:
                penalty += abs(r) * 10
            x, y = centers[i]
            if x - r < 0 or x + r > w:
                penalty += max(0, r - x) + max(0, x + r - w)
            if y - r < 0 or y + r > h:
                penalty += max(0, r - y) + max(0, y + r - h)
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return penalty
    
    def is_valid(centers, radii, w, h, tol=1e-6):
        """Check if configuration is valid."""
        if w + h != 2.0:
            return False
        for i in range(N):
            r = radii[i]
            if r < -tol:
                return False
            x, y = centers[i]
            if x - r < -tol or x + r > w + tol:
                return False
            if y - r < -tol or y + r > h + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def sum_radii(centers, radii, w, h):
        return np.sum(radii)
    
    # Simulated annealing
    centers, radii, w, h = initial_config()
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_w = w
    best_h = h
    best_score = sum_radii(best_centers, best_radii, best_w, best_h)
    best_penalty = energy(best_centers, best_radii, best_w, best_h)
    
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.9995
    
    np.random.seed(42)
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        
        # Random perturbation
        new_centers = centers.copy()
        new_radii = radii.copy()
        new_w = w
        new_h = h
        
        # Perturb a random circle
        idx = np.random.randint(N)
        new_centers[idx, 0] += np.random.normal(0, 0.1)
        new_centers[idx, 1] += np.random.normal(0, 0.1)
        new_radii[idx] += np.random.normal(0, 0.02)
        new_radii[idx] = max(0.01, new_radii[idx])
        
        # Sometimes perturb aspect ratio
        if np.random.random() < 0.1:
            delta = np.random.normal(0, 0.05)
            new_w = max(0.3, min(1.7, w + delta))
            new_h = 2.0 - new_w
        
        # Scale radii when aspect ratio changes
        if new_w != w:
            scale = new_w / w
            new_radii *= scale
            new_centers[:, 0] *= scale
        
        new_penalty = energy(new_centers, new_radii, new_w, new_h)
        old_penalty = energy(centers, radii, w, h)
        
        delta_penalty = new_penalty - old_penalty
        if delta_penalty < 0 or np.random.random() < np.exp(-delta_penalty / T):
            centers, radii, w, h = new_centers, new_radii, new_w, new_h
            
            # Track best
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_centers = centers.copy()
                best_radii = radii.copy()
                best_w = w
                best_h = h
                best_score = sum_radii(best_centers, best_radii, best_w, best_h)
        
        T *= cooling
        
        # Every 100 steps, try to improve by scaling radii up
        if step % 100 == 0 and time.time() < deadline:
            # Try to increase all radii slightly while maintaining validity
            for _ in range(5):
                trial_radii = best_radii * 1.01
                trial_penalty = energy(best_centers, trial_radii, best_w, best_h)
                if trial_penalty < 1e-4:
                    best_radii = trial_radii
                    best_score = sum_radii(best_centers, best_radii, best_w, best_h)
                else:
                    break
    
    # Final validation and cleanup
    if not is_valid(best_centers, best_radii, best_w, best_h):
        # Fall back to simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * w / side
            y = (row + 0.5) * h / side
            r = 0.5 * w / side - 0.01
            cells.append([x, y, r])
        return np.array(cells)
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_centers[i, 0]
        result[i, 1] = best_centers[i, 1]
        result[i, 2] = best_radii[i]
    
    return result
