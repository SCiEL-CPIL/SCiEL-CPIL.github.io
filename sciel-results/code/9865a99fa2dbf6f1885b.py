"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def check_time():
        return (time.time() - start_time) > deadline
    
    # Rectangle: W + H = 2. Try square first (W=H=1)
    W = 1.0
    H = 1.0
    
    def hexagonal_initial(n):
        """Generate initial hexagonal-like packing positions and radii"""
        # For 21 circles in a 1x1 square, use a grid with hexagonal offset
        # Try different arrangements
        positions = []
        radii = []
        
        # Use a 5x5 grid subset or similar
        # Let's try rows of circles
        # 21 = 3+4+5+4+3 or 4+5+4+5+3 etc.
        # Try 5 rows: 4,5,4,5,3 = 21? No, 4+5+4+5+3=21. Yes!
        row_counts = [4, 5, 4, 5, 3]
        n_rows = len(row_counts)
        
        # Vertical spacing
        row_heights = np.linspace(0.1, 0.9, n_rows)
        
        for i, count in enumerate(row_counts):
            y = row_heights[i]
            if count > 0:
                xs = np.linspace(0.1, 0.9, count)
                for x in xs:
                    positions.append([x, y])
                    radii.append(0.08)
        
        return np.array(positions), np.array(radii)
    
    def compute_score(positions, radii, W, H):
        """Compute objective: sum of radii minus penalties"""
        n = len(radii)
        score = np.sum(radii)
        
        # Penalty for circles outside rectangle
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Distance to boundaries
            dx = min(x, W - x)
            dy = min(y, H - y)
            if dx < r:
                score -= 100 * (r - dx)
            if dy < r:
                score -= 100 * (r - dy)
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    score -= 100 * (min_dist - dist)
        
        return score
    
    def is_valid(positions, radii, W, H, tol=1e-6):
        """Check if configuration is valid (no overlaps, all inside)"""
        n = len(radii)
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            if x < -tol or x > W + tol or y < -tol or y > H + tol:
                return False
            if r < -tol:
                return False
            if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def optimize(positions, radii, W, H, max_iters=5000):
        """Local optimization using gradient-free approach"""
        best_score = compute_score(positions, radii, W, H)
        best_pos = positions.copy()
        best_rad = radii.copy()
        
        step_size_pos = 0.02
        step_size_rad = 0.005
        temp = 0.01
        
        for it in range(max_iters):
            if check_time():
                break
            
            # Random perturbation
            new_pos = positions + np.random.randn(*positions.shape) * step_size_pos
            new_rad = radii + np.random.randn(len(radii)) * step_size_rad
            new_rad = np.maximum(new_rad, 0.01)
            
            # Clamp positions to stay roughly in bounds
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, W)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, H)
            
            new_score = compute_score(new_pos, new_rad, W, H)
            
            if new_score > best_score:
                positions = new_pos
                radii = new_rad
                best_score = new_score
                best_pos = new_pos.copy()
                best_rad = new_rad.copy()
                # Slowly decrease step sizes
                step_size_pos *= 0.995
                step_size_rad *= 0.995
            else:
                # Accept with some probability (simulated annealing style)
                if np.random.random() < np.exp((new_score - best_score) / max(temp, 1e-10)):
                    positions = new_pos
                    radii = new_rad
                temp *= 0.999
        
        return best_pos, best_rad, best_score
    
    # Try multiple initial configurations
    best_overall_score = -np.inf
    best_overall_pos = None
    best_overall_rad = None
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if check_time():
            break
        
        # Different initial patterns
        if restart == 0:
            pos, rad = hexagonal_initial(N)
        else:
            # Random perturbation of hexagonal pattern
            pos, rad = hexagonal_initial(N)
            pos += np.random.randn(*pos.shape) * 0.05
            pos[:, 0] = np.clip(pos[:, 0], 0.05, W - 0.05)
            pos[:, 1] = np.clip(pos[:, 1], 0.05, H - 0.05)
            rad = np.maximum(rad + np.random.randn(N) * 0.02, 0.02)
        
        # Optimize
        pos_opt, rad_opt, score = optimize(pos, rad, W, H, max_iters=3000)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pos = pos_opt
            best_overall_rad = rad_opt
    
    # Final result
    if best_overall_pos is None:
        # Fallback: simple grid
        pos, rad = hexagonal_initial(N)
        return np.column_stack([pos, rad])
    
    result = np.column_stack([best_overall_pos, best_overall_rad])
    return result

# EVOLVE-BLOCK-END
