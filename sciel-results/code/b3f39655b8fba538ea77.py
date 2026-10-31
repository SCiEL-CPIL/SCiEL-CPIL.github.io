"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    best_result = None
    best_score = -np.inf
    
    def compute_score(circles):
        """Compute the sum of radii, or -inf if invalid."""
        if len(circles) != N:
            return -np.inf
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        # Check boundaries
        if np.any(r < 0):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def initial_grid():
        """Create a grid of equal circles."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells[:N])
    
    def jitter_and_expand(circles, max_iters=50):
        """Apply jitter and expand heuristic."""
        circles = circles.copy()
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Random angular perturbation
            angles = np.random.uniform(0, 2*np.pi, N)
            magnitudes = np.random.uniform(0, 0.01, N)
            dx = magnitudes * np.cos(angles)
            dy = magnitudes * np.sin(angles)
            
            new_x = x + dx
            new_y = y + dy
            
            # Check if move improves validity or keeps valid
            new_circles = np.column_stack([new_x, new_y, r])
            
            # Try to expand radii using a simple iterative approach
            # For each circle, try to increase radius while maintaining constraints
            improved = False
            for i in range(N):
                # Try increasing radius of circle i
                current_r = new_circles[i, 2]
                max_r = min(new_circles[i, 0], 1 - new_circles[i, 0],
                           new_circles[i, 1], 1 - new_circles[i, 1])
                
                # Check pairwise constraints
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((new_circles[i, 0] - new_circles[j, 0])**2 + 
                                   (new_circles[i, 1] - new_circles[j, 1])**2)
                    max_r = min(max_r, dist - new_circles[j, 2])
                
                # If we can increase radius, do so
                if max_r > current_r + 1e-6:
                    new_circles[i, 2] = max_r
                    improved = True
            
            # Accept if improved or if score increased
            new_score = compute_score(new_circles)
            if new_score > compute_score(circles):
                circles = new_circles
                x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
            
            if not improved:
                break
        
        return circles
    
    def local_search(circles, n_restarts=5):
        """Perform local search with multiple restarts."""
        best_circles = circles.copy()
        best_score = compute_score(circles)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Jitter the current solution
            jittered = circles.copy()
            jittered[:, 0] += np.random.uniform(-0.05, 0.05, N)
            jittered[:, 1] += np.random.uniform(-0.05, 0.05, N)
            
            # Clip to valid range
            jittered[:, 0] = np.clip(jittered[:, 0], 0.01, 0.99)
            jittered[:, 1] = np.clip(jittered[:, 1], 0.01, 0.99)
            
            # Apply jitter and expand
            expanded = jitter_and_expand(jittered)
            score = compute_score(expanded)
            
            if score > best_score:
                best_score = score
                best_circles = expanded
        
        return best_circles
    
    # Initial solution
    circles = initial_grid()
    score = compute_score(circles)
    
    if score > best_score:
        best_score = score
        best_result = circles.copy()
    
    # Perform local search
    if time.time() < deadline - 15.0:
        circles = local_search(circles, n_restarts=3)
        score = compute_score(circles)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
    
    # Final polish: try to maximize radii
    if time.time() < deadline - 5.0 and best_result is not None:
        # Simple radius maximization
        for _ in range(10):
            improved = False
            for i in range(N):
                current_r = best_result[i, 2]
                max_r = min(best_result[i, 0], 1 - best_result[i, 0],
                           best_result[i, 1], 1 - best_result[i, 1])
                
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((best_result[i, 0] - best_result[j, 0])**2 + 
                                   (best_result[i, 1] - best_result[j, 1])**2)
                    max_r = min(max_r, dist - best_result[j, 2])
                
                if max_r > current_r + 1e-6:
                    best_result[i, 2] = max_r
                    improved = True
            
            if not improved:
                break
    
    if best_result is None:
        best_result = initial_grid()
    
    return best_result
# EVOLVE-BLOCK-END
