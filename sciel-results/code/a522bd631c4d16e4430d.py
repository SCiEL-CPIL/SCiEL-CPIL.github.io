"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    """Pack N circles in unit square, maximizing sum of radii."""
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def initial_grid(n):
        """Create initial grid placement."""
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    def check_validity(circles, tol=1e-8):
        """Check if circles are valid (inside square, non-overlapping)."""
        n = len(circles)
        # Check bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True
    
    def compute_score(circles):
        """Compute the objective: sum of radii."""
        return np.sum(circles[:, 2])
    
    def project_constraints(circles, max_iter=100, tol=1e-10):
        """
        Project circles to satisfy constraints using iterative projection.
        Returns updated circles array.
        """
        n = len(circles)
        circles = circles.copy()
        
        for iteration in range(max_iter):
            moved = False
            
            # Project to bounds
            for i in range(n):
                x, y, r = circles[i]
                # Clamp x
                if x - r < 0:
                    x = r
                    moved = True
                elif x + r > 1:
                    x = 1 - r
                    moved = True
                # Clamp y
                if y - r < 0:
                    y = r
                    moved = True
                elif y + r > 1:
                    y = 1 - r
                    moved = True
                circles[i] = [x, y, r]
            
            # Project to non-overlap
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dx = xj - xi
                    dy = yj - yi
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = ri + rj
                    
                    if dist < min_dist and dist > 1e-15:
                        # Push apart
                        overlap = min_dist - dist
                        # Move each circle half the overlap along the line connecting them
                        push_x = (dx / dist) * overlap * 0.5
                        push_y = (dy / dist) * overlap * 0.5
                        
                        circles[i, 0] -= push_x
                        circles[i, 1] -= push_y
                        circles[j, 0] += push_x
                        circles[j, 1] += push_y
                        moved = True
                    elif dist <= 1e-15:
                        # Circles at same point, push in arbitrary direction
                        circles[i, 0] -= min_dist * 0.5
                        circles[j, 0] += min_dist * 0.5
                        moved = True
            
            if not moved:
                break
        
        # Final clamp to ensure we're inside the square
        for i in range(n):
            x, y, r = circles[i]
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            circles[i] = [x, y, r]
        
        return circles
    
    def grow_radii(circles, growth_factor=1.01, max_grow_iter=200):
        """
        Iteratively grow radii while maintaining feasibility.
        Returns best valid configuration found.
        """
        n = len(circles)
        best_circles = circles.copy()
        best_score = compute_score(circles)
        current_circles = circles.copy()
        
        for grow_iter in range(max_grow_iter):
            if time.time() > deadline - 5:
                break
            
            # Try to grow all radii by growth_factor
            new_radii = current_circles[:, 2] * growth_factor
            
            # Check if any radius would exceed 0.5 (max possible in unit square)
            if np.any(new_radii > 0.5):
                break
            
            # Tentatively update radii
            candidate = current_circles.copy()
            candidate[:, 2] = new_radii
            
            # Project to satisfy constraints
            candidate = project_constraints(candidate, max_iter=50)
            
            # Check validity
            if check_validity(candidate, tol=1e-8):
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_circles = candidate.copy()
                current_circles = candidate.copy()
            else:
                # If invalid, try smaller growth
                break
        
        return best_circles
    
    def local_optimization(circles, max_iter=500, lr=0.01):
        """
        Simple gradient-free local optimization using random perturbations.
        """
        n = len(circles)
        best_circles = circles.copy()
        best_score = compute_score(circles)
        current_circles = circles.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            candidate = current_circles.copy()
            noise_scale = 0.01
            candidate[:, :2] += np.random.randn(n, 2) * noise_scale
            candidate[:, 2] *= (1 + np.random.randn(n) * 0.005)
            candidate[:, 2] = np.clip(candidate[:, 2], 1e-6, 0.5)
            
            # Project to constraints
            candidate = project_constraints(candidate, max_iter=30)
            
            # Check validity
            if check_validity(candidate, tol=1e-8):
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_circles = candidate.copy()
                    current_circles = candidate.copy()
        
        return best_circles
    
    def multi_start(n_starts=5):
        """
        Try multiple random starting configurations and return the best.
        """
        best_overall = None
        best_overall_score = -np.inf
        
        for start in range(n_starts):
            if time.time() > deadline - 10:
                break
            
            if start == 0:
                # Start with grid
                circles = initial_grid(N)
            else:
                # Random start
                circles = np.zeros((N, 3))
                circles[:, :2] = np.random.uniform(0.1, 0.9, (N, 2))
                circles[:, 2] = np.random.uniform(0.05, 0.15, N)
                circles = project_constraints(circles, max_iter=50)
            
            # Grow radii
            circles = grow_radii(circles, growth_factor=1.02, max_grow_iter=100)
            
            # Local optimization
            circles = local_optimization(circles, max_iter=200)
            
            score = compute_score(circles)
            if score > best_overall_score:
                best_overall_score = score
                best_overall = circles.copy()
        
        return best_overall
    
    # Main execution
    best_result = initial_grid(N)
    best_score = compute_score(best_result)
    
    # Multi-start optimization
    result = multi_start(n_starts=5)
    if result is not None:
        score = compute_score(result)
        if score > best_score:
            best_result = result
            best_score = score
    
    # Final verification
    if not check_validity(best_result, tol=1e-8):
        # Fall back to grid if somehow invalid
        best_result = initial_grid(N)
    
    return best_result
# EVOLVE-BLOCK-END
