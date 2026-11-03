"""Rewritten program for circle packing in unit square; stronger algorithm."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses a local search / optimization approach with initial grid placement.
    """
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    
    x0 = np.array(cells, dtype=float)
    n = len(x0)
    
    # Variables: x, y, r for each circle
    # Constraints:
    # 1. x_i >= r_i
    # 2. x_i <= 1 - r_i
    # 3. y_i >= r_i
    # 4. y_i <= 1 - r_i
    # 5. r_i > 0
    # 6. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    def objective(vars):
        # Maximize sum of radii -> minimize negative sum
        radii = vars[2::3]
        return -np.sum(radii)
    
    # Use SLSQP with constraints
    # This can be slow for 26 circles, so let's try a different approach:
    # Use a simulated annealing or local search approach
    
    # Let's use a more practical approach: iterative local search
    # Start with grid, then try to grow circles and move them
    
    x, y, r = x0[:, 0].copy(), x0[:, 1].copy(), x0[:, 2].copy()
    
    def check_valid(x, y, r, eps=1e-10):
        """Check if placement is valid."""
        n = len(x)
        # Check boundaries
        if np.any(x < r - eps) or np.any(x > 1 - r + eps):
            return False
        if np.any(y < r - eps) or np.any(y > 1 - r + eps):
            return False
        if np.any(r < eps):
            return False
        # Check pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2 - eps:
                    return False
        return True
    
    # Local search: try to increase radii and adjust positions
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = np.sum(best_r)
    
    # Try multiple random restarts with different initial configurations
    np.random.seed(42)
    
    for restart in range(50):
        # Create initial configuration
        if restart == 0:
            # Grid
            x_curr, y_curr, r_curr = x.copy(), y.copy(), r.copy()
        else:
            # Random valid initial configuration
            x_curr = np.random.uniform(0.05, 0.95, n)
            y_curr = np.random.uniform(0.05, 0.95, n)
            r_curr = np.full(n, 0.02)
        
        # Local search iterations
        for it in range(2000):
            # Try to increase a random radius
            idx = np.random.randint(n)
            old_r = r_curr[idx]
            # Try to increase radius
            new_r = old_r * 1.01 + 0.001
            
            # Check if we can fit with increased radius
            valid = True
            # Check boundaries
            if x_curr[idx] < new_r or x_curr[idx] > 1 - new_r:
                valid = False
            if y_curr[idx] < new_r or y_curr[idx] > 1 - new_r:
                valid = False
            
            if valid:
                # Check pairwise
                for j in range(n):
                    if j == idx:
                        continue
                    dist_sq = (x_curr[idx] - x_curr[j])**2 + (y_curr[idx] - y_curr[j])**2
                    min_dist = new_r + r_curr[j]
                    if dist_sq < min_dist**2:
                        valid = False
                        break
            
            if valid:
                r_curr[idx] = new_r
            else:
                # Try to move the circle to make room
                # Nudge position
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                new_x = np.clip(x_curr[idx] + dx, new_r, 1 - new_r)
                new_y = np.clip(y_curr[idx] + dy, new_r, 1 - new_r)
                
                valid = True
                for j in range(n):
                    if j == idx:
                        continue
                    dist_sq = (new_x - x_curr[j])**2 + (new_y - y_curr[j])**2
                    min_dist = new_r + r_curr[j]
                    if dist_sq < min_dist**2:
                        valid = False
                        break
                
                if valid:
                    x_curr[idx] = new_x
                    y_curr[idx] = new_y
                    r_curr[idx] = new_r
        
        # Check final validity
        if check_valid(x_curr, y_curr, r_curr):
            score = np.sum(r_curr)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_curr.copy(), y_curr.copy(), r_curr.copy()
    
    return np.column_stack([best_x, best_y, best_r])
# EVOLVE-BLOCK-END
