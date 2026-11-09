import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    def unpack(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def compute_penalty(centers, sides, tol=1e-7):
        """Compute total overlap area + boundary violation penalty."""
        p = 0.0
        # Boundary checks
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                p += 1e6
                continue
            if cx - s/2 < -tol:
                p += (cx - s/2 + tol)**2 * 100
            if cx + s/2 > 1 + tol:
                p += (cx + s/2 - 1 - tol)**2 * 100
            if cy - s/2 < -tol:
                p += (cy - s/2 + tol)**2 * 100
            if cy + s/2 > 1 + tol:
                p += (cy + s/2 - 1 - tol)**2 * 100
        
        # Overlap checks
        for i in range(n):
            for j in range(i+1, n):
                xi1 = centers[i][0] - sides[i]/2
                xi2 = centers[i][0] + sides[i]/2
                yi1 = centers[i][1] - sides[i]/2
                yi2 = centers[i][1] + sides[i]/2
                xj1 = centers[j][0] - sides[j]/2
                xj2 = centers[j][0] + sides[j]/2
                yj1 = centers[j][1] - sides[j]/2
                yj2 = centers[j][1] + sides[j]/2
                
                dx = min(xi2, xj2) - max(xi1, xj1)
                dy = min(yi2, yj2) - max(yi1, yj1)
                
                if dx > 0 and dy > 0:
                    p += dx * dy
        
        return p

    def is_valid(centers, sides, tol=1e-7):
        """Check if the configuration is valid."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                return False
            if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
                return False
            if cx - s/2 < -tol:
                return False
            if cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol:
                return False
            if cy + s/2 > 1 + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                xi1 = centers[i][0] - sides[i]/2
                xi2 = centers[i][0] + sides[i]/2
                yi1 = centers[i][1] - sides[i]/2
                yi2 = centers[i][1] + sides[i]/2
                xj1 = centers[j][0] - sides[j]/2
                xj2 = centers[j][0] + sides[j]/2
                yj1 = centers[j][1] - sides[j]/2
                yj2 = centers[j][1] + sides[j]/2
                
                dx = min(xi2, xj2) - max(xi1, xj1)
                dy = min(yi2, yj2) - max(yi1, yj1)
                
                if dx > tol and dy > tol:
                    return False
        return True

    def objective(x):
        centers, sides = unpack(x)
        if np.any(sides < 1e-8):
            return 1e6
        return -np.sum(sides) + 1000.0 * compute_penalty(centers, sides)

    # Generate diverse initial layouts
    def generate_grid_layout(rows, cols):
        """Generate a grid layout with rows x cols squares."""
        if rows * cols < n:
            return None, None
        side = 1.0 / max(rows, cols)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            r = i // cols
            c = i % cols
            # Center in cell
            cx = (c + 0.5) * side
            cy = (r + 0.5) * side
            centers[i] = [cx, cy]
            sides[i] = side
        return centers, sides

    def generate_staggered_layout():
        """Generate a staggered layout for better packing."""
        # 2 rows of 5, staggered
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s = 0.2  # side length
        for i in range(n):
            row = i // 5
            col = i % 5
            if row == 0:
                cx = (col + 0.5) * s
                cy = 0.5 * s
            else:
                # Stagger: shift by half a cell
                cx = (col + 0.5) * s
                cy = 1.5 * s
            centers[i] = [cx, cy]
            sides[i] = s
        return centers, sides

    def generate_mixed_layout():
        """Generate a mixed layout with different sizes."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # 4 large squares in corners, 6 smaller in middle
        s_large = 0.3
        s_small = 0.2
        # Corners
        corners = [
            (s_large/2, s_large/2),
            (1 - s_large/2, s_large/2),
            (s_large/2, 1 - s_large/2),
            (1 - s_large/2, 1 - s_large/2)
        ]
        for i in range(4):
            centers[i] = corners[i]
            sides[i] = s_large
        # Middle 6
        mid_positions = [
            (0.5, 0.15), (0.5, 0.5), (0.5, 0.85),
            (0.15, 0.5), (0.5, 0.5), (0.85, 0.5)
        ]
        for i in range(6):
            idx = i + 4
            centers[idx] = mid_positions[i]
            sides[idx] = s_small
        return centers, sides

    def local_improve(centers, sides, max_iter=1000, tol=1e-9):
        """Local improvement using coordinate descent."""
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_score = np.sum(sides)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                # Try adjusting center x
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    new_cx = best_centers[i][0] + delta
                    new_centers = best_centers.copy()
                    new_centers[i][0] = new_cx
                    if is_valid(new_centers, best_sides):
                        score = np.sum(best_sides)
                        if score > best_score + 1e-10:
                            best_centers = new_centers
                            best_score = score
                            improved = True
                
                # Try adjusting center y
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    new_cy = best_centers[i][1] + delta
                    new_centers = best_centers.copy()
                    new_centers[i][1] = new_cy
                    if is_valid(new_centers, best_sides):
                        score = np.sum(best_sides)
                        if score > best_score + 1e-10:
                            best_centers = new_centers
                            best_score = score
                            improved = True
                
                # Try increasing side length
                for delta in [0.01, 0.005, 0.001, 0.0005]:
                    new_s = best_sides[i] + delta
                    if new_s > 1.0:
                        continue
                    new_sides = best_sides.copy()
                    new_sides[i] = new_s
                    if is_valid(best_centers, new_sides):
                        score = np.sum(new_sides)
                        if score > best_score + 1e-10:
                            best_sides = new_sides
                            best_score = score
                            improved = True
            
            if not improved:
                break
        
        return best_centers, best_sides, best_score

    # Strategy 1: Try various grid layouts
    best_score = 0
    best_centers = None
    best_sides = None
    
    layouts = [
        (2, 5), (5, 2), (3, 4), (4, 3), (1, 10), (10, 1)
    ]
    
    for rows, cols in layouts:
        centers, sides = generate_grid_layout(rows, cols)
        if centers is not None and is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Strategy 2: Staggered layouts
    centers, sides = generate_staggered_layout()
    if centers is not None and is_valid(centers, sides):
        score = np.sum(sides)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Strategy 3: Mixed layouts
    centers, sides = generate_mixed_layout()
    if centers is not None and is_valid(centers, sides):
        score = np.sum(sides)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Local improvement on best layout
    if best_centers is not None:
        best_centers, best_sides, best_score = local_improve(best_centers, best_sides)
    
    # If no valid layout found, use fallback
    if best_centers is None:
        k = 4
        side = 1.0 / k
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            r = i // k
            c = i % k
            centers[i] = [(c + 0.5) * side, (r + 0.5) * side]
            sides[i] = side
        best_centers = centers
        best_sides = sides
        best_score = np.sum(sides)
    
    # Run DE optimization if time permits
    if time.time() < deadline - 60:
        bounds = [(0, 1) for _ in range(2*n)] + [(1e-6, 1) for _ in range(n)]
        
        # Multiple DE runs with different seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline - 30:
                break
            
            try:
                result = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=200, 
                    popsize=15, 
                    tol=1e-6,
                    seed=seed,
                    polish=False
                )
                
                if result.fun < -best_score:
                    c, s = unpack(result.x)
                    if is_valid(c, s):
                        score = np.sum(s)
                        if score > best_score:
                            best_centers = c
                            best_sides = s
                            best_score = score
            except Exception:
                pass
    
    # Final local improvement
    if best_centers is not None:
        best_centers, best_sides, best_score = local_improve(best_centers, best_sides, max_iter=500)
    
    # Ensure output types
    return best_centers.astype(np.float64), best_sides.astype(np.float64)
