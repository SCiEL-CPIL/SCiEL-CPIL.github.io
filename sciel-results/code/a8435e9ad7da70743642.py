"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    # We'll use a local optimization approach starting from a good initial configuration
    # The objective is to maximize the sum of radii, which is equivalent to maximizing
    # the minimum distance between circles and boundaries, scaled appropriately.
    
    # For circle packing, a common approach is to maximize the minimum "clearance"
    # We can formulate this as: maximize t subject to:
    #   dist(i,j) >= r_i + r_j for all i != j
    #   x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    #
    # A simpler approach: assume all circles have the same radius r, and maximize r
    # subject to the constraints. This is a convex feasibility problem for a given r.
    
    # Let's try a different approach: use gradient-based optimization to maximize
    # the sum of radii directly, with constraints that circles don't overlap.
    
    # Actually, let's use a well-known heuristic: start from a hexagonal packing
    # arrangement and use local optimization to improve it.
    
    # Initial configuration: hexagonal packing
    def hexagonal_init(n):
        """Generate a hexagonal packing initial configuration."""
        # For n circles, we can use a hexagonal grid pattern
        # The optimal hexagonal packing for n circles in a unit square
        # involves arranging circles in a hexagonal lattice
        
        # Let's try a simple approach: place circles in a hexagonal pattern
        # and then optimize
        
        # Start with a grid-like hexagonal arrangement
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        rows = int(np.ceil(n / cols))
        
        # Hexagonal spacing
        # In a hexagonal packing, the distance between centers is 2r
        # The vertical spacing between rows is sqrt(3) * r
        # The horizontal spacing is 2r
        
        # Let's estimate r from the bounding box
        # Width: cols * 2r - r (offset) = (2*cols - 1) * r approximately
        # Height: (rows - 1) * sqrt(3) * r + 2r
        
        # For now, let's just use a reasonable initial radius
        # and let the optimizer figure out the rest
        
        centers = []
        radii = []
        
        # Try to fit as many circles as possible in a hexagonal pattern
        r_est = 0.08  # initial estimate
        
        for row in range(rows):
            for col in range(cols):
                if len(centers) >= n:
                    break
                # Hexagonal offset
                x = (col + 0.5) * 2 * r_est + (row % 2) * r_est
                y = (row + 0.5) * np.sqrt(3) * r_est
                centers.append((x, y))
                radii.append(r_est)
        
        # If we don't have enough, add more in a grid
        while len(centers) < n:
            i = len(centers)
            # Place in a grid
            col = i % cols
            row = i // cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            centers.append((x, y))
            radii.append(0.05)
        
        return np.array(centers[:n]), np.array(radii[:n])
    
    # Let's try a different, more robust approach:
    # Use a grid initialization and then do local optimization
    
    def grid_init(n):
        """Generate a grid initial configuration."""
        side = int(np.ceil(np.sqrt(n)))
        centers = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                centers.append((x, y))
                radii.append(r)
        return np.array(centers), np.array(radii)
    
    def random_init(n):
        """Generate a random initial configuration."""
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        radii = np.full(n, 0.05)
        return centers, radii
    
    # Objective function: maximize sum of radii
    # We'll use a penalty method for the constraints
    
    def objective_and_constraints(params):
        """
        params: 2n x 1 array [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        Returns: -sum(radii) + penalty for violations
        """
        n_local = len(params) // 3
        centers = params[:2*n_local].reshape(n_local, 2)
        radii = params[2*n_local:]
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.001)
        
        # Penalty for boundary violations
        penalty = 0.0
        for i in range(n_local):
            x, y = centers[i]
            r = radii[i]
            # Distance to boundaries
            d_left = x - r
            d_right = 1 - x - r
            d_bottom = y - r
            d_top = 1 - y - r
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d < 0:
                penalty += min_d**2 * 100
        
        # Penalty for overlap
        for i in range(n_local):
            for j in range(i+1, n_local):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2 * 100
        
        return -np.sum(radii) + penalty
    
    # Let's use a different strategy: fix the radii to be equal and optimize
    # the positions to maximize the common radius
    
    def optimize_equal_radius():
        """Optimize positions for equal radii to maximize the common radius."""
        n_local = N
        # We'll use a bisection on the radius r
        # For a given r, check if we can place n circles of radius r
        # without overlap and within the unit square
        
        def feasible(r, centers):
            """Check if circles of radius r at given centers are feasible."""
            # Check boundaries
            for i in range(n_local):
                x, y = centers[i]
                if x < r or x > 1 - r or y < r or y > 1 - r:
                    return False
            # Check overlaps
            for i in range(n_local):
                for j in range(i+1, n_local):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 2 * r - 1e-9:
                        return False
            return True
        
        # We need to find the maximum r such that there exists a feasible configuration
        # This is hard, so let's use a heuristic: start from a good configuration
        # and try to increase r
        
        # Start with a grid configuration
        side = int(np.ceil(np.sqrt(n_local)))
        centers = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                           for i in range(side) for j in range(side)][:n_local])
        
        # Initial radius
        r = 0.5 / side - 0.01
        
        # Try to increase r by moving centers
        # Use local optimization to maximize the minimum distance
        
        def min_distance(centers_arr):
            """Compute the minimum distance between any two circles and to boundaries."""
            min_d = np.inf
            for i in range(n_local):
                x, y = centers_arr[i]
                # Distance to boundaries
                min_d = min(min_d, x, 1 - x, y, 1 - y)
            for i in range(n_local):
                for j in range(i+1, n_local):
                    dx = centers_arr[i, 0] - centers_arr[j, 0]
                    dy = centers_arr[i, 1] - centers_arr[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_d = min(min_d, dist)
            return min_d / 2  # radius = half the min distance
        
        # Maximize min_distance
        def neg_min_distance(params):
            centers_arr = params.reshape(n_local, 2)
            return -min_distance(centers_arr)
        
        # Optimize
        best_centers = centers.copy()
        best_r = r
        
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            
            # Random perturbation
            if restart == 0:
                centers_arr = centers.copy()
            else:
                centers_arr = centers + np.random.randn(n_local, 2) * 0.05
                centers_arr = np.clip(centers_arr, 0.05, 0.95)
            
            result = minimize(neg_min_distance, centers_arr.flatten(), 
                            method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < -best_r:
                best_r = -result.fun
                best_centers = result.x.reshape(n_local, 2)
        
        return best_centers, best_r
    
    # Let's try the optimization approach
    centers, r = optimize_equal_radius()
    
    # Check if we're within time limit
    if time.time() > deadline - 10:
        # Return what we have
        result = np.zeros((N, 3))
        result[:, :2] = centers
        result[:, 2] = r
        return result
    
    # Now try to improve by allowing different radii
    # Use a local search approach
    
    def local_search(centers_arr, radii_arr, max_iters=1000):
        """Local search to improve the packing."""
        n_local = len(radii_arr)
        best_score = np.sum(radii_arr)
        best_centers = centers_arr.copy()
        best_radii = radii_arr.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try to move a random circle
            i = np.random.randint(n_local)
            dx = np.random.randn() * 0.02
            dy = np.random.randn() * 0.02
            new_x = best_centers[i, 0] + dx
            new_y = best_centers[i, 1] + dy
            
            # Check if the new position is valid
            valid = True
            if new_x < best_radii[i] or new_x > 1 - best_radii[i] or \
               new_y < best_radii[i] or new_y > 1 - best_radii[i]:
                valid = False
            else:
                for j in range(n_local):
                    if j == i:
                        continue
                    dx2 = new_x - best_centers[j, 0]
                    dy2 = new_y - best_centers[j, 1]
                    dist = np.sqrt(dx2**2 + dy2**2)
                    if dist < best_radii[i] + best_radii[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                # Try to increase the radius of circle i
                new_r = best_radii[i]
                # Max possible radius based on boundaries
                max_r_boundary = min(new_x, 1 - new_x, new_y, 1 - new_y)
                # Max possible radius based on other circles
                max_r_overlap = np.inf
                for j in range(n_local):
                    if j == i:
                        continue
                    dx2 = new_x - best_centers[j, 0]
                    dy2 = new_y - best_centers[j, 1]
                    dist = np.sqrt(dx2**2 + dy2**2)
                    max_r_overlap = min(max_r_overlap, (dist - best_radii[j]) / 2)
                
                new_r = min(new_r + 0.005, max_r_boundary, max_r_overlap)
                new_r = max(new_r, 0.01)
                
                # Check if this improves the score
                new_score = np.sum(best_radii) - best_radii[i] + new_r
                if new_score > best_score:
                    best_centers[i] = [new_x, new_y]
                    best_radii[i] = new_r
                    best_score = new_score
        
        return best_centers, best_radii, best_score
    
    # Run local search
    centers_arr = centers.copy()
    radii_arr = np.full(N, r)
    centers_arr, radii_arr, score = local_search(centers_arr, radii_arr)
    
    # Final check
    if time.time() > deadline - 5:
        pass
    
    result = np.zeros((N, 3))
    result[:, :2] = centers_arr
    result[:, 2] = radii_arr
    
    # Final validation: ensure all circles are within bounds and non-overlapping
    for i in range(N):
        x, y, r_i = result[i]
        # Clamp to valid range
        r_i = max(0.001, min(r_i, min(x, 1-x, y, 1-y)))
        result[i, 2] = r_i
    
    # Ensure non-overlapping by reducing radii if needed
    for i in range(N):
        for j in range(i+1, N):
            dx = result[i, 0] - result[j, 0]
            dy = result[i, 1] - result[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            max_r = (dist - 1e-6) / 2
            if result[i, 2] + result[j, 2] > dist:
                # Reduce the larger radius
                if result[i, 2] > result[j, 2]:
                    result[i, 2] = max(0.001, max_r - result[j, 2])
                else:
                    result[j, 2] = max(0.001, max_r - result[i, 2])
    
    return result
# EVOLVE-BLOCK-END
