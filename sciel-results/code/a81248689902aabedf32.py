"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses a physics-based relaxation approach with iterative radius growth.
    """
    n = N
    
    # 1. Initialization: Grid packing
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n: break
            x[idx] = (i + 0.5) / side
            y[idx] = (j + 0.5) / side
            # Initial radius slightly less than half cell width
            r[idx] = 0.45 / side
            idx += 1
            
    # If n is not a perfect square, the last few might be crowded, 
    # but 0.45/side is small enough to be valid initially.
    
    best_value = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())

    # 2. Relaxation Function
    def relax_positions(x_curr, y_curr, r_curr, iterations=200, step_size=0.01):
        """
        Minimize overlap and boundary violations for fixed radii.
        Returns optimized x, y.
        """
        x = x_curr.copy()
        y = y_curr.copy()
        
        # Simple gradient descent on penalty
        # Penalty: sum of overlaps^2 + sum of boundary violations^2
        for _ in range(iterations):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Boundary forces
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] -= (x[i] - r[i])
                if x[i] + r[i] > 1:
                    fx[i] -= (x[i] + r[i] - 1)
                if y[i] - r[i] < 0:
                    fy[i] -= (y[i] - r[i])
                if y[i] + r[i] > 1:
                    fy[i] -= (y[i] + r[i] - 1)
            
            # Overlap forces
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        # Push apart
                        if dist > 1e-8:
                            force = (min_dist - dist)
                            fx[i] += force * (dx / dist)
                            fy[i] += force * (dy / dist)
                            fx[j] -= force * (dx / dist)
                            fy[j] -= force * (dy / dist)
                        else:
                            # Random kick if exactly overlapping
                            fx[i] += np.random.uniform(-1, 1)
                            fy[i] += np.random.uniform(-1, 1)
                            fx[j] -= np.random.uniform(-1, 1)
                            fy[j] -= np.random.uniform(-1, 1)
            
            # Update positions
            x += step_size * fx
            y += step_size * fy
            
            # Hard projection to unit square
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
        return x, y

    def is_valid(x, y, r, tol=1e-8):
        # Check boundaries
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol): return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol): return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    # 3. Growth Loop
    # We try to increase radii.
    # Strategy: Multiply all radii by a factor, relax, check validity.
    # If valid, accept. If not, revert and try smaller factor or perturb.
    
    np.random.seed(42)
    
    max_grow_steps = 500
    current_factor = 1.01
    
    for step in range(max_grow_steps):
        # Try to grow
        r_new = r * current_factor
        
        # Relax positions for new radii
        x_relaxed, y_relaxed = relax_positions(x, y, r_new, iterations=100, step_size=0.02)
        
        # Check validity
        if is_valid(x_relaxed, y_relaxed, r_new):
            # Accept
            x, y, r = x_relaxed, y_relaxed, r_new
            best_value = np.sum(r)
            best_state = (x.copy(), y.copy(), r.copy())
            
            # Increase growth factor slightly if we've been successful?
            # Or keep it constant. Constant is safer.
            # Let's try to increase the step size of relaxation if it converges fast?
            # For simplicity, keep step_size fixed.
            
        else:
            # Failed to grow.
            # Try perturbing current positions and relaxing again with same radii?
            # Or reduce factor?
            # Let's try a few random perturbations on the current valid state
            # and see if we can find a better packing for the SAME radii,
            # which might allow a larger growth step next time.
            
            improved = False
            for _ in range(10):
                x_pert = x + np.random.uniform(-0.01, 0.01, n)
                y_pert = y + np.random.uniform(-0.01, 0.01, n)
                x_pert = np.clip(x_pert, r, 1-r)
                y_pert = np.clip(y_pert, r, 1-r)
                
                x_rel, y_rel = relax_positions(x_pert, y_pert, r, iterations=100, step_size=0.02)
                
                if is_valid(x_rel, y_rel, r):
                    # Check if this packing is "tighter" (lower potential energy)
                    # Lower energy might allow larger growth.
                    # Simple metric: sum of distances between pairs (closer is better for packing? No, we want to maximize min distance relative to radii)
                    # Actually, if it's valid, it's fine.
                    # We can accept it if it allows a larger growth step later.
                    # For now, just accept if it's valid.
                    x, y = x_rel, y_rel
                    improved = True
                    break
            
            if not improved:
                # If we can't even maintain current radii with perturbations,
                # maybe we are stuck. Try a smaller growth factor next time?
                # Or just break if we've tried many times.
                pass

    # Final polish: Try to optimize radii individually?
    # The growth method increases all radii proportionally.
    # We can try to increase specific radii that have more space.
    
    # Simple local search on radii
    for _ in range(20):
        for i in range(n):
            r_try = r.copy()
            r_try[i] *= 1.02
            x_rel, y_rel = relax_positions(x, y, r_try, iterations=50, step_size=0.01)
            if is_valid(x_rel, y_rel, r_try):
                x, y, r = x_rel, y_rel, r_try
                best_value = np.sum(r)
                best_state = (x.copy(), y.copy(), r.copy())

    return np.column_stack([best_state[0], best_state[1], best_state[2]])
# EVOLVE-BLOCK-END
