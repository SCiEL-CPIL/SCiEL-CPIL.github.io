"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Coordinate Descent with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        
        # Initialize at a random point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls_made = 1
        
        # Adaptive step sizes for each coordinate
        steps = np.ones(self.dim) * 1.0
        # Track success rates for each coordinate (simple exponential moving average)
        success_rates = np.ones(self.dim) * 0.5
        
        # Phase 1: Coarse search with coordinate descent
        # We interleave coordinate descent with occasional random jumps to escape local minima
        # The "derandomized" aspect is that we cycle through coordinates in a fixed order 
        # but adapt the step size based on recent performance.
        
        # To handle the rotated landscape, we can't just use standard axes.
        # However, the directive says "aligning search directions with the function's principal axes".
        # Since we don't know the principal axes, we can use a random rotation matrix 
        # and perform coordinate descent in that rotated space.
        # Let's pick a random rotation matrix R.
        # x_orig = R @ x_rot
        # We optimize in x_rot space, but evaluate in x_orig space.
        
        # Generate a random orthogonal matrix (Haar measure)
        # Using QR decomposition of a random Gaussian matrix
        A = rng.standard_normal((self.dim, self.dim))
        Q, R_mat = np.linalg.qr(A)
        # Ensure Q is a proper rotation (det=1)
        d = np.diag(R_mat)
        ph = np.sign(d)
        Q = Q * ph
        
        # Convert initial x to rotated space
        x_rot = np.linalg.solve(Q, x)  # Q.T @ x, since Q is orthogonal Q.T = Q^{-1}
        
        # We will perform coordinate descent in the rotated space
        # Step sizes in rotated space
        steps_rot = np.ones(self.dim) * 2.0
        success_rates_rot = np.ones(self.dim) * 0.5
        
        # Cycle through coordinates in rotated space
        # To avoid getting stuck, we periodically re-rotate or do a random jump
        # Let's stick to one rotation for simplicity first, but adapt steps.
        
        # We'll use a "cycling" approach: iterate through dimensions 0..dim-1 repeatedly.
        # For each dimension, try a step in + and - direction.
        
        # To make it more effective, we can use a "golden section" or simple line search 
        # along each coordinate, but that's expensive.
        # Let's stick to a fixed number of trials per coordinate per cycle, 
        # with adaptive step size.
        
        # Strategy:
        # 1. Start with a random rotation.
        # 2. Perform coordinate descent in the rotated space.
        # 3. If no improvement for a while, try a different rotation or a random jump.
        
        # Let's implement a simpler version:
        # - Maintain current best x and fx.
        # - Use a list of candidate directions (initially the columns of Q).
        # - For each direction, perform a 1D search (trying a few step sizes).
        # - Adapt step size based on success.
        
        # Actually, the directive says "Derandomized Coordinate Descent with adaptive step sizes, 
        # where each coordinate's step is dynamically scaled based on the success rate of recent moves along that axis".
        # This implies we are working in some coordinate system. Let's use the rotated system.
        
        # We need to be careful with the budget. 500 calls.
        # Let's reserve some calls for a final polish.
        
        # Loop until budget or deadline
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Cycle through coordinates in rotated space
            for i in range(self.dim):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try moving in the i-th coordinate of the rotated space
                # Try both + and - directions
                for sign in [1, -1]:
                    if calls_made >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Calculate new point in rotated space
                    x_rot_new = x_rot.copy()
                    x_rot_new[i] += sign * steps_rot[i]
                    
                    # Convert back to original space
                    x_new = Q @ x_rot_new
                    # Clip to box
                    x_new = np.clip(x_new, -5, 5)
                    
                    # Evaluate
                    fx_new = func(x_new)
                    calls_made += 1
                    
                    # Update success rate
                    if fx_new < fx:
                        # Success
                        success_rates_rot[i] = 0.8 * success_rates_rot[i] + 0.2 * 1.0
                        # Increase step size
                        steps_rot[i] *= 1.1
                        # Update best
                        x_rot = x_rot_new
                        x = x_new
                        fx = fx_new
                    else:
                        # Failure
                        success_rates_rot[i] = 0.8 * success_rates_rot[i] + 0.2 * 0.0
                        # Decrease step size
                        steps_rot[i] *= 0.9
                    
                    # If we improved, we can stop trying the other direction for this coordinate
                    if fx_new < fx:
                        break
        
        # Final polish: use a smaller step size and more iterations if time permits
        # But we are already using the budget.
        
        return x, fx
# EVOLVE-BLOCK-END
