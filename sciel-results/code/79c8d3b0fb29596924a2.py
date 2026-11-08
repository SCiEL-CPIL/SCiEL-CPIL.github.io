import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_result = None
    
    # We will search over different aspect ratios W (where H = 2 - W)
    # W is the width. Perimeter = 2(W+H) = 4 => W+H=2.
    # Symmetric around W=1.0. Let's try W from 0.7 to 1.3.
    
    def run_sa(W_init, seed=42):
        """
        Runs Simulated Annealing for a fixed initial aspect ratio W_init.
        Returns (value, result_array)
        """
        np.random.seed(seed)
        
        H_init = 2.0 - W_init
        
        # Initial state: Grid placement
        # We need to place N circles in W x H.
        # Let's use a grid of size cols x rows.
        # Approximate cols = sqrt(N * W/H)
        cols = int(np.ceil(np.sqrt(N * W_init / H_init)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Fill grid
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) * (W_init / cols)
            y[i] = (r_idx + 0.5) * (H_init / rows)
            
        # Initial radius: small
        r[:] = 0.01
        
        # State variables
        W = W_init
        H = H_init
        
        # SA Parameters
        T = 0.1
        T_end = 1e-6
        alpha = 0.999  # Cooling rate
        max_iter = 50000
        
        # Precompute indices for pairs
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                pairs.append((i, j))
        
        def calc_energy(pos, rad, w, h):
            """
            Returns (sum_radii, penalty)
            Penalty is sum of squared violations.
            """
            sum_r = np.sum(rad)
            penalty = 0.0
            
            # Boundary constraints
            # x_i >= r_i, x_i <= w - r_i
            # y_i >= r_i, y_i <= h - r_i
            # violation_x_low = max(0, r_i - x_i)
            # violation_x_high = max(0, x_i + r_i - w)
            
            # Vectorized penalty calculation for speed
            # Low X
            viol_xl = np.maximum(0, rad - pos[:, 0])
            viol_xh = np.maximum(0, pos[:, 0] + rad - w)
            viol_yl = np.maximum(0, rad - pos[:, 1])
            viol_yh = np.maximum(0, pos[:, 1] + rad - h)
            
            penalty += np.sum(viol_xl**2) + np.sum(viol_xh**2) + np.sum(viol_yl**2) + np.sum(viol_yh**2)
            
            # Pairwise constraints
            # dist(i,j) >= r_i + r_j
            # violation = max(0, r_i + r_j - dist)
            
            # To speed up, we can just loop in Python for N=21, it's 210 pairs.
            # Or use numpy broadcasting, but 21x21 matrix is small.
            # Let's use a simple loop for clarity and sufficient speed, or vectorize.
            # Vectorized:
            # dx = pos[:, 0][:, None] - pos[:, 0][None, :]
            # dy = pos[:, 1][:, None] - pos[:, 1][None, :]
            # dist = np.sqrt(dx**2 + dy**2)
            # min_dist = rad[:, None] + rad[None, :]
            # viol = np.maximum(0, min_dist - dist)
            # penalty += np.sum(viol**2) # This sums all i!=j, so divide by 2 if needed, or just use it as is.
            
            # Let's use the vectorized version for speed
            dx = pos[:, 0][:, None] - pos[:, 0][None, :]
            dy = pos[:, 1][:, None] - pos[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = rad[:, None] + rad[None, :]
            viol = np.maximum(0, min_dist - dist)
            # The diagonal is 0, so viol is 0 there.
            # We sum all elements. Each pair is counted twice.
            penalty += 0.5 * np.sum(viol**2)
            
            return sum_r, penalty
        
        # Initial energy
        current_pos = np.column_stack((x, y))
        current_rad = r.copy()
        current_sum, current_pen = calc_energy(current_pos, current_rad, W, H)
        current_energy = -current_sum + 10.0 * current_pen # Lambda = 10
        
        best_sum = current_sum
        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_W = W
        best_H = H
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a move
            move_type = np.random.random()
            
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            new_W = W
            new_H = H
            
            if move_type < 0.4:
                # Move a position
                i = np.random.randint(N)
                # Perturb x and y
                dx = np.random.normal(0, 0.05)
                dy = np.random.normal(0, 0.05)
                new_pos[i, 0] += dx
                new_pos[i, 1] += dy
            elif move_type < 0.8:
                # Move a radius
                i = np.random.randint(N)
                dr = np.random.normal(0, 0.02)
                new_rad[i] += dr
                if new_rad[i] < 0:
                    new_rad[i] = 0
            else:
                # Change aspect ratio
                dW = np.random.normal(0, 0.05)
                new_W = W + dW
                # Clamp W to reasonable range [0.5, 1.5]
                if new_W < 0.5: new_W = 0.5
                if new_W > 1.5: new_W = 1.5
                new_H = 2.0 - new_W
                # Scale positions to new box? 
                # If we change W and H, the positions might go out of bounds.
                # It's better to keep positions relative? 
                # Or just let the penalty handle it.
                # If we change W, the "space" changes.
                # Let's just update W and H. The penalty will push circles back in.
                # However, if W changes significantly, the circles might be stuck.
                # A better move for W is to scale x by W_new/W_old and y by H_new/H_old.
                scale_x = new_W / W
                scale_y = new_H / H
                new_pos[:, 0] *= scale_x
                new_pos[:, 1] *= scale_y
                # Also scale radii? No, radii are absolute.
                # If we scale positions, the relative distances change.
                # This is a valid move in the space of (pos, W).
            
            new_sum, new_pen = calc_energy(new_pos, new_rad, new_W, new_H)
            new_energy = -new_sum + 10.0 * new_pen
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_pos = new_pos
                current_rad = new_rad
                W = new_W
                H = new_H
                current_sum = new_sum
                current_pen = new_pen
                current_energy = new_energy
                
                if new_sum > best_sum and new_pen < 1e-6: # Accept only if feasible or near feasible
                    best_sum = new_sum
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
                    best_W = W
                    best_H = H
                    # We can also accept if penalty is very small, but sum is higher
                    # Actually, we should track the best *feasible* solution.
                    # If penalty is 0, it's feasible.
            
            T *= alpha
            
        # Final check: if best_pen was not 0, we might have an infeasible best_sum.
        # We should ensure we return a feasible solution.
        # Let's do a final local polish on the best feasible solution found.
        # Or simply, during SA, only update best if penalty is 0 (or < epsilon).
        # Let's re-run the loop logic to be safer: only update best if feasible.
        
        # If no feasible solution found, return a trivial one?
        # Unlikely for N=21.
        
        result = np.zeros((N, 3))
        result[:, 0] = best_pos[:, 0]
        result[:, 1] = best_pos[:, 1]
        result[:, 2] = best_rad
        
        return best_sum, result
    
    # Try multiple aspect ratios
    # W from 0.8 to 1.2 with step 0.1
    for W_init in np.linspace(0.8, 1.2, 9):
        if time.time() > deadline:
            break
        
        # Run SA with a few different seeds for robustness?
        # Given time limit, maybe 1 run per W is enough, or 2.
        val, res = run_sa(W_init, seed=int(W_init*100))
        
        if val > best_value:
            best_value = val
            best_result = res
        
        # Try a second seed if time permits
        if time.time() < deadline - 10:
            val2, res2 = run_sa(W_init, seed=int(W_init*100) + 1)
            if val2 > best_value:
                best_value = val2
                best_result = res2
    
    if best_result is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) / cols
            y[i] = (r_idx + 0.5) / rows
            r[i] = 0.09
        best_result = np.column_stack((x, y, r))
        best_value = np.sum(r)
        
    return best_result
