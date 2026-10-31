import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -np.inf
    best_circles = None
    best_alpha = 1.0
    
    # Helper to check validity and score
    def evaluate(circles, alpha):
        # circles: (N, 3) array of [x, y, r]
        # alpha: aspect ratio W/H
        # Perimeter 4 => W + H = 2
        H = 2.0 / (1.0 + alpha)
        W = 2.0 * alpha / (1.0 + alpha)
        
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundaries
        # x must be in [r, W-r]
        # y must be in [r, H-r]
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9:
                valid = False
                break
            if y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9:
                valid = False
                break
        if not valid:
            return -np.inf, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r*sum_r - 1e-9:
                    return -np.inf, False
                    
        score = np.sum(r)
        return score, True

    def get_energy(circles, alpha):
        """Returns penalty energy. Lower is better. 0 means valid."""
        H = 2.0 / (1.0 + alpha)
        W = 2.0 * alpha / (1.0 + alpha)
        
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary violations
        energy = 0.0
        # Left
        viol = np.maximum(0.0, r - x)
        energy += np.sum(viol**2)
        # Right
        viol = np.maximum(0.0, (W - x) - r)
        energy += np.sum(viol**2)
        # Bottom
        viol = np.maximum(0.0, r - y)
        energy += np.sum(viol**2)
        # Top
        viol = np.maximum(0.0, (H - y) - r)
        energy += np.sum(viol**2)
        
        # Overlap violations
        # Vectorized distance calculation
        # dist_sq[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
        # We only need i < j
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        sum_r = r[:, None] + r[None, :]
        
        # Overlap amount is max(0, sum_r - dist)
        # We square it for smoothness
        # dist = np.sqrt(dist_sq)
        # overlap = np.maximum(0.0, sum_r - dist)
        # energy += np.sum(overlap**2)
        
        # To avoid sqrt for speed, we can use squared distances carefully?
        # No, overlap is linear in distance. We need dist.
        # For N=21, sqrt is fine.
        dist = np.sqrt(dist_sq)
        # Set diagonal to inf to ignore self-overlap
        np.fill_diagonal(dist, np.inf)
        overlap = np.maximum(0.0, sum_r - dist)
        energy += np.sum(overlap**2)
        
        return energy

    def sa_restart(alpha_init):
        nonlocal best_score, best_circles, best_alpha
        nonlocal start_time, deadline
        
        # Initialize circles
        # Start with small radii and random positions in a square
        H = 2.0 / (1.0 + alpha_init)
        W = 2.0 * alpha_init / (1.0 + alpha_init)
        
        # Initial radii: small
        r_init = 0.05
        # Initial positions: random within safe bounds
        circles = np.zeros((N, 3))
        for i in range(N):
            circles[i, 2] = r_init
            circles[i, 0] = r_init + random.random() * (W - 2*r_init)
            circles[i, 1] = r_init + random.random() * (H - 2*r_init)
            
        alpha = alpha_init
        T = 0.1 # Initial temperature
        cooling_rate = 0.999
        min_T = 1e-6
        
        current_energy = get_energy(circles, alpha)
        # Score is sum(r) - penalty
        # We want to maximize sum(r). 
        # Let's define Objective = sum(r) - M * energy
        # M should be large.
        M = 100.0
        current_obj = np.sum(circles[:, 2]) - M * current_energy
        
        local_best_obj = current_obj
        local_best_circles = circles.copy()
        local_best_alpha = alpha
        
        step_size_pos = 0.1
        step_size_r = 0.05
        step_size_alpha = 0.05
        
        while T > min_T:
            if time.time() > deadline:
                break
                
            # Generate move
            move_type = random.random()
            new_circles = circles.copy()
            new_alpha = alpha
            
            if move_type < 0.7:
                # Move position
                i = random.randint(0, N-1)
                new_circles[i, 0] += random.gauss(0, step_size_pos)
                new_circles[i, 1] += random.gauss(0, step_size_pos)
            elif move_type < 0.9:
                # Move radius
                i = random.randint(0, N-1)
                new_circles[i, 2] += random.gauss(0, step_size_r)
                # Keep radius positive
                if new_circles[i, 2] < 0.01:
                    new_circles[i, 2] = 0.01
            else:
                # Move aspect ratio
                new_alpha += random.gauss(0, step_size_alpha)
                if new_alpha < 0.5: new_alpha = 0.5
                if new_alpha > 2.0: new_alpha = 2.0
                
            new_energy = get_energy(new_circles, new_alpha)
            new_obj = np.sum(new_circles[:, 2]) - M * new_energy
            
            delta_obj = new_obj - current_obj
            
            if delta_obj > 0:
                # Accept
                circles = new_circles
                alpha = new_alpha
                current_energy = new_energy
                current_obj = new_obj
                
                if current_obj > local_best_obj:
                    local_best_obj = current_obj
                    local_best_circles = circles.copy()
                    local_best_alpha = alpha
            else:
                # Accept with probability exp(delta_obj / T)
                if random.random() < np.exp(delta_obj / T):
                    circles = new_circles
                    alpha = new_alpha
                    current_energy = new_energy
                    current_obj = new_obj
                    
            T *= cooling_rate
            
            # Periodic check for validity to update global best
            # Doing this every 1000 steps might be too slow if we do full O(N^2) check in Python
            # But get_energy is vectorized. Let's check validity using the energy.
            # If energy is very close to 0, it's likely valid.
            if current_energy < 1e-6:
                score, valid = evaluate(circles, alpha)
                if valid and score > best_score:
                    best_score = score
                    best_circles = circles.copy()
                    best_alpha = alpha

        # Final check for this restart
        score, valid = evaluate(local_best_circles, local_best_alpha)
        if valid and score > best_score:
            best_score = score
            best_circles = local_best_circles.copy()
            best_alpha = local_best_alpha

    # Run multiple restarts with different initial aspect ratios
    # We have ~140s. Let's do 3 restarts.
    alphas = [1.0, 0.8, 1.2]
    
    for alpha_init in alphas:
        if time.time() > deadline:
            break
        sa_restart(alpha_init)
        
    if best_circles is None:
        # Fallback to seed if something went wrong
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
