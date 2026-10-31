"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    def energy_and_grad(x, y, r, W, H):
        """
        Compute energy and gradients for force-directed relaxation.
        Variables: x, y, r (arrays of length N), W, H (scalars)
        Energy terms:
        1. Overlap penalty: sum_{i<j} max(0, r_i + r_j - dist_ij)^2
        2. Wall penalty: sum_i max(0, r_i - min(x_i, W-x_i, y_i, H-y_i))^2
        3. Perimeter penalty: (2*(W+H) - 4)^2
        4. Radius regularization: -sum(r_i) (to maximize radii, we minimize -sum(r))
           Actually, we want to maximize sum(r). So we can minimize E = overlap + walls - lambda * sum(r)
           Or simply: Minimize E = overlap + walls, and maximize sum(r) via the optimizer's objective?
           Standard approach: Minimize E = overlap + walls + penalty_for_perimeter.
           But we want to maximize sum(r).
           Let's define the objective to minimize as:
           E = overlap_penalty + wall_penalty + perimeter_penalty - alpha * sum(r)
           where alpha is a weight to encourage larger circles.
           However, a better way is to fix the perimeter and just minimize overlap+wall,
           letting the circles grow as much as possible.
           Actually, the force-directed method usually works by:
           1. Start with small circles.
           2. Grow them slightly.
           3. Relax positions to reduce overlap.
           4. Repeat.
           
           Let's implement a single optimization step that minimizes:
           E = sum_{i<j} (max(0, r_i+r_j-d_ij))^2 + sum_i (max(0, r_i - dist_to_wall_i))^2
           + (2*(W+H)-4)^2 * 100  # Strong perimeter constraint
           
           And we want to maximize sum(r).
           We can do this by adding a term: -beta * sum(r) to the energy.
           But this might conflict.
           
           Alternative: 
           Use scipy.optimize.minimize to minimize E_overlap + E_wall subject to perimeter=4.
           But we want to maximize sum(r).
           
           Let's use the "Grow and Relax" strategy within the annealing loop.
           In each step:
           1. Increase all r by a small factor.
           2. Perform a few gradient descent steps on positions x,y to minimize overlap+wall.
           3. Update W,H to satisfy perimeter.
           
           For the gradient descent on x,y:
           E = sum_{i<j} max(0, r_i+r_j-d_ij)^2 + sum_i max(0, r_i - min(x_i, W-x_i, ...))^2
           Gradients are straightforward.
        """
        n = len(x)
        E = 0.0
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        grad_W = 0.0
        grad_H = 0.0
        
        # 1. Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                min_dist = r[i] + r[j]
                overlap = min_dist - dist
                if overlap > 0:
                    E += overlap * overlap
                    # dE/ddist = -2 * overlap
                    # ddist/dx_i = dx/dist
                    force_x = 2 * overlap * (dx / dist)
                    force_y = 2 * overlap * (dy / dist)
                    grad_x[i] += force_x
                    grad_y[i] += force_y
                    grad_x[j] -= force_x
                    grad_y[j] -= force_y
                    # dE/dr_i = 2 * overlap
                    grad_r[i] += 2 * overlap
                    grad_r[j] += 2 * overlap
                else:
                    # No overlap, no force from this pair
                    pass
        
        # 2. Walls
        for i in range(n):
            # Left
            d_left = x[i]
            if d_left < r[i]:
                viol = r[i] - d_left
                E += viol * viol
                grad_x[i] += 2 * viol * (-1)  # d(viol)/dx = -1
                grad_r[i] += 2 * viol
                grad_W += 0 # W doesn't affect left wall distance directly in this term? 
                # Actually, x is absolute. Left wall is x=0.
            # Right
            d_right = W - x[i]
            if d_right < r[i]:
                viol = r[i] - d_right
                E += viol * viol
                grad_x[i] += 2 * viol * (1)  # d(viol)/dx = -d(W-x)/dx = 1? No.
                # viol = r - (W - x) = r - W + x
                # d(viol)/dx = 1
                # d(viol)/dW = -1
                # d(viol)/dr = 1
                grad_x[i] += 2 * viol * 1
                grad_r[i] += 2 * viol
                grad_W += 2 * viol * (-1)
            
            # Bottom
            d_bottom = y[i]
            if d_bottom < r[i]:
                viol = r[i] - d_bottom
                E += viol * viol
                grad_y[i] += 2 * viol * (-1)
                grad_r[i] += 2 * viol
            
            # Top
            d_top = H - y[i]
            if d_top < r[i]:
                viol = r[i] - d_top
                E += viol * viol
                grad_y[i] += 2 * viol * 1
                grad_r[i] += 2 * viol
                grad_H += 2 * viol * (-1)
        
        # 3. Perimeter constraint: 2(W+H) = 4 => W+H=2
        # We will handle this by projecting W,H after each step or adding a penalty.
        # Let's add a penalty to energy to keep W+H close to 2.
        perim_viol = (W + H - 2.0)
        E += 100.0 * perim_viol * perim_viol
        grad_W += 200.0 * perim_viol
        grad_H += 200.0 * perim_viol
        
        # 4. Radius maximization: We want to maximize sum(r).
        # This is equivalent to minimizing -sum(r).
        # But if we just minimize E, circles will shrink to 0 to avoid overlap.
        # So we MUST add a term that encourages growth.
        # E_growth = - lambda * sum(r)
        # Let's use lambda = 1.0.
        E -= 1.0 * np.sum(r)
        grad_r -= 1.0
        
        return E, grad_x, grad_y, grad_r, grad_W, grad_H

    def hexagonal_init(W, H, N):
        """Initialize circles in a hexagonal pattern scaled to fit W,H."""
        # Approximate number of circles in a hex grid
        # We just place them in a grid and jitter
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Grid positions
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) * W / side
            y[i] = (row + 0.5) * H / side
            r[i] = 0.1 * min(W, H) / side # Small initial radius
            
        # Add jitter
        x += np.random.uniform(-0.01, 0.01, N)
        y += np.random.uniform(-0.01, 0.01, N)
        
        # Clip to bounds
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)
        
        return x, y, r

    def relax(x, y, r, W, H, steps=50, lr=0.01):
        """Perform gradient descent steps to minimize energy."""
        for s in range(steps):
            E, gx, gy, gr, gW, gH = energy_and_grad(x, y, r, W, H)
            
            # Update positions
            x -= lr * gx
            y -= lr * gy
            
            # Update radii (only allow growth? or free?)
            # If we allow free update, they might shrink.
            # We want to grow them. So we only take the positive part of the gradient step for r?
            # Or we can just let the optimizer find the balance.
            # Given the -sum(r) term, it will push r up.
            r -= lr * gr
            r = np.maximum(r, 1e-4) # Prevent negative radii
            
            # Update W, H
            W -= lr * gW
            H -= lr * gH
            
            # Project W, H to maintain perimeter 4
            # W + H = 2
            # If W+H != 2, adjust.
            # Simple projection: 
            # Let S = W + H. We want S=2.
            # Scale W, H? Or just set H = 2 - W?
            # Let's keep W free and set H = 2 - W.
            # But we have gradients for both.
            # Let's just enforce H = 2 - W at the end of each step.
            H = 2.0 - W
            
            # Clip W, H to reasonable ranges
            W = np.clip(W, 0.2, 1.8)
            H = 2.0 - W
            
            # Clip circles to bounds
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
        return x, y, r, W, H

    def is_valid(x, y, r, W, H, tol=1e-4):
        """Check validity."""
        if abs(2*(W+H) - 4) > 0.1:
            return False
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - tol:
                    return False
        return True

    # Main loop
    np.random.seed(42)
    
    # Try different initial aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W_init in aspect_ratios:
        if time.time() > deadline - 20:
            break
        
        H_init = 2.0 - W_init
        if H_init <= 0.1:
            continue
        
        # Multiple restarts for each aspect ratio
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            x, y, r = hexagonal_init(W_init, H_init, N)
            W = W_init
            H = H_init
            
            # Annealing loop
            # We will grow the radii and relax.
            # Instead of just gradient descent, we use a simulated annealing-like approach:
            # 1. Grow radii by a factor.
            # 2. Relax positions.
            # 3. Accept if energy decreases or with some probability (Metropolis).
            
            current_E, _, _, _, _, _ = energy_and_grad(x, y, r, W, H)
            best_local_E = current_E
            best_local_x, best_local_y, best_local_r, best_local_W, best_local_H = x.copy(), y.copy(), r.copy(), W, H
            
            T = 1.0 # Temperature
            cooling_rate = 0.995
            
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Grow radii
                growth = 1.0 + 0.005
                r_new = r * growth
                # Cap radius to not exceed half min(W,H)
                max_r = min(W, H) / 2.0 * 0.9
                r_new = np.minimum(r_new, max_r)
                
                # Relax positions with new radii
                x_rel, y_rel, r_rel, W_rel, H_rel = relax(x, y, r_new, W, H, steps=20, lr=0.05)
                
                # Calculate new energy
                new_E, _, _, _, _, _ = energy_and_grad(x_rel, y_rel, r_rel, W_rel, H_rel)
                
                # Metropolis criterion
                delta_E = new_E - current_E
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    x, y, r, W, H = x_rel, y_rel, r_rel, W_rel, H_rel
                    current_E = new_E
                    
                    # Track best
                    if current_E < best_local_E:
                        best_local_E = current_E
                        best_local_x, best_local_y, best_local_r, best_local_W, best_local_H = x.copy(), y.copy(), r.copy(), W, H
                        
                T *= cooling_rate
            
            # Check best local configuration
            if is_valid(best_local_x, best_local_y, best_local_r, best_local_W, best_local_H):
                total_r = np.sum(best_local_r)
                if total_r > best_sum_r:
                    best_sum_r = total_r
                    best_config = np.column_stack((best_local_x, best_local_y, best_local_r))
                    best_W = best_local_W
                    best_H = best_local_H

    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
