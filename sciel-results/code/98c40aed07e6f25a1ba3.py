"""Rewritten program: continuous geometric optimization for circle packing."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_score = 0.0
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = np.linspace(0.5, 2.0, 15)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        # Rectangle dimensions: perimeter = 2*(w+h) = 4 => w+h = 2
        # Let w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Initialize with hexagonal lattice pattern
        np.random.seed(int(ar * 1000) % 10000)
        
        # Create initial positions in a grid-like pattern with some randomness
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Estimate initial radius
        # Area of rectangle = w*h, 21 circles
        # Average area per circle = w*h/21
        # r_avg = sqrt(area_per_circle / pi)
        area_per_circle = w * h / N
        r_avg = np.sqrt(area_per_circle / np.pi)
        
        # Place circles in a grid pattern
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        for idx in range(N):
            i = idx % cols
            j = idx // cols
            # Hexagonal offset for odd rows
            x_offset = (i + 0.5) * w / cols
            if j % 2 == 1:
                x_offset += 0.5 * w / cols
            y_offset = (j + 0.5) * h / rows
            
            # Add small random perturbation
            positions[idx, 0] = x_offset + np.random.uniform(-0.02, 0.02)
            positions[idx, 1] = y_offset + np.random.uniform(-0.02, 0.02)
            radii[idx] = r_avg * (0.9 + 0.2 * np.random.random())
        
        # Optimize using gradient-based approach
        result = _optimize(positions, radii, w, h, deadline)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result.copy()
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result


def _optimize(positions, radii, w, h, deadline):
    """Optimize circle packing using gradient descent with constraint handling."""
    n = len(radii)
    
    # Combine positions and radii into single parameter vector
    params = np.concatenate([positions.flatten(), radii])
    
    learning_rate = 0.01
    penalty_weight = 100.0
    
    for iteration in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        # Unpack parameters
        pos = params[:2*n].reshape(n, 2)
        rad = params[2*n:]
        
        # Compute objective: maximize sum of radii
        # Penalty for violations:
        # 1. Circles outside rectangle
        # 2. Overlapping circles
        
        # Gradient of objective: d(sum r)/d(r_i) = 1
        grad = np.zeros_like(params)
        grad[2*n:] = 1.0  # derivative of sum(radii) w.r.t. radii
        
        # Penalty gradients
        # 1. Boundary violations
        for i in range(n):
            # Left boundary: x_i - r_i >= 0
            if pos[i, 0] - rad[i] < 0:
                grad[2*i] += penalty_weight
                grad[2*n+i] -= penalty_weight
            # Right boundary: w - x_i - r_i >= 0
            if w - pos[i, 0] - rad[i] < 0:
                grad[2*i] -= penalty_weight
                grad[2*n+i] -= penalty_weight
            # Bottom boundary: y_i - r_i >= 0
            if pos[i, 1] - rad[i] < 0:
                grad[2*i+1] += penalty_weight
                grad[2*n+i] -= penalty_weight
            # Top boundary: h - y_i - r_i >= 0
            if h - pos[i, 1] - rad[i] < 0:
                grad[2*i+1] -= penalty_weight
                grad[2*n+i] -= penalty_weight
        
        # 2. Overlap violations: distance(i,j) >= r_i + r_j
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                overlap = rad[i] + rad[j] - dist
                
                if overlap > 0:
                    # Penalty gradient
                    px = dx / dist
                    py = dy / dist
                    
                    # Force i away from j
                    grad[2*i] += penalty_weight * px
                    grad[2*i+1] += penalty_weight * py
                    # Force j away from i
                    grad[2*j] -= penalty_weight * px
                    grad[2*j+1] -= penalty_weight * py
                    # Radii should decrease
                    grad[2*n+i] -= penalty_weight * 0.5
                    grad[2*n+j] -= penalty_weight * 0.5
        
        # Update parameters
        params += learning_rate * grad
        
        # Clip radii to be positive
        params[2*n:] = np.maximum(params[2*n:], 0.001)
        
        # Clip positions to stay within bounds
        for i in range(n):
            params[2*i] = np.clip(params[2*i], params[2*n+i], w - params[2*n+i])
            params[2*i+1] = np.clip(params[2*i+1], params[2*n+i], h - params[2*n+i])
        
        # Decay learning rate
        learning_rate *= 0.999
    
    # Final result
    pos = params[:2*n].reshape(n, 2)
    rad = params[2*n:]
    
    # Verify validity
    valid = True
    for i in range(n):
        if pos[i, 0] < rad[i] - 1e-9 or pos[i, 0] > w - rad[i] + 1e-9:
            valid = False
            break
        if pos[i, 1] < rad[i] - 1e-9 or pos[i, 1] > h - rad[i] + 1e-9:
            valid = False
            break
    
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < rad[i] + rad[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return None
    
    return np.column_stack([pos, rad])
# EVOLVE-BLOCK-END
