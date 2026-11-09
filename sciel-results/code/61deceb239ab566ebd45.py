import numpy as np
import random

N = 21

def construct():
    # Simulated Annealing for Circle Packing
    # Variables: w, x[0..N-1], y[0..N-1], r[0..N-1]
    
    def get_energy(w, x, y, r):
        h = 2.0 - w
        # Penalty for out of bounds
        penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > w - r[i]:
                penalty += (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > h - r[i]:
                penalty += (y[i] - (h - r[i]))**2
        
        # Penalty for overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    penalty += (sum_r - dist)**2
        
        # Objective: Minimize -sum(r) + Penalty
        # We want to maximize sum(r), so minimize -sum(r)
        # If penalty is 0, we want sum(r) to be as large as possible.
        # So Energy = -sum(r) + Penalty.
        return -np.sum(r) + 10.0 * penalty

    def generate_initial():
        # Hexagonal packing in a 1x1 square (w=1, h=1)
        # 21 circles. 
        # Let's try a simple grid 5x5 in 1x1, r=0.18
        w = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # 5x5 grid
        side = 5
        spacing = 1.0 / side
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x[idx] = (i + 0.5) * spacing
                y[idx] = (j + 0.5) * spacing
                r[idx] = spacing / 2.0 * 0.9 # Start slightly smaller
                idx += 1
            if idx >= N: break
        
        # Fill remaining if any (should be none for 25 grid, but we only take 21)
        return w, x, y, r

    w, x, y, r = generate_initial()
    
    best_w, best_x, best_y, best_r = w, x.copy(), y.copy(), r.copy()
    best_energy = get_energy(w, x, y, r)
    
    T = 0.05 # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    max_iter = 50000 # Adjust based on time limit
    
    # Pre-calculate indices for speed? No, N=21 is small.
    
    for i in range(max_iter):
        if T < T_min:
            break
            
        # Generate new state
        new_w = w
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Move a circle
            idx = random.randint(0, N-1)
            step = T * 0.1
            new_x[idx] += random.gauss(0, step)
            new_y[idx] += random.gauss(0, step)
        elif move_type < 0.7:
            # Change radius
            idx = random.randint(0, N-1)
            step = T * 0.01
            new_r[idx] += random.gauss(0, step)
            if new_r[idx] < 0.01: new_r[idx] = 0.01
            if new_r[idx] > 1.0: new_r[idx] = 1.0
        elif move_type < 0.9:
            # Change width
            step = T * 0.05
            new_w += random.gauss(0, step)
            if new_w < 0.5: new_w = 0.5
            if new_w > 1.5: new_w = 1.5
        else:
            # Random kick: move 2 circles
            idx1 = random.randint(0, N-1)
            idx2 = random.randint(0, N-1)
            step = T * 0.1
            new_x[idx1] += random.gauss(0, step)
            new_y[idx1] += random.gauss(0, step)
            new_x[idx2] += random.gauss(0, step)
            new_y[idx2] += random.gauss(0, step)
            
        # Clip positions to box roughly to avoid huge penalties
        h = 2.0 - new_w
        for k in range(N):
            new_x[k] = np.clip(new_x[k], 0, new_w)
            new_y[k] = np.clip(new_y[k], 0, h)
            
        new_energy = get_energy(new_w, new_x, new_y, new_r)
        
        # Acceptance criterion
        delta = new_energy - best_energy
        if delta < 0 or random.random() < np.exp(-delta / T):
            w, x, y, r = new_w, new_x, new_y, new_r
            if new_energy < best_energy:
                best_energy = new_energy
                best_w, best_x, best_y, best_r = w, x.copy(), y.copy(), r.copy()
                
        T *= cooling
        
    # Return the best found
    # Check validity
    h = 2.0 - best_w
    valid = True
    for i in range(N):
        if best_x[i] < best_r[i] or best_x[i] > best_w - best_r[i] or best_y[i] < best_r[i] or best_y[i] > h - best_r[i] or best_r[i] <= 0:
            valid = False
            break
    for i in range(N):
        for j in range(i+1, N):
            dist2 = (best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2
            sum_r = best_r[i] + best_r[j]
            if dist2 < sum_r**2 - 1e-9:
                valid = False
                break
        if not valid: break
        
    if valid:
        return np.column_stack((best_x, best_y, best_r))
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
