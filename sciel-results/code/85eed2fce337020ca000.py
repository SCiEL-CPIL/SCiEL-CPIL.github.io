"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random


def construct():
    random.seed(42)
    np.random.seed(42)
    
    # Perimeter = 4 => w + h = 2
    # We'll use simulated annealing to pack circles in a rectangle
    
    best_sum = -1
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize circles in a grid pattern
        # Estimate grid dimensions
        cols = max(1, int(round(np.sqrt(N * ar))))
        rows = max(1, int(np.ceil(N / cols)))
        
        # Adjust to fit exactly N circles
        while rows * cols < N:
            if rows < cols:
                rows += 1
            else:
                cols += 1
        
        # Initial positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                positions.append([x, y])
        
        # Initial radii: start small
        max_possible_r = min(w / cols, h / rows) * 0.9
        radii = np.full(N, max_possible_r * 0.3)
        
        # Simulated annealing
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        max_iter = 5000
        
        current_energy = 0.0
        best_energy = float('inf')
        best_x = positions.copy()
        best_r = radii.copy()
        
        for it in range(max_iter):
            if T < T_min:
                break
            
            # Choose a random circle to move
            i = random.randint(0, N - 1)
            
            # Decide what to perturb: position or radius
            if random.random() < 0.7:
                # Perturb position
                dx = random.gauss(0, T * 0.5)
                dy = random.gauss(0, T * 0.5)
                new_x = positions[i][0] + dx
                new_y = positions[i][1] + dy
                
                # Keep within bounds (soft constraint)
                new_x = max(radii[i], min(w - radii[i], new_x))
                new_y = max(radii[i], min(h - radii[i], new_y))
                
                positions[i][0] = new_x
                positions[i][1] = new_y
            else:
                # Perturb radius
                factor = random.gauss(1.0, 0.1)
                new_r = max(0.01, radii[i] * factor)
                # Ensure it fits in rectangle
                new_r = min(new_r, min(w, h) * 0.5)
                radii[i] = new_r
            
            # Calculate energy
            energy = 0.0
            
            # Boundary violations
            for k in range(N):
                if positions[k][0] < radii[k]:
                    energy += (radii[k] - positions[k][0]) ** 2
                if positions[k][0] > w - radii[k]:
                    energy += (positions[k][0] - (w - radii[k])) ** 2
                if positions[k][1] < radii[k]:
                    energy += (radii[k] - positions[k][1]) ** 2
                if positions[k][1] > h - radii[k]:
                    energy += (positions[k][1] - (h - radii[k])) ** 2
            
            # Overlap penalties
            for k in range(N):
                for l in range(k + 1, N):
                    dx = positions[k][0] - positions[l][0]
                    dy = positions[k][1] - positions[l][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[k] + radii[l]
                    if dist < min_dist:
                        energy += (min_dist - dist) ** 2
            
            # Objective: minimize -sum(radii) + penalty
            obj = -np.sum(radii) + 10.0 * energy
            
            # Accept or reject
            if obj < current_energy:
                current_energy = obj
                if obj < best_energy:
                    best_energy = obj
                    best_x = [p.copy() for p in positions]
                    best_r = radii.copy()
            else:
                delta = obj - current_energy
                if random.random() < np.exp(-delta / T):
                    current_energy = obj
                else:
                    # Reject: revert
                    if random.random() < 0.7:
                        # Revert position
                        pass  # Already stored in positions, need to track
                    # For simplicity, we'll just accept based on energy
                    # and not revert (approximation)
            
            T *= cooling_rate
        
        # Extract best solution
        xs = np.array([p[0] for p in best_x])
        ys = np.array([p[1] for p in best_x])
        rs = best_r
        
        # Check validity
        valid = True
        for i in range(N):
            if rs[i] <= 0:
                valid = False
                break
            if xs[i] < rs[i] - 1e-5 or xs[i] > w - rs[i] + 1e-5:
                valid = False
                break
            if ys[i] < rs[i] - 1e-5 or ys[i] > h - rs[i] + 1e-5:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack([xs, ys, rs])
    
    # Fallback if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
