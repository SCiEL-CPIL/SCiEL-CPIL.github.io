import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin
    
    best_result = None
    best_sum = 0.0
    
    # Try various aspect ratios
    aspect_ratios = np.linspace(0.3, 0.7, 15)
    
    for a in aspect_ratios:
        if time.time() - start_time > deadline:
            break
        
        w = 2.0 * a
        h = 2.0 * (1.0 - a)
        
        # Initial guess: grid placement
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        
        # Map to actual coordinates
        init_params = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x0 = xs[i] * w
                y0 = ys[j] * h
                # Initial radius: half the minimum spacing
                r0 = min(w, h) / side * 0.4
                init_params.extend([x0, y0, r0])
                count += 1
        
        # Pad if needed
        while len(init_params) < 3 * N:
            init_params.extend([w/2, h/2, 0.01])
        
        init_params = np.array(init_params[:3*N])
        
        def objective(params):
            # params: [x1, y1, r1, x2, y2, r2, ...]
            x = params[0::3]
            y = params[1::3]
            r = params[2::3]
            
            # Sum of radii (we want to maximize, so negate)
            sum_r = np.sum(r)
            
            # Penalty for violations
            penalty = 0.0
            
            # Boundary constraints
            for i in range(N):
                if r[i] < 0:
                    penalty += 100.0 * r[i]
                if x[i] - r[i] < 0:
                    penalty += 10.0 * (0 - (x[i] - r[i]))
                if x[i] + r[i] > w:
                    penalty += 10.0 * ((x[i] + r[i]) - w)
                if y[i] - r[i] < 0:
                    penalty += 10.0 * (0 - (y[i] - r[i]))
                if y[i] + r[i] > h:
                    penalty += 10.0 * ((y[i] + r[i]) - h)
            
            # Overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += 10.0 * (min_dist - dist)
            
            return -sum_r + penalty
        
        # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))      # x
            bounds.append((0, h))      # y
            bounds.append((0.001, min(w, h)/2))  # r
        
        try:
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                params = result.x
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                
                # Verify validity
                valid = True
                for i in range(N):
                    if r[i] <= 0 or x[i] - r[i] < -1e-9 or x[i] + r[i] > w + 1e-9 or \
                       y[i] - r[i] < -1e-9 or y[i] + r[i] > h + 1e-9:
                        valid = False
                        break
                    if not valid:
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    total_sum = np.sum(r)
                    if total_sum > best_sum:
                        best_sum = total_sum
                        # Scale to perimeter 4: already w+h=2, so perimeter=4
                        best_result = np.column_stack([x, y, r])
                        
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
