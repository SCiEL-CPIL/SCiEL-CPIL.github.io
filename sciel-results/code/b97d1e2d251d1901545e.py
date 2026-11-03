"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_radius_for_circle(i, centers, radii, n):
    """Compute maximum radius for circle i given other circles and boundaries."""
    x, y = centers[i]
    r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(n):
        if i == j:
            continue
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        d = np.sqrt(dx*dx + dy*dy)
        r_candidate = d - radii[j]
        if r_candidate < r:
            r = r_candidate
    return max(0.0, r)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    n = N
    
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    def try_configuration(centers_init, radii_init, max_iterations=3000):
        nonlocal best_centers, best_radii, best_sum, deadline
        
        centers = centers_init.copy()
        radii = radii_init.copy()
        current_sum = float(np.sum(radii))
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Grow pass
            for i in range(n):
                if time.time() > deadline:
                    break
                max_r = _max_radius_for_circle(i, centers, radii, n)
                if max_r > radii[i] + 1e-12:
                    radii[i] = max_r
                    current_sum = float(np.sum(radii))
                    improved = True
            
            # Move pass
            for i in range(n):
                if time.time() > deadline:
                    break
                
                best_move_sum = current_sum
                best_pos = centers[i].copy()
                
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    for dist in [0.01, 0.02, 0.03, 0.05, 0.08]:
                        new_x = centers[i, 0] + dist * np.cos(angle)
                        new_y = centers[i, 1] + dist * np.sin(angle)
                        
                        if new_x < 0.001 or new_x > 0.999 or new_y < 0.001 or new_y > 0.999:
                            continue
                        
                        valid = True
                        for j in range(n):
                            if i == j:
                                continue
                            dx = new_x - centers[j, 0]
                            dy = new_y - centers[j, 1]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d < radii[i] + radii[j] - 1e-12:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        old_c = centers[i].copy()
                        centers[i] = [new_x, new_y]
                        new_max_r = _max_radius_for_circle(i, centers, radii, n)
                        new_sum = current_sum - radii[i] + new_max_r
                        centers[i] = old_c
                        
                        if new_sum > best_move_sum + 1e-12:
                            best_move_sum = new_sum
                            best_pos = np.array([new_x, new_y])
                            improved = True
                
                if np.any(best_pos != centers[i]):
                    centers[i] = best_pos
                    radii[i] = _max_radius_for_circle(i, centers, radii, n)
                    current_sum = float(np.sum(radii))
            
            if not improved:
                break
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        return best_centers, best_radii, best_sum
    
    # Initial configs
    side = 5
    base_configs = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:n]
    
    # Config 1: Grid
    c1 = np.array(base_configs)
    r1 = np.full(n, 0.5 / side)
    try_configuration(c1, r1)
    
    # Config 2: Perturbed grid
    c2 = c1 + np.random.randn(n, 2) * 0.015
    c2 = np.clip(c2, 0.05, 0.95)
    r2 = np.full(n, 0.5 / side)
    try_configuration(c2, r2)
    
    # Config 3: Offset grid
    c3 = c1.copy()
    for i in range(n):
        row = i // side
        if row % 2 == 1:
            c3[i, 0] += 0.5 / side
    c3 = np.clip(c3, 0.05, 0.95)
    r3 = np.full(n, 0.5 / side)
    try_configuration(c3, r3)
    
    # Config 4: Random
    c4 = np.random.rand(n, 2) * 0.7 + 0.15
    r4 = np.full(n, 0.04)
    try_configuration(c4, r4)
    
    # Polish best
    if best_centers is not None:
        centers = best_centers.copy()
        radii = best_radii.copy()
        
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                        for dist in [0.002, 0.005, 0.01]:
                            new_x = centers[i, 0] + dist * np.cos(angle)
                            new_y = centers[i, 1] + dist * np.sin(angle)
                            if new_x < 0.001 or new_x > 0.999 or new_y < 0.001 or new_y > 0.999:
                                continue
                            valid = True
                            for j in range(n):
                                if i == j:
                                    continue
                                dx = new_x - centers[j, 0]
                                dy = new_y - centers[j, 1]
                                d = np.sqrt(dx*dx + dy*dy)
                                if d < radii[i] + radii[j] - 1e-12:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            old_c = centers[i].copy()
                            old_r = radii[i]
                            centers[i] = [new_x, new_y]
                            new_r = _max_radius_for_circle(i, centers, radii, n)
                            new_sum = np.sum(radii) - old_r + new_r
                            if new_sum > np.sum(radii) + 1e-12:
                                radii[i] = new_r
                                improved = True
                            else:
                                centers[i] = old_c
                                radii[i] = old_r
                
                for i in range(n):
                    max_r = _max_radius_for_circle(i, centers, radii, n)
                    if max_r > radii[i] + 1e-12:
                        radii[i] = max_r
                        improved = True
            
            if np.sum(radii) > best_sum:
                best_sum = float(np.sum(radii))
                best_centers = centers.copy()
                best_radii = radii.copy()
    
    if best_centers is not None:
        result = np.column_stack([best_centers, best_radii])
        for i in range(n):
            r = result[i, 2]
            result[i, 2] = min(r, result[i, 0], 1-result[i, 0], result[i, 1], 1-result[i, 1])
            result[i, 0] = np.clip(result[i, 0], result[i, 2], 1-result[i, 2])
            result[i, 1] = np.clip(result[i, 1], result[i, 2], 1-result[i, 2])
        return result
    else:
        cells = [((i + 0.5) / 5, (j + 0.5) / 5, 0.1 - 1e-3) for i in range(5) for j in range(5)]
        return np.array(cells[:n])
# EVOLVE-BLOCK-END
