import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    # Analytic seed
    a = 1.0 / 3.0
    b = 1.0 / 3.0
    c = 1.0 / 6.0

    centers_seed = np.zeros((n, 2))
    sides_seed = np.zeros(n)

    centers_seed[0] = [a/2, a/2]
    centers_seed[1] = [1-a/2, a/2]
    centers_seed[2] = [a/2, 1-a/2]
    centers_seed[3] = [1-a/2, 1-a/2]
    sides_seed[0:4] = a

    centers_seed[4] = [0.5, b/2]
    centers_seed[5] = [0.5, 1-b/2]
    centers_seed[6] = [b/2, 0.5]
    centers_seed[7] = [1-b/2, 0.5]
    sides_seed[4:8] = b

    centers_seed[8] = [0.5, 5.0/12.0]
    centers_seed[9] = [0.5, 7.0/12.0]
    sides_seed[8:10] = c

    def get_cost_and_validity(centers, sides):
        pen = 0.0
        total_side = 0.0
        for i in range(n):
            total_side += sides[i]
            cx, cy = centers[i]
            s = sides[i]
            
            if cx - s/2 < 0:
                pen += (cx - s/2)**2
            if cx + s/2 > 1:
                pen += (cx + s/2 - 1)**2
            if cy - s/2 < 0:
                pen += (cy - s/2)**2
            if cy + s/2 > 1:
                pen += (cy + s/2 - 1)**2
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    ox = t - dx
                    oy = t - dy
                    pen += ox * oy
        return -total_side + 1000.0 * pen, pen < 1e-10, total_side

    best_centers = centers_seed.copy()
    best_sides = sides_seed.copy()
    best_score = np.sum(sides_seed)
    
    current_centers = centers_seed.copy()
    current_sides = sides_seed.copy()
    current_cost, _, _ = get_cost_and_validity(current_centers, current_sides)
    
    T_start = 0.5
    T_end = 1e-5
    alpha = 0.9995
    max_iter = 200000
    step_size = 0.02
    
    for it in range(max_iter):
        T = T_start * (alpha ** it)
        if T < T_end:
            T = T_end
            
        i = np.random.randint(n)
        var = np.random.randint(3)
        
        old_val = 0.0
        if var == 0:
            old_val = current_centers[i, 0]
            current_centers[i, 0] += np.random.uniform(-step_size, step_size)
            current_centers[i, 0] = np.clip(current_centers[i, 0], -0.1, 1.1)
        elif var == 1:
            old_val = current_centers[i, 1]
            current_centers[i, 1] += np.random.uniform(-step_size, step_size)
            current_centers[i, 1] = np.clip(current_centers[i, 1], -0.1, 1.1)
        else:
            old_val = current_sides[i]
            current_sides[i] += np.random.uniform(-step_size, step_size)
            current_sides[i] = max(1e-4, current_sides[i])
            
        new_cost, is_valid, total_side = get_cost_and_validity(current_centers, current_sides)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
            if is_valid and total_side > best_score:
                best_score = total_side
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
        else:
            if var == 0:
                current_centers[i, 0] = old_val
            elif var == 1:
                current_centers[i, 1] = old_val
            else:
                current_sides[i] = old_val
                
    if best_score <= 0:
        return centers_seed, sides_seed
        
    return best_centers, best_sides
