<<<<<<< SEARCH
import numpy as np

N = 200

def construct():
    # Generate multiple initial configurations
    configs = []
    
    # Config 1: 10x20 Rectangular Grid
    m, n = 10, 20
    xs = np.linspace(-0.5, 0.5, n)
    ys = np.linspace(-0.5, 0.5, m)
    grid = np.array([(x, y) for y in ys for x in xs])
    configs.append(grid)

    # Config 2: 14x14 Grid (196 points) + 4 points
    # We can just use a 15x14 grid (210 points) and drop 10
    # Or better, use a 14x14 grid and add 4 points in the center of the 4 largest "gaps"?
    # Simpler: Use a 15x15 grid (225 points) and drop 25.
    # Let's try 14x14 (196) and add 4 points at specific locations.
    # Actually, let's just use a 10x20 grid and a 14x14 grid and a Hex grid.
    
    # Config 3: Hexagonal Lattice
    # Estimate spacing d such that we get ~200 points
    # Area per point ~ d^2 * sqrt(3) / 2
    # 200 * d^2 * sqrt(3) / 2 ~ 1 => d ~ 0.075
    d = 0.075
    hex_points = []
    # Range of j
    y_max = 0.5
    y_min = -0.5
    # y = j * d * sqrt(3)/2
    # j range: -y_max / (d*sqrt(3)/2) to y_max / (d*sqrt(3)/2)
    j_max = int(np.ceil(y_max / (d * np.sqrt(3) / 2)))
    j_min = int(np.floor(y_min / (d * np.sqrt(3) / 2)))
    
    for j in range(j_min, j_max + 1):
        y = j * d * np.sqrt(3) / 2
        if y < y_min or y > y_max:
            continue
        # x = i * d + (j % 2) * d / 2
        # i range: -x_max / d to x_max / d
        x_offset = (j % 2) * d / 2
        i_max = int(np.ceil((0.5 - x_offset) / d))
        i_min = int(np.floor((-0.5 - x_offset) / d))
        for i in range(i_min, i_max + 1):
            x = i * d + x_offset
            if x < -0.5 or x > 0.5:
                continue
            hex_points.append([x, y])
            
    hex_points = np.array(hex_points)
    if len(hex_points) > N:
        # Select N points. Keep the ones closest to the center? 
        # Or just the first N? 
        # To maintain symmetry, we should select points that are well-distributed.
        # A simple heuristic: sort by distance to center, take N? No, that clusters in center.
        # Better: Take points in a grid-like order.
        # Let's just take the first N. It might be biased to one corner.
        # Let's center the lattice.
        # The lattice generation above is roughly centered if j_min, j_max are symmetric.
        # Let's just use the first N.
        configs.append(hex_points[:N])
    elif len(hex_points) < N:
        # Add random points in the free space?
        # For now, just use what we have and pad with random points in the center?
        # This might be bad. Let's skip this config if it's too small.
        # Or pad with points from a grid.
        # Let's just use the hex points we have and add random points.
        current = hex_points
        remaining = N - len(current)
        new_points = np.random.uniform(-0.5, 0.5, size=(remaining, 2))
        configs.append(np.vstack([current, new_points]))
    else:
        configs.append(hex_points)

    best_score = -1
    best_points = configs[0]
    
    for points in configs:
        # Ensure shape is (N, 2)
        if len(points) != N:
            continue
        # Run local search
        optimized = local_search(points)
        score = calculate_score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized
            
    return best_points
=======
import numpy as np

N = 200

def construct():
    best_points = None
    best_r = -1.0
    
    # Multiple initializations
    inits = []
    
    # 1. Grid 10x20
    xs = np.linspace(-0.5, 0.5, 20)
    ys = np.linspace(-0.5, 0.5, 10)
    grid = np.array([(x, y) for y in ys for x in xs])
    inits.append(grid)
    
    # 2. Hexagonal
    d = 0.075
    hex_points = []
    y_max = 0.5
    y_min = -0.5
    j_max = int(np.ceil(y_max / (d * np.sqrt(3) / 2)))
    j_min = int(np.floor(y_min / (d * np.sqrt(3) / 2)))
    for j in range(j_min, j_max + 1):
        y = j * d * np.sqrt(3) / 2
        if y < y_min or y > y_max:
            continue
        x_offset = (j % 2) * d / 2
        i_max = int(np.ceil((0.5 - x_offset) / d))
        i_min = int(np.floor((-0.5 - x_offset) / d))
        for i in range(i_min, i_max + 1):
            x = i * d + x_offset
            if x < -0.5 or x > 0.5:
                continue
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        inits.append(hex_points[:N])
        
    # 3. Random
    inits.append(np.random.uniform(-0.49, 0.49, size=(N, 2)))
    
    # 4. Jittered Grid
    grid_jitter = grid + np.random.normal(0, 0.005, size=(N, 2))
    grid_jitter[:, 0] = np.clip(grid_jitter[:, 0], -0.49, 0.49)
    grid_jitter[:, 1] = np.clip(grid_jitter[:, 1], -0.49, 0.49)
    inits.append(grid_jitter)
    
    for init_points in inits:
        if len(init_points) != N:
            continue
        # Run SA
        points = simulated_annealing(init_points, steps=10000, T_start=0.01, T_end=1e-6)
        # Polish with local search
        points = vectorized_local_search(points, iterations=50)
        r = calculate_score(points)
        if r > best_r:
            best_r = r
            best_points = points
            
    return best_points
>>>>>>> REPLACE

