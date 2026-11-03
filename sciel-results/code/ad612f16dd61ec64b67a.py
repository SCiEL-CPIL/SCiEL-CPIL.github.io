<<<<<<< SEARCH
    # Initial guess: Honeycomb lattice
    # 3 rows, 4 cols
    init_params = [0.0, 0.0, 6.0, 0.0] # Outer guess
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
            y = row * 1.5
            # Center the grid roughly
            x -= 2.6
            y -= 1.5
            # Add small random jitter to break symmetry
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)
            angle = 0.0
            init_params.extend([x, y, angle])
=======
    # Initial guess: Honeycomb lattice
    # 3 rows, 4 cols
    init_params = [0.0, 0.0, 6.0, 0.0] # Outer guess
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
            y = row * 1.5
            # Center the grid roughly
            x -= 2.6
            y -= 1.5
            # Add small random jitter to break symmetry
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)
            angle = 0.0
            init_params.extend([x, y, angle])
>>>>>>> REPLACE

