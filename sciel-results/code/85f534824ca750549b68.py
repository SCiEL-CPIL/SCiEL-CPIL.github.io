def _initial_configuration():
    """Create a jittered grid initial configuration."""
    np.random.seed(42)
    positions = []
    # 5x5x2 grid
    for i in range(5):
        for j in range(5):
            x = -0.4 + i * 0.2
            y = -0.4 + j * 0.2
            z = -0.25
            # Add jitter
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            z += np.random.uniform(-0.02, 0.02)
            positions.append([x, y, z])
    for i in range(5):
        for j in range(5):
            x = -0.3 + i * 0.2
            y = -0.3 + j * 0.2
            z = 0.25
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            z += np.random.uniform(-0.02, 0.02)
            positions.append([x, y, z])
    
    return np.array(positions[:N])
