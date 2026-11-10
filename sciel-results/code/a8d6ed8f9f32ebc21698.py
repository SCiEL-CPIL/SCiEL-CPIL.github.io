    k_points = []
    # Path 1: Gamma (0,0) to X (pi,0)
    for t in np.linspace(0, 1, 10):
        k_points.append((t * np.pi, 0))
    # Path 2: X (pi,0) to M' (pi/2, pi/2)
    for t in np.linspace(0, 1, 10):
        k_points.append((np.pi * (1 - 0.5*t), 0.5 * t * np.pi))
    # Path 3: M' (pi/2, pi/2) to Gamma (0,0)
    for t in np.linspace(0, 1, 10):
        k_points.append((0.5 * np.pi * (1-t), 0.5 * np.pi * (1-t)))
