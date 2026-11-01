def get_hex_width(theta):
    # Width of a unit hexagon (side 1, angle 0) in direction theta
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # x = cos, y = sin
    vs = np.array([np.cos(np.pi/3 * np.arange(6)), np.sin(np.pi/3 * np.arange(6))]).T
    u = np.array([np.cos(theta), np.sin(theta)])
    proj = vs @ u
    return np.max(proj) - np.min(proj)
