def _clamp_cartesian(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    # 1. y >= 0
    y = np.clip(y, 0, None)
    # 2. y <= sqrt(3)*x  => x >= y/sqrt(3)
    x = np.clip(x, y/np.sqrt(3), None)
    # 3. y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x = np.clip(x, None, 1 - y/np.sqrt(3))
    # Re-clip y in case x changes affect valid y range?
    # If x is clamped, y must be <= min(sqrt(3)*x, sqrt(3)*(1-x))
    y_max = np.minimum(np.sqrt(3)*x, np.sqrt(3)*(1-x))
    y = np.clip(y, 0, y_max)
    return np.column_stack((x, y))
