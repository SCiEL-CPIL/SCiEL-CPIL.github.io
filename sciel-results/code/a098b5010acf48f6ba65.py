def _side_from(theta, P, c):
    U = _normals(theta)
    d = (P - c) @ U.T                     # (n,6)
    return (np.abs(d).max() + 1.0) * _T
