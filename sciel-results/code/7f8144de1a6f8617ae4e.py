def _box_area(positions):
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_extents = positions[:, 0] + (i_flat + 1) / N
    y_extents = positions[:, 1] + (j_flat + 1) / N
    
    X0 = np.min(positions[:, 0])
    X1 = np.max(x_extents)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(y_extents)
    
    return (X1 - X0) * (Y1 - Y0)

def _objective(x_flat):
    positions = x_flat.reshape(64, 2)
    area = _box_area(positions)
    if area < 1e-6:
        return 1e10 # Penalty
        
    nodes, weights = gauss_legendre(16)
    vol = _compute_volume_quad(positions, nodes, weights)
    return vol
