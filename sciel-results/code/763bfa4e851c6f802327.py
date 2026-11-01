<<<<<<< SEARCH
def _grid_init(side_x, side_y, n):
    """Create a grid-based initialization."""
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            x = (i + 0.5) / side_x
            y = (j + 0.5) / side_y
            r = 0.5 / max(side_x, side_y) * 0.95
            cells.append([x, y, r])
    
    # Shuffle and take first n
    np.random.shuffle(cells)
    cells = cells[:n]
    
    params = np.array([v for c in cells for v in c])
    return params
=======
def _grid_init(side_x, side_y, n):
    """Create a grid-based initialization."""
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            x = (i + 0.5) / side_x
            y = (j + 0.5) / side_y
            r = 0.5 / max(side_x, side_y) * 0.95
            cells.append([x, y, r])
    
    # Shuffle and take first n
    idx = np.random.permutation(len(cells))[:n]
    cells = [cells[i] for i in idx]
    
    params = np.array([v for c in cells for v in c])
    return params
>>>>>>> REPLACE

