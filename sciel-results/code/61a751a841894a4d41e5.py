<<<<<<< SEARCH
def _initial_grid():
    """FCC-inspired grid initialization."""
    # Use a 4x4x4 grid = 64 positions, pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle deterministically and pick first N
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()
=======
def _initial_grid():
    """FCC-inspired grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_bcc():
    """BCC-inspired initialization: corner + face-center + body-center layers."""
    centers = np.zeros((N, 3))
    # BCC: 8 corners + 6 face centers + 1 body center per cell
    # Use a 3x3x3 supercell of BCC-like points, pick 50
    pts = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                x = (ix + 0.5) / 3.0 - 0.5
                y = (iy + 0.5) / 3.0 - 0.5
                z = (iz + 0.5) / 3.0 - 0.5
                pts.append([x, y, z])
                # Add body-center offset
                ox, oy, oz = 0.5/3, 0.5/3, 0.5/3
                nx, ny, nz = x + ox, y + oy, z + oz
                if abs(nx) < 0.5 and abs(ny) < 0.5 and abs(nz) < 0.5:
                    pts.append([nx, ny, nz])
    pts = np.array(pts)
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(pts))
    pts = pts[perm[:N]]
    # Clip to cube
    np.clip(pts, -0.49, 0.49, out=pts)
    return pts
>>>>>>> REPLACE

