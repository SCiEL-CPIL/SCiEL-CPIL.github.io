<<<<<<< SEARCH
def _generate_fcc_38():
    """
    Generate the 38 closest points in an FCC lattice.
    FCC points are integer coordinates (i,j,k) with i+j+k even.
    We generate points within a bounding box and sort by distance.
    """
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    # Sort by distance squared
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the first 38
    selected = pts[:38, :3]
    
    # Center
    selected -= np.mean(selected, axis=0)
    
    # Scale to standard LJ bond length ~ 2^(1/6)
    # Calculate average nearest neighbor distance
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
        
    return selected
=======
def _generate_fcc_38():
    """
    Generate the 38 closest points in an FCC lattice.
    FCC points are integer coordinates (i,j,k) with i+j+k even.
    We generate points within a bounding box and sort by distance.
    """
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    # Sort by distance squared
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the first 38
    selected = pts[:38, :3]
    
    # Center
    selected -= np.mean(selected, axis=0)
    
    # Scale to standard LJ bond length ~ 2^(1/6)
    # Calculate average nearest neighbor distance
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
        
    return selected

def _make_truncated_octahedron_start():
    """
    Construct a truncated octahedron starting geometry for LJ38.
    
    The truncated octahedron is the global minimum for LJ38.
    It has:
    - 1 center atom
    - 12 atoms at vertices of a cube (±a, ±a, ±a) type positions
    - 12 atoms at edge midpoints
    - 13 atoms on the outer shell
    
    Actually, the standard LJ38 truncated octahedron structure:
    - Center: 1
    - First shell (12): FCC nearest neighbors
    - Second shell (12): FCC next-nearest neighbors  
    - Third shell (13): outer atoms
    
    Let's construct this properly using FCC lattice vectors.
    """
    a = 2.0 ** (1.0/6.0)  # LJ equilibrium distance
    
    # FCC primitive vectors
    a1 = np.array([0.5, 0.5, 0.0]) * a
    a2 = np.array([0.5, 0.0, 0.5]) * a
    a3 = np.array([0.0, 0.5, 0.5]) * a
    
    # Generate FCC lattice points
    pts = [np.zeros(3)]  # Center
    
    # Generate all FCC points within a radius
    max_coord = 3
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    # FCC: (i,j,k) with i+j+k even, scaled by a/2
                    pt = np.array([i, j, k]) * (a / 2.0)
                    fcc_points.append(pt)
    
    fcc_points = np.array(fcc_points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_points, axis=1)
    order = np.argsort(dists)
    fcc_points = fcc_points[order]
    
    # Take 37 more points (total 38 with center)
    selected = np.vstack([np.zeros(3), fcc_points[:37]])
    
    return selected
>>>>>>> REPLACE

