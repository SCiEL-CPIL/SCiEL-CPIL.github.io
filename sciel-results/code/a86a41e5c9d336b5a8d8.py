import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    # Build a spectral-graph embedding for the 12 unit hexagons, then refine
    # positions and outer-container orientation with Nelder-Mead to minimise
    # the required outer hexagon side length.
    
    # --- 1. Define a reasonable adjacency graph for 12 hexagons ---
    # Use a 3x4 grid topology with hexagonal-lattice connectivity.
    # Nodes indexed row-major: row 0 (top), row 1 (mid), row 2 (bottom)
    # Each row has 4 nodes.
    
    # Build adjacency: connect neighbours in hexagonal lattice sense.
    # For a flat-top hexagonal lattice with 3 rows of 4:
    # Row 0: nodes 0,1,2,3
    # Row 1: nodes 4,5,6,7
    # Row 2: nodes 8,9,10,11
    # Adjacencies (hex lattice):
    adj = np.zeros((12, 12))
    # Horizontal neighbours
    for r in range(3):
        for c in range(3):
            i = r * 4 + c
            j = r * 4 + c + 1
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    # Vertical / diagonal neighbours (hex lattice)
    # Row 0 to Row 1: each node connects to the two below
    for c in range(4):
        i = c          # row 0
        j1 = 4 + c     # row 1 same col
        j2 = 4 + c + 1 if c < 3 else 4 + c - 1  # row 1 adjacent
        adj[i, j1] = 1.0
        adj[j1, i] = 1.0
        if c < 3:
            adj[i, j2] = 0.5
            adj[j2, i] = 0.5
        else:
            adj[i, j2] = 0.5
            adj[j2, i] = 0.5
    # Row 1 to Row 2
    for c in range(4):
        i = 4 + c      # row 1
        j1 = 8 + c     # row 2 same col
        j2 = 8 + c + 1 if c < 3 else 8 + c - 1
        adj[i, j1] = 1.0
        adj[j1, i] = 1.0
        if c < 3:
            adj[i, j2] = 0.5
            adj[j2, i] = 0.5
        else:
            adj[i, j2] = 0.5
            adj[j2, i] = 0.5
    
    # --- 2. Spectral embedding: compute graph Laplacian eigenvectors ---
    L = np.diag(adj.sum(axis=1)) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    # Use the two smallest non-trivial eigenvectors (skip the constant one)
    # Sort by eigenvalue
    idx = np.argsort(eigvals)
    x_spec = eigvecs[idx[1]]
    y_spec = eigvecs[idx[2]]
    
    # Scale to approximate hexagonal lattice spacing
    # Hex lattice: horizontal spacing 2, vertical spacing sqrt(3)
    x_spec = x_spec - x_spec.mean()
    y_spec = y_spec - y_spec.mean()
    # Normalize and scale
    x_scale = 2.0 / max(1e-10, np.max(np.abs(x_spec)))
    y_scale = SQRT3 / max(1e-10, np.max(np.abs(y_spec)))
    x_init = x_spec * x_scale
    y_init = y_spec * y_scale
    
    # --- 3. Define the optimization objective ---
    # Parameters: 12 x-positions, 12 y-positions, container center (Cx, Cy), container rotation theta
    # Objective: minimize outer hexagon side length S, with penalty for overlaps
    
    def get_hex_vertices(cx, cy, angle_deg=0.0):
        """Return 6 vertices of unit hexagon centered at (cx,cy) with given rotation."""
        a = np.deg2rad(angle_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def compute_min_side(xs, ys, Cx, Cy, theta):
        """Compute minimum outer hexagon side length containing all inner hexagons."""
        # Container normals: 6 directions at theta + k*60 degrees
        n_angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
        
        max_proj = 0.0
        for i in range(12):
            verts = get_hex_vertices(xs[i], ys[i])
            rel = verts - np.array([Cx, Cy])
            projs = rel @ normals.T
            curr_max = np.max(projs)
            if curr_max > max_proj:
                max_proj = curr_max
        # For a regular hexagon with side s and flat-to-flat orientation,
        # the distance from center to each edge (apothem) is s * sqrt(3)/2
        # The max projection onto normals equals the apothem.
        # So s = 2 * max_proj / sqrt(3)
        return (2.0 / SQRT3) * max_proj
    
    def objective(params):
        xs = params[0:12]
        ys = params[12:24]
        Cx = params[24]
        Cy = params[25]
        theta = params[26]
        
        S = compute_min_side(xs, ys, Cx, Cy, theta)
        
        # Overlap penalty: minimum center-to-center distance for non-overlapping
        # unit hexagons is sqrt(3) (flat-to-flat touching)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < SQRT3:
                    penalty += (SQRT3 - d) ** 2
        
        return S + 50.0 * penalty
    
    # --- 4. Run Nelder-Mead optimization ---
    x0 = np.concatenate([x_init, y_init, [0.0, 0.0, 0.0]])
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    opt = res.x
    xs = opt[0:12]
    ys = opt[12:24]
    Cx = opt[24]
    Cy = opt[25]
    theta = opt[26]
    
    # --- 5. Build result ---
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], 0.0])
    
    side = compute_min_side(xs, ys, Cx, Cy, theta)
    
    return inner, (Cx, Cy), side, np.degrees(theta)
# EVOLVE-BLOCK-END
