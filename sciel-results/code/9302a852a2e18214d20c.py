import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Define a graph for 12 hexagons in a hexagonal lattice pattern
    # 3 rows of 4: nodes 0-3 (row -1), 4-7 (row 0), 8-11 (row +1)
    # Adjacency: connect neighboring hexagons in the lattice
    # Row structure:
    #   Row -1 (y=-sqrt3): nodes 0,1,2,3 at x=-3,-1,1,3
    #   Row  0 (y=0):      nodes 4,5,6,7 at x=-3,-1,1,3  
    #   Row +1 (y=+sqrt3):  nodes 8,9,10,11 at x=-3,-1,1,3
    # 
    # For hexagonal packing with offset, neighbors are:
    # Same row: (i, i+1)
    # Adjacent rows with offset: each node connects to 2 nodes in row above/below
    
    # Build adjacency matrix (variable weights)
    A = np.zeros((N, N))
    # Same-row connections (weight 1.0)
    for row in range(3):
        for col in range(3):
            idx = row * 4 + col
            A[idx, idx+1] = 1.0
            A[idx+1, idx] = 1.0
    # Cross-row connections (weight 0.8 to allow some flexibility)
    # Row 0 to Row 1: each node connects to node below-left and below-right
    for col in range(4):
        idx0 = col  # row 0
        idx1 = 4 + col  # row 1
        A[idx0, idx1] = 0.8
        A[idx1, idx0] = 0.8
        if col > 0:
            idx1b = 4 + col - 1
            A[idx0, idx1b] = 0.8
            A[idx1b, idx0] = 0.8
        if col < 3:
            idx1c = 4 + col + 1
            A[idx0, idx1c] = 0.8
            A[idx1c, idx0] = 0.8
    # Row 1 to Row 2: similar
    for col in range(4):
        idx1 = 4 + col  # row 1
        idx2 = 8 + col  # row 2
        A[idx1, idx2] = 0.8
        A[idx2, idx1] = 0.8
        if col > 0:
            idx2b = 8 + col - 1
            A[idx1, idx2b] = 0.8
            A[idx2b, idx1] = 0.8
        if col < 3:
            idx2c = 8 + col + 1
            A[idx1, idx2c] = 0.8
            A[idx2c, idx1] = 0.8
    
    # Compute graph Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Get eigenvectors (skip trivial first one)
    eigvals, eigvecs = np.linalg.eigh(L)
    # Use 2nd and 3rd eigenvectors for x and y coordinates (Fiedler vectors)
    # Normalize to have zero mean and unit norm
    x_init = eigvecs[:, 1]
    y_init = eigvecs[:, 2]
    
    # Scale to reasonable size
    x_init = x_init / np.linalg.norm(x_init) * 3.0
    y_init = y_init / np.linalg.norm(y_init) * 2.0
    
    # Initial positions
    positions = np.column_stack([x_init, y_init])
    
    # Define the objective function: minimize container side length
    def container_side_length(params):
        # params: [x0,y0, ..., x11,y11, angle0, ..., angle11, cx, cy, s, rot]
        # But we'll optimize positions and angles, then compute minimal s
        pos = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N] * np.pi / 180.0
        
        # Compute vertices of each inner hexagon
        all_vertices = []
        for i in range(N):
            cx, cy = pos[i]
            a = angles[i]
            # Unit hexagon vertices at angles a, a+60, ..., a+300
            for k in range(6):
                theta = a + k * np.pi / 3.0
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                all_vertices.append([vx, vy])
        all_vertices = np.array(all_vertices)
        
        # For a regular hexagon container of side s centered at origin with rotation rot,
        # the constraint is that all vertices must satisfy the 6 half-plane inequalities
        # For flat-top hexagon (rot=0): 
        #   |x| <= s
        #   |y| * 2/sqrt(3) + |x| / s <= 1 ... actually let me use the standard form
        # 
        # Regular hexagon with vertices at angles rot, rot+60, ..., rot+300, side s
        # The 6 edges define half-planes. For a hexagon centered at origin with 
        # circumradius R = s (for regular hexagon, side = circumradius):
        # Vertices at R*(cos(rot+k*60), sin(rot+k*60))
        # Edge normals point outward at angles rot+30, rot+90, ...
        # Constraint: dot(vertex, normal_k) <= R * cos(30) = R * sqrt(3)/2 for all k
        # Actually for regular hexagon with side s, circumradius = s
        # The apothem (inradius) = s * sqrt(3)/2
        # So constraint: for each edge normal direction, dot(v, n) <= s * sqrt(3)/2
        
        # We need to find the minimum s such that all vertices fit
        # For a given container orientation (let's try rot=0 first), 
        # the 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
        # (for flat-top hexagon with vertices at 0, 60, 120, 180, 240, 300)
        
        # Actually, let me parameterize: container has center (0,0) for now, 
        # rotation angle rot, side length s
        # Vertices of container: s*(cos(rot+k*60), sin(rot+k*60)) for k=0..5
        # Edge normals: perpendicular to edges, pointing outward
        # Edge between vertex k and k+1 has normal at angle rot + (k+0.5)*60
        
        # For minimum s, we need: for each edge normal direction,
        # max over all inner vertices of dot(v, n) <= s * apothem_factor
        # where apothem = s * sqrt(3)/2
        
        # So s >= max_k (max_v dot(v, n_k)) / (sqrt(3)/2)
        
        # Let's try rot = 0 (flat-top)
        rot = 0.0
        normals = []
        for k in range(6):
            angle = rot + (k + 0.5) * np.pi / 3.0
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # For each normal, compute max projection
        projections = all_vertices @ normals.T  # (72, 6)
        max_proj = np.max(projections, axis=0)  # (6,)
        
        # s must satisfy: max_proj[k] <= s * sqrt(3)/2 for all k
        # So s >= max(max_proj) / (sqrt(3)/2)
        s_required = np.max(max_proj) / (sqrt3 / 2.0)
        
        # Also need to ensure no overlaps between inner hexagons
        # Simple check: distance between centers >= 2 * sqrt(3)/2 = sqrt(3) for non-overlapping
        # (this is a necessary but not sufficient condition)
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_dist = min(min_dist, d)
        
        # Penalty if hexagons overlap (distance < sqrt(3) means potential overlap)
        overlap_penalty = 0.0
        if min_dist < sqrt3:
            overlap_penalty = (sqrt3 - min_dist) * 100.0
        
        return s_required + overlap_penalty
    
    # Initial parameters: positions (24) + angles (12)
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[2*i] = positions[i, 0]
        init_params[2*i+1] = positions[i, 1]
        init_params[2*N+i] = 0.0  # initial angle
    
    # Optimize using Nelder-Mead
    result = minimize(container_side_length, init_params, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    opt_params = result.x
    pos = opt_params[:2*N].reshape(N, 2)
    angles_deg = opt_params[2*N:3*N]
    
    # Compute final container side length
    all_vertices = []
    for i in range(N):
        cx, cy = pos[i]
        a = angles_deg[i] * np.pi / 180.0
        for k in range(6):
            theta = a + k * np.pi / 3.0
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            all_vertices.append([vx, vy])
    all_vertices = np.array(all_vertices)
    
    rot = 0.0
    normals = []
    for k in range(6):
        angle = rot + (k + 0.5) * np.pi / 3.0
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    projections = all_vertices @ normals.T
    max_proj = np.max(projections, axis=0)
    s = np.max(max_proj) / (sqrt3 / 2.0)
    
    # Build inner hexagon list
    inner = []
    for i in range(N):
        inner.append([pos[i, 0], pos[i, 1], angles_deg[i]])
    
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
