import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import laplacian
import time

N = 12

def construct():
    # 1. Create a graph for spectral embedding
    # 3x4 grid
    rows, cols = 3, 4
    n_nodes = rows * cols
    A = lil_matrix((n_nodes, n_nodes))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Connect to right
            if c < cols - 1:
                A[idx, idx + 1] = 1
                A[idx + 1, idx] = 1
            # Connect to bottom
            if r < rows - 1:
                A[idx, idx + cols] = 1
                A[idx + cols, idx] = 1
    
    L = laplacian(A)
    eigvals, eigvecs = np.linalg.eigh(L.toarray())
    
    # Use the first two non-zero eigenvectors for x and y
    # The first eigenvector is constant (0 eigenvalue), so take index 1 and 2
    x_init = eigvecs[:, 1]
    y_init = eigvecs[:, 2]
    
    # Normalize and scale
    x_init = (x_init - np.mean(x_init)) * 4.0
    y_init = (y_init - np.mean(y_init)) * 3.0
    
    # Initial guess for positions
    pos_init = np.column_stack((x_init, y_init))
    
    # Define objective function
    # Minimize the required side length s
    # s is determined by the max projection of vertices onto container normals
    # Container normals for flat-top hexagon (vertices at 0, 60, ...)
    # Normals point to the midpoints of the edges?
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Edges are between these vertices.
    # Normal to edge between v0 and v1 (midpoint angle 30 deg):
    # Angle 30, 90, 150, 210, 270, 330.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Hexagon vertices for a unit hexagon at origin, 0 deg rotation
    # Vertices at 0, 60, 120, 180, 240, 300
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    def objective(params):
        pos = params.reshape((N, 2))
        # Check disjointness
        # Distance between all pairs must be >= 2
        # If not, add penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 2.0:
                    penalty += (2.0 - d) ** 2 * 10.0 # Strong penalty
        
        # Calculate required s
        max_proj = 0.0
        for i in range(N):
            # Vertices of this hexagon
            verts = pos[i] + hex_verts
            for k in range(6):
                for m in range(6):
                    proj = np.dot(verts[k], normals[m])
                    if proj > max_proj:
                        max_proj = proj
        
        # s = max_proj / (sqrt(3)/2)
        s = max_proj / (np.sqrt(3) / 2.0)
        
        return s + penalty

    # Optimize
    start_time = time.time()
    try:
        res = minimize(objective, pos_init.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success or (time.time() - start_time < 60):
            pos_opt = res.x.reshape((N, 2))
        else:
            pos_opt = pos_init
    except:
        pos_opt = pos_init
        
    # Calculate final s
    max_proj = 0.0
    for i in range(N):
        verts = pos_opt[i] + hex_verts
        for k in range(6):
            for m in range(6):
                proj = np.dot(verts[k], normals[m])
                if proj > max_proj:
                    max_proj = proj
    s_final = max_proj / (np.sqrt(3) / 2.0)
    
    # Ensure disjointness? If penalty was high, it might still overlap.
    # For a robust solution, we might need to return a valid packing.
    # If the optimization failed to find a valid packing, fall back to a known good one?
    # The current program returns a valid packing with s=5.
    # Let's check if pos_opt is valid.
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pos_opt[i] - pos_opt[j]) < 2.0 - 1e-6:
                valid = False
    if not valid:
        # Fall back to the 3x4 grid with s=5
        sqrt3 = np.sqrt(3)
        inner = []
        for row in range(3):
            y = (row - 1) * sqrt3
            for col in range(4):
                x = (col - 1.5) * 2.0
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0
        
    inner = []
    for i in range(N):
        inner.append([pos_opt[i, 0], pos_opt[i, 1], 0.0])
        
    return inner, (0.0, 0.0), s_final, 0.0

# EVOLVE-BLOCK-END
