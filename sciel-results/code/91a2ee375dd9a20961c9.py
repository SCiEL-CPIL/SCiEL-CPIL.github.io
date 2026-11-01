import numpy as np
import time
import scipy.optimize as so

N = 55
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _lj_energy_and_grad(x):
    """
    Calculate Lennard-Jones energy and gradient.
    x: (N, 3) array of coordinates
    returns: energy (scalar), grad (N, 3) array
    """
    # x is (N, 3)
    # Calculate distance matrix
    # d_ij = x_i - x_j
    # r_ij = ||d_ij||
    
    # To save memory and time for N=55, we can just loop or use broadcasting.
    # Broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    r = np.sqrt(r2)
    r6 = r**6
    r12 = r6**2
    
    # Energy term for pair (i,j): 4 * (r^-12 - r^-6)
    # Sum over all i,j / 2
    E_pair = 4.0 * (1.0/r12 - 1.0/r6)
    np.fill_diagonal(E_pair, 0.0)
    energy = 0.5 * np.sum(E_pair)
    
    # Gradient
    # dE/dx_i = sum_{j != i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j) / r
    # = sum_{j != i} 4 * (-12/r^13 + 6/r^7) * (x_i - x_j)
    # Let's compute the factor F_ij = 4 * (-12 * r^-13 + 6 * r^-7)
    # Note: r^-13 = r^-12 / r, r^-7 = r^-6 / r
    
    # Vectorized gradient calculation
    # We need sum over j for each i.
    # Factor shape (N, N)
    # r is (N, N)
    # inv_r = 1/r
    # inv_r6 = 1/r6
    # inv_r12 = 1/r12
    
    # Term: 4 * ( -12 * inv_r12 * inv_r + 6 * inv_r6 * inv_r )
    # = 4 * inv_r * ( -12 * inv_r12 + 6 * inv_r6 )
    
    inv_r = 1.0 / r
    factor = 4.0 * inv_r * (-12.0 * (1.0/r12) + 6.0 * (1.0/r6))
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (x_i - x_j)
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    # We want to sum over j (axis 1)
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    # Using einsum: 'ij,ijk->ik'
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return energy, grad

def _construct_mackay_icosahedron():
    """
    Construct the coordinates of a Mackay icosahedron (55 atoms).
    Returns (N, 3) array.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of a regular icosahedron centered at origin
    # Coordinates are permutations of (0, 1, phi), (1, phi, 0), (phi, 0, 1) with even number of minus signs
    verts = []
    # (0, 1, phi)
    verts.append([0, 1, phi])
    verts.append([0, -1, phi])
    verts.append([0, 1, -phi])
    verts.append([0, -1, -phi])
    # (1, phi, 0)
    verts.append([1, phi, 0])
    verts.append([-1, phi, 0])
    verts.append([1, -phi, 0])
    verts.append([-1, -phi, 0])
    # (phi, 0, 1)
    verts.append([phi, 0, 1])
    verts.append([phi, 0, -1])
    verts.append([-phi, 0, 1])
    verts.append([-phi, 0, -1])
    
    verts = np.array(verts, dtype=float)
    
    # Normalize vertices to radius 1
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Shell 1: Center + 12 vertices
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = np.vstack([center, verts])
    
    # Shell 2: 30 edge midpoints + 12 face centers
    # Edge midpoints: Midpoints of all edges of the icosahedron
    # An icosahedron has 30 edges.
    # We can find edges by checking distances between vertices.
    # In a regular icosahedron of radius 1, edge length is 2*sin(pi/5) ? No.
    # Distance between adjacent vertices.
    # Let's just compute all pairwise distances and pick the smallest 12 for each vertex (which are the neighbors).
    
    # Actually, easier way:
    # The icosahedron vertices are defined.
    # Edges connect vertices that are nearest neighbors.
    
    # Calculate distances between all pairs of the 12 vertices
    d_mat = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    # The minimum non-zero distance is the edge length
    min_dist = np.min(d_mat[d_mat > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.abs(d_mat[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
                
    # Generate midpoints
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        midpoints.append(mid)
        
    # Face centers: Centroids of the 20 faces
    # Faces are triangles. We can find them by looking for triplets (i, j, k) that are mutually connected.
    # Or simply: The face centers of an icosahedron correspond to the vertices of a dodecahedron.
    # But easier: iterate over all triplets and check if they form a face.
    # A face is a set of 3 vertices such that all 3 pairwise distances are equal to min_dist.
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.abs(d_mat[i, j] - min_dist) < 1e-6:
                for k in range(j+1, 12):
                    if np.abs(d_mat[i, k] - min_dist) < 1e-6 and np.abs(d_mat[j, k] - min_dist) < 1e-6:
                        # Found a face (i, j, k)
                        face_center = (verts[i] + verts[j] + verts[k]) / 3.0
                        faces.append(face_center)
                        
    # We should have 30 midpoints and 20 face centers?
    # Wait, Mackay icosahedron shell 2 has 42 atoms.
    # 30 edge midpoints + 12 face centers?
    # Let's check the Mackay icosahedron structure.
    # Shell 1: 13 atoms.
    # Shell 2: 42 atoms.
    # The 42 atoms are located at:
    # - 30 positions corresponding to the edges of the icosahedron (midpoints of edges? No, slightly off?)
    # - 12 positions corresponding to the faces (centroids? No, slightly off?)
    # Actually, the "Mackay" icosahedron is formed by stacking icosahedra.
    # The second shell consists of 30 atoms at the edge midpoints of the icosahedron and 12 atoms at the face centers?
    # Let's verify the count: 30 + 12 = 42. Yes.
    # However, the exact positions in the relaxed LJ cluster are NOT exactly the geometric midpoints/centroids.
    # They are displaced slightly. But starting from the geometric positions is a very good initialization.
    
    # Let's combine
    shell2 = np.array(midpoints + faces)
    
    # Total 55 atoms
    coords = np.vstack([shell1, shell2])
    
    # The current coordinates are based on an icosahedron of radius 1.
    # The nearest neighbor distance in this geometric construction is min_dist.
    # We want the nearest neighbor distance to be approx 1.122 (LJ minimum).
    # So we scale the whole structure by factor = 1.122 / min_dist
    
    scale_factor = 1.122 / min_dist
    coords = coords * scale_factor
    
    return coords

def construct():
    best_x = None
    best_E = np.inf
    
    # 1. Try Mackay Icosahedron seed
    try:
        x0 = _construct_mackay_icosahedron()
        
        # Relax
        res = so.minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        
        if res.success or res.fun < best_E:
            best_x = res.x
            best_E = res.fun
            
        # 2. Try perturbed Mackay Icosahedron to ensure we are in the global basin
        # Add small random noise
        np.random.seed(42)
        for i in range(3):
            x_pert = x0 + np.random.normal(0, 0.01, x0.shape)
            res = so.minimize(_lj_energy_and_grad, x_pert, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            if res.fun < best_E:
                best_x = res.x
                best_E = res.fun
                
        # Check time
        if time.time() - START_TIME > TIME_LIMIT * 0.6:
            return best_x
            
    except Exception as e:
        pass

    # 3. Backup: Random initialization or FCC fragment
    # If Mackay failed or we have time, try a few random starts
    np.random.seed(0)
    for i in range(5):
        if time.time() - START_TIME > TIME_LIMIT * 0.8:
            break
            
        # Random cluster in a box
        # Ensure min distance 0.5
        # Simple rejection sampling for 55 points in a box of size ~5x5x5
        # Or just use a scaled random Gaussian
        x_rand = np.random.normal(0, 2.0, (N, 3))
        
        # Rough check for overlaps (optional, L-BFGS might handle it if we start far enough, but LJ is steep)
        # Let's just run it. If it fails, we skip.
        try:
            res = so.minimize(_lj_energy_and_grad, x_rand, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
            if res.fun < best_E:
                best_x = res.x
                best_E = res.fun
        except:
            pass

    if best_x is None:
        # Fallback to seed program output
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_x = np.array(points[:N], dtype=float) * 1.12

    return best_x
