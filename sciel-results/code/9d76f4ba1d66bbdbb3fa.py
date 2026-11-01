import numpy as np
from scipy.optimize import minimize

N = 55

def lj_energy_and_grad(x):
    """
    x: array of shape (N, 3)
    Returns: energy, gradient
    """
    # x is flattened or reshaped? minimize passes flat array.
    # We will reshape inside.
    x = x.reshape(-1, 3)
    N = x.shape[0]
    
    # Compute all pairwise distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # LJ terms
    # V = 4 * (r^-12 - r^-6)
    # r^-6 = dist_sq^-3
    r6_inv = dist_sq ** (-1.5)
    r12_inv = r6_inv ** 2
    
    # Energy = 4 * sum (r12_inv - r6_inv) for i < j
    # Sum over all i,j includes i=j (which is 0 if we mask) and i!=j twice.
    # E = 2 * 4 * sum_{i<j} (r12_inv - r6_inv)
    # Sum over all i != j is 2 * sum_{i<j}.
    # So E = 4 * 0.5 * sum_{i!=j} (r12_inv - r6_inv) = 2 * sum_{i!=j} (r12_inv - r6_inv)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    energy = 2.0 * np.sum((r12_inv - r6_inv)[mask])
    
    # Gradient
    # dE/dx_i = sum_{j != i} dE/dr_ij * d r_ij / d x_i
    # E = sum_{k<l} 4(r_kl^-12 - r_kl^-6)
    # dE/dx_i = sum_{j != i} 4(-12 r_ij^-13 + 6 r_ij^-7) * (x_i - x_j)/r_ij
    # = sum_{j != i} 4(-12 r_ij^-14 + 6 r_ij^-8) * (x_i - x_j)
    # = sum_{j != i} 24(r_ij^-8 - 2 r_ij^-14) * (x_j - x_i)  <-- Note sign change if we define force
    # Let's stick to dE/dx_i.
    # term = 24 * (r8_inv - 2 * r14_inv)
    # grad_i = sum_{j != i} term_ij * (x_j - x_i)
    
    r8_inv = r6_inv * dist_sq ** (-1.0)
    r14_inv = r12_inv * dist_sq ** (-1.0)
    
    # factor_ij = 24 * (r8_inv - 2 * r14_inv)
    factor = 24.0 * (r8_inv - 2.0 * r14_inv)
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (x[j] - x[i])
    # grad[i] = sum_j factor[i,j] * x[j] - x[i] * sum_j factor[i,j]
    # sum_j factor[i,j] is row sum of factor
    row_sums = np.sum(factor, axis=1)
    
    # grad = factor.T @ x - row_sums[:, np.newaxis] * x
    # Wait, factor[i,j] multiplies (x[j] - x[i]).
    # sum_j factor[i,j] x[j] - x[i] sum_j factor[i,j]
    # This is correct.
    
    grad = (factor.T @ x) - row_sums[:, np.newaxis] * x
    
    return energy, grad.flatten()

def construct():
    import time
    start_time = time.time()
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of regular icosahedron
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize vertices to unit radius
    norm_verts = np.linalg.norm(verts, axis=1)
    verts = verts / norm_verts[:, np.newaxis]
    
    # Find edges (distance between vertices)
    # In a regular icosahedron, edges connect vertices with smallest distance.
    # There are 30 edges.
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            # The edge length is ~1.0515 if radius is 1.
            # We can just pick the 30 smallest distances, or check against a threshold.
            # Threshold: 1.05
            if d < 1.06:
                edges.append((i, j))
    
    # Find faces (triangles)
    # There are 20 faces.
    # A face is formed by 3 mutually connected vertices.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                # Check if i-j, j-k, k-i are edges
                if (i,j) in edges or (j,i) in edges:
                    if (j,k) in edges or (k,j) in edges:
                        if (k,i) in edges or (i,k) in edges:
                            faces.append((i, j, k))
    
    # Initial radii
    # LJ minimum r = 1.122.
    # Inner shell radius R1, Outer shell radius R2.
    # Typical Mackay icosahedron radii: R1 ~ 1.4, R2 ~ 2.0
    R1 = 1.4
    R2 = 2.0
    
    # Generate positions
    positions = []
    
    # 1. Center
    positions.append([0, 0, 0])
    
    # 2. 12 Inner Shell Vertices
    for v in verts:
        positions.append(v * R1)
        
    # 3. 30 Outer Shell Edge Midpoints
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        # The midpoint of the chord is not on the sphere of radius R2 in the same way.
        # The vector from center to midpoint has length cos(theta/2) * R1?
        # Actually, the midpoint vector is (v_i + v_j)/2.
        # Its length is |v_i + v_j|/2.
        # For unit vectors, |v_i + v_j|^2 = 1 + 1 + 2 cos(theta) = 2(1+cos(theta)).
        # cos(theta) for icosahedron edge: a = 2 sin(theta/2).
        # a = 1.0515. sin(theta/2) = 0.525. theta/2 = 31.7 deg. theta = 63.4 deg.
        # cos(63.4) = 0.447.
        # |v_i + v_j| = sqrt(2(1.447)) = sqrt(2.894) = 1.701.
        # Length of midpoint vector = 0.85.
        # We want this atom to be at radius R2.
        # So we scale the midpoint vector to length R2.
        mid_vec = verts[i] + verts[j]
        mid_len = np.linalg.norm(mid_vec)
        pos = (mid_vec / mid_len) * R2
        positions.append(pos)
        
    # 4. 12 Outer Shell Face Centers
    for i, j, k in faces:
        # Centroid of triangle
        cent = (verts[i] + verts[j] + verts[k]) / 3.0
        cent_vec = verts[i] + verts[j] + verts[k]
        cent_len = np.linalg.norm(cent_vec)
        pos = (cent_vec / cent_len) * R2
        positions.append(pos)
        
    positions = np.array(positions)
    
    # Ensure we have 55 atoms
    if positions.shape[0] != N:
        # Fallback to cubic if construction failed (should not happen)
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        positions = np.array(points[:N], dtype=float) * 1.12
        return positions

    # Relaxation
    # Use L-BFGS-B with analytical gradient
    # Time limit check
    if time.time() - start_time > 140:
        return positions
        
    res = minimize(
        lj_energy_and_grad,
        positions.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
    )
    
    if res.success:
        return res.x.reshape(-1, 3)
    else:
        return positions

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def lj_energy_and_grad(x):
    x = x.reshape(-1, 3)
    N = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    r6_inv = dist_sq ** (-1.5)
    r12_inv = r6_inv ** 2
    mask = ~np.eye(N, dtype=bool)
    energy = 2.0 * np.sum((r12_inv - r6_inv)[mask])
    r8_inv = r6_inv * dist_sq ** (-1.0)
    r14_inv = r12_inv * dist_sq ** (-1.0)
    factor = 24.0 * (r8_inv - 2.0 * r14_inv)
    np.fill_diagonal(factor, 0.0)
    row_sums = np.sum(factor, axis=1)
    grad = (factor.T @ x) - row_sums[:, np.newaxis] * x
    return energy, grad.flatten()

def construct():
    import time
    start_time = time.time()
    
    phi = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norm_verts = np.linalg.norm(verts, axis=1)
    verts = verts / norm_verts[:, np.newaxis]
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.06:
                edges.append((i, j))
                
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (i,j) in edges or (j,i) in edges:
                    if (j,k) in edges or (k,j) in edges:
                        if (k,i) in edges or (i,k) in edges:
                            faces.append((i, j, k))
                            
    R1 = 1.4
    R2 = 2.0
    positions = []
    positions.append([0, 0, 0])
    for v in verts:
        positions.append(v * R1)
    for i, j in edges:
        mid_vec = verts[i] + verts[j]
        mid_len = np.linalg.norm(mid_vec)
        pos = (mid_vec / mid_len) * R2
        positions.append(pos)
    for i, j, k in faces:
        cent_vec = verts[i] + verts[j] + verts[k]
        cent_len = np.linalg.norm(cent_vec)
        pos = (cent_vec / cent_len) * R2
        positions.append(pos)
        
    positions = np.array(positions)
    
    if positions.shape[0] != N:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12

    if time.time() - start_time > 140:
        return positions
        
    res = minimize(
        lj_energy_and_grad,
        positions.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
    )
    
    if res.success:
        return res.x.reshape(-1, 3)
    else:
        return positions
# EVOLVE-BLOCK-END
