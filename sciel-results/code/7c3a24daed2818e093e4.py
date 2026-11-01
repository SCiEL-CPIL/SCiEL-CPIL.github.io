import numpy as np
import time
from scipy.optimize import minimize

N = 55

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Generate Icosahedron vertices (unit radius)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])

    # 2. Generate Edge Midpoints (unit radius relative to ico_verts)
    # Find edges: pairs with minimum distance
    dists = np.linalg.norm(ico_verts[:, np.newaxis] - ico_verts[np.newaxis, :], axis=2)
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-8:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                # Normalize midpoint to unit radius? 
                # The midpoints of a unit icosahedron do NOT lie on the unit sphere.
                # They lie on a smaller sphere. 
                # We want the 30 atoms to be at radius r2.
                # So we calculate the direction vector and normalize it.
                mid_dir = mid / np.linalg.norm(mid)
                edges.append(mid_dir)
    edge_dirs = np.array(edges)

    # 3. Construct initial positions
    # Shell 1: 12 atoms at r1
    # Shell 2: 30 atoms at r2
    # Shell 3: 12 atoms at r3 (same directions as Shell 1)
    # Center: 1 atom at 0
    
    r1 = 1.06
    r2 = 1.45
    r3 = 1.80

    pos1 = r1 * ico_verts
    pos2 = r2 * edge_dirs
    pos3 = r3 * ico_verts
    pos0 = np.array([[0.0, 0.0, 0.0]])

    positions = np.vstack([pos0, pos1, pos2, pos3])
    
    # Check count
    assert len(positions) == 55, f"Expected 55, got {len(positions)}"

    # 4. Optimization
    def energy(x):
        x = x.reshape(-1, 3)
        # Calculate pairwise distances
        diff = x[:, np.newaxis] - x[np.newaxis, :]
        r2 = np.sum(diff ** 2, axis=2)
        # Avoid division by zero
        r2[r2 == 0] = 1e-10
        r = np.sqrt(r2)
        # Mask out self-interactions
        mask = ~np.eye(N, dtype=bool)
        r_flat = r[mask]
        # LJ Energy: 4 * (r^-12 - r^-6)
        inv_r6 = r_flat ** -6
        e = 4.0 * (inv_r6 ** 2 - inv_r6)
        return np.sum(e)

    def gradient(x):
        x = x.reshape(-1, 3)
        diff = x[:, np.newaxis] - x[np.newaxis, :]
        r2 = np.sum(diff ** 2, axis=2)
        r2[r2 == 0] = 1e-10
        r = np.sqrt(r2)
        
        # Gradient of 4*(r^-12 - r^-6) w.r.t r is 4*(-12*r^-13 + 6*r^-7)
        # dE/dx_i = sum_j (dE/dr_ij * dr_ij/dx_i)
        # dr_ij/dx_i = (x_i - x_j) / r_ij
        # So term is 4 * (-12/r^13 + 6/r^7) * (x_i - x_j) / r
        # = 4 * (-12/r^14 + 6/r^8) * (x_i - x_j)
        
        coeff = 4.0 * (-12.0 * r ** -14 + 6.0 * r ** -8)
        # coeff is (N, N)
        # Set diagonal to 0
        np.fill_diagonal(coeff, 0.0)
        
        # grad_i = sum_j coeff[i, j] * (x_i - x_j)
        # This can be computed as:
        # grad = coeff * diff  (element-wise)
        # Then sum over j for each i
        grad = np.sum(coeff * diff, axis=1)
        
        return grad

    # Use L-BFGS-B
    result = minimize(
        energy, 
        positions.flatten(), 
        jac=gradient, 
        method='L-BFGS-B', 
        options={'maxiter': 1000, 'gtol': 1e-8}
    )

    best_pos = result.x.reshape(-1, 3)
    
    # Check time
    if time.time() > deadline:
        return best_pos

    # Optional: A second polish pass with tighter tolerance if time permits
    if time.time() < deadline - 10:
        result2 = minimize(
            energy, 
            best_pos.flatten(), 
            jac=gradient, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'gtol': 1e-10}
        )
        if result2.fun < result.fun:
            best_pos = result2.x.reshape(-1, 3)

    return best_pos
