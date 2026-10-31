"""
Strategy: Distance Matrix Optimization with Gram Matrix Projection.
Rationale: The standard Cartesian coordinate search often gets trapped in the icosahedral 
funnel for LJ38. By optimizing the pairwise distance matrix D directly, we can use the 
geometry of the distance space to navigate towards the FCC global minimum. 
The constraint that D corresponds to a valid Euclidean configuration is that the Gram 
matrix G = -1/2 * J * D^2 * J is Positive Semi-Definite (PSD). 
We use Projected Gradient Descent (PGD) on D:
1. Compute gradient of LJ energy w.r.t. D.
2. Update D.
3. Project D to the nearest valid distance matrix (by repairing negative eigenvalues of G).
This approach allows "non-Euclidean" intermediate steps that can cross energy barriers 
that are difficult to cross in Cartesian space, while always projecting back to a valid 
3D geometry.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin before 150s kill

def lj_energy_from_d(D):
    """Compute LJ energy from distance matrix D."""
    # D is (N, N), lower triangle holds distances
    # We only need unique pairs i < j
    iu = np.triu_indices(N, k=1)
    r = D[iu]
    # Avoid division by zero (should not happen if D is valid, but safety)
    r = np.maximum(r, 1e-6)
    inv_r6 = r ** -6
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def lj_energy_grad_d(D):
    """Compute gradient of LJ energy w.r.t. distance matrix D."""
    iu = np.triu_indices(N, k=1)
    r = D[iu]
    r = np.maximum(r, 1e-6)
    
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7) = 24 * (r^-7 - 2*r^-13)
    # Note: E = 4 * (r^-12 - r^-6)
    # dE/dr = 4 * (-12*r^-13 - (-6)*r^-7) = 4 * (-12*r^-13 + 6*r^-7) = 24 * (r^-7 - 2*r^-13)
    grad_r = 24.0 * (r ** -7 - 2.0 * r ** -13)
    
    # Gradient w.r.t. D
    # D[i,j] = r, so dE/dD[i,j] = grad_r
    # Since D is symmetric, the gradient matrix G_D has G_D[i,j] = grad_r and G_D[j,i] = grad_r
    G_D = np.zeros((N, N))
    G_D[iu] = grad_r
    G_D.T[iu] = grad_r
    return G_D

def d_to_gram(D):
    """Convert distance matrix D to Gram matrix G."""
    # J is centering matrix
    # G = -1/2 * J * D^2 * J
    # Efficient way: 
    # Let M = D^2
    # G_ij = -1/2 * (M_ij - M_i. - M_.j + M_..)
    M = D ** 2
    M_i = np.sum(M, axis=1, keepdims=True)
    M_j = np.sum(M, axis=0, keepdims=True)
    M_all = np.sum(M)
    G = -0.5 * (M - M_i - M_j + M_all / N)
    return G

def gram_to_d(G):
    """Convert Gram matrix G to distance matrix D (projecting to PSD if needed)."""
    # Ensure G is symmetric
    G = (G + G.T) / 2.0
    
    # Eigenvalue decomposition
    try:
        w, v = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        # Fallback: return large distances to break symmetry?
        return np.full((N, N), 10.0)
    
    # Clip negative eigenvalues to 0 for projection
    w_clipped = np.maximum(w, 0.0)
    
    # Reconstruct G_psd
    G_psd = (v * w_clipped) @ v.T
    G_psd = (G_psd + G_psd.T) / 2.0
    
    # Compute D from G_psd
    # D^2 = -2 * J * G_psd * J
    # Or simpler: D_ij = sqrt(-G_ij) for i != j? No, G_ij is dot product of centered coords.
    # Wait, standard relation:
    # If G is Gram matrix of centered coords X_c (X_c @ X_c.T = G), then
    # ||x_i - x_j||^2 = G_ii + G_jj - 2 G_ij
    # So D_ij = sqrt(G_ii + G_jj - 2 G_ij)
    
    diag_G = np.diag(G_psd)
    D2 = diag_G[:, np.newaxis] + diag_G[np.newaxis, :] - 2.0 * G_psd
    
    # Numerical errors might make D2 slightly negative
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D

def pgd_search(D_init, steps=500, step_size=0.01, anneal=True):
    """
    Perform Projected Gradient Descent on the distance matrix.
    """
    D = D_init.copy()
    best_D = D.copy()
    best_E = lj_energy_from_d(D)
    
    # Initial step size
    alpha = step_size
    
    for i in range(steps):
        # Compute gradient
        grad_D = lj_energy_grad_d(D)
        
        # Update
        D_new = D - alpha * grad_D
        
        # Ensure D is symmetric and non-negative
        D_new = (D_new + D_new.T) / 2.0
        D_new = np.maximum(D_new, 1e-6)
        
        # Project to valid distance matrix (PSD Gram)
        G = d_to_gram(D_new)
        D_projected = gram_to_d(G)
        
        # Calculate energy
        E_new = lj_energy_from_d(D_projected)
        
        if E_new < best_E:
            best_E = E_new
            best_D = D_projected.copy()
            
        D = D_projected
        
        # Simple annealing
        if anneal:
            alpha *= 0.995
            
    return best_D, best_E

def d_to_coords(D):
    """Convert distance matrix D to Cartesian coordinates using MDS."""
    G = d_to_gram(D)
    # Ensure G is PSD for SVD
    G = (G + G.T) / 2.0
    w, v = np.linalg.eigh(G)
    w_clipped = np.maximum(w, 0.0)
    G_psd = (v * w_clipped) @ v.T
    G_psd = (G_psd + G_psd.T) / 2.0
    
    # MDS: X = V * sqrt(W)
    # We only need the top 3 dimensions
    idx = np.argsort(w_clipped)[::-1][:3]
    V = v[:, idx]
    W = w_clipped[idx]
    coords = V * np.sqrt(W)
    
    # Center
    coords -= np.mean(coords, axis=0)
    return coords

def local_minimize_cartesian(positions, maxiter=500, gtol=1e-8):
    """
    Perform a local L-BFGS minimization in Cartesian space.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, 1.0)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
        return energy
    
    try:
        result = minimize(
            energy_func,
            positions.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-12}
        )
        if result.success or result.fun < energy_func(positions):
            min_pos = result.x.reshape(N, 3)
            min_energy = energy_func(min_pos)
            return min_pos, min_energy
    except Exception:
        pass
    
    return positions, energy_func(positions)

def make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron points.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    if len(selected) < N:
        return None
        
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    return coords

def make_icosahedron():
    """
    Generate an icosahedral cluster.
    """
    pts = [[0, 0, 0]]
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ]
    scale = 1.0 / math.sqrt(2) 
    for v in verts:
        pts.append([v[0]*scale, v[1]*scale, v[2]*scale])
    
    rng = np.random.default_rng(123)
    for _ in range(25):
        z = rng.uniform(-1, 1)
        a = rng.uniform(0, 2 * math.pi)
        r = 1.7
        x = math.sqrt(1 - z*z) * math.cos(a)
        y = math.sqrt(1 - z*z) * math.sin(a)
        pts.append([x*r, y*r, z*r])
        
    coords = np.array(pts[:38], dtype=float)
    coords -= np.mean(coords, axis=0)
    return coords

def coords_to_d(coords):
    """Convert Cartesian coordinates to distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))
    return D

def construct():
    start_time = time.time()
    
    best_pos = None
    best_energy = float('inf')
    
    # 1. Generate initial candidates in Cartesian space
    candidates = []
    
    try:
        fcc_coords = make_fcc_truncated_octahedron()
        if fcc_coords is not None:
            candidates.append(fcc_coords)
    except Exception:
        pass
        
    try:
        ico_coords = make_icosahedron()
        if ico_coords is not None:
            candidates.append(ico_coords)
    except Exception:
        pass

    if not candidates:
        # Fallback random
        rng = np.random.default_rng(0)
        candidates.append(rng.uniform(-2, 2, size=(N, 3)))

    # 2. For each candidate, try Distance Matrix Optimization
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        D_init = coords_to_d(cand)
        
        # Run PGD on D
        # We use a moderate number of steps. PGD is slower per step than L-BFGS due to Eigh,
        # but it explores a different space.
        # 38 atoms -> 38x38 matrix. Eigh is fast.
        D_opt, E_opt = pgd_search(D_init, steps=200, step_size=0.02)
        
        # Convert back to Cartesian
        coords_opt = d_to_coords(D_opt)
        
        # Check validity (min distance)
        # Simple check
        diff = coords_opt[:, np.newaxis, :] - coords_opt[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, 1e10)
        if np.min(r2) >= 0.25:
            # Local polish in Cartesian space to fix any MDS artifacts
            polished_pos, polished_energy = local_minimize_cartesian(coords_opt, maxiter=200, gtol=1e-6)
            
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
    
    # 3. If PGD didn't find anything good, or as a backup, run standard L-BFGS on FCC
    if best_energy > -100.0: # Threshold to trigger backup
        if time.time() - start_time < DEADLINE - 10:
            try:
                fcc_coords = make_fcc_truncated_octahedron()
                if fcc_coords is not None:
                    polished_pos, polished_energy = local_minimize_cartesian(fcc_coords, maxiter=500, gtol=1e-8)
                    if polished_energy < best_energy:
                        best_energy = polished_energy
                        best_pos = polished_pos
            except Exception:
                pass

    # 4. Final Validity Check and Return
    if best_pos is None:
        # Emergency fallback
        best_pos = make_fcc_truncated_octahedron()
        if best_pos is None:
            best_pos = np.zeros((N, 3))
            for i in range(N):
                best_pos[i] = [i*0.6, 0, 0]
                
    return best_pos

# EVOLVE-BLOCK-END
