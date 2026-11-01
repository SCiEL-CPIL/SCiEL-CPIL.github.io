"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient."""
    pos = pos.reshape(-1, 3)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    
    # Energy: 4 * sum(r^-12 - r^-6)
    r6 = r ** -6
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * (N - 1) # Subtract self-energy terms? No, diagonal was set to 1, so r12-r6 is 0.
    # Actually, diagonal r=1 -> 4(1-1)=0. So just sum is fine.
    
    # Gradient
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 * (r^-7 - 2 r^-13)
    # dE/dx_i = sum_j dE/dr_ij * d r_ij / d x_i
    # d r_ij / d x_i = - (x_i - x_j) / r_ij
    # So dE/dx_i = sum_j 24 * (r^-7 - 2 r^-13) * (x_i - x_j) / r
    # = 24 * sum_j (r^-8 - 2 r^-14) * (x_i - x_j)
    
    # Vectorized gradient calculation
    # r8 = r^-8, r14 = r^-14
    r8 = r ** -8
    r14 = r ** -14
    coeff = 24.0 * (r8 - 2.0 * r14)
    
    # coeff is (N, N) matrix. coeff[i, j] is the coefficient for vector (x_i - x_j)
    # grad[i] = sum_j coeff[i, j] * (x_i - x_j)
    # grad[i] = sum_j coeff[i, j] * x_i - sum_j coeff[i, j] * x_j
    # sum_j coeff[i, j] is row_sum
    row_sum = np.sum(coeff, axis=1)
    grad = row_sum[:, np.newaxis] * pos - np.dot(coeff, pos)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Generate Mackay Icosahedron Initial Guess
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) # Normalize to unit sphere
    
    # Edge midpoints (30)
    # Icosahedron edges connect vertices that are distance 2 apart (in unnormalized coords with 1, phi)
    # Distance squared between adjacent vertices is 4.
    # Let's find all pairs with dist^2 close to 4 (in unnormalized) or 2.0 in normalized?
    # Normalized ico: distance between adjacent vertices is 2/sqrt(1+phi^2) ?
    # Norm of [0,1,phi] is sqrt(1+phi^2).
    # Adjacent vertex [1,phi,0]. Dot product: 0*1 + 1*phi + phi*0 = phi.
    # cos(theta) = phi / (1+phi^2).
    # dist^2 = 2 - 2*phi/(1+phi^2).
    
    # Let's just compute all pairwise distances and pick the smallest ones.
    # There are 12 vertices, 66 pairs. 30 edges.
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # The 30 edges correspond to the 30 smallest distances.
    # Actually, all edge distances are equal.
    # Sort distances and take unique?
    unique_dists = np.unique(dists)
    min_dist = unique_dists[0]
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i, j], min_dist):
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    # Normalize edge mids to unit sphere
    norms = np.linalg.norm(edge_mids, axis=1)
    edge_mids = edge_mids / norms[:, np.newaxis]
    
    # Construct initial positions
    # Shell 1: 1 center + 12 ico verts
    # Shell 2: 12 ico verts + 30 edge mids
    
    r1 = 1.12 # Typical LJ minimum distance
    r2 = 2.2  # Typical 2nd shell radius
    
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]
    pos[1:13] = ico_verts * r1
    pos[13:25] = ico_verts * r2
    pos[25:55] = edge_mids * r2
    
    # 2. Relaxation
    best_pos = pos
    best_energy = _lj_energy_and_grad(pos)[0]
    
    # Use L-BFGS-B
    # We need to ensure atoms don't overlap. L-BFGS-B handles bounds, but we don't have bounds.
    # We can use a penalty or just hope the relaxation stays valid.
    # The LJ potential goes to infinity at r->0, so the optimizer should avoid overlaps.
    
    def fun(x):
        e, g = _lj_energy_and_grad(x)
        return e, g
    
    # Run optimization
    try:
        res = minimize(fun, pos, jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < best_energy:
            best_pos = res.x
            best_energy = res.fun
    except Exception as e:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_pos
    
    # 3. Multi-start / Local Search if time permits
    # Try a few random perturbations of the initial guess to escape local minima?
    # The Mackay icosahedron is the global minimum, so we should be close.
    # However, the initial guess might not be perfect.
    
    # Try relaxing with different initial radii
    radii_set = [(1.1, 2.1), (1.15, 2.25), (1.12, 2.2)]
    
    for r1_try, r2_try in radii_set:
        if time.time() > deadline - 5.0:
            break
            
        pos_try = np.zeros((55, 3))
        pos_try[0] = [0, 0, 0]
        pos_try[1:13] = ico_verts * r1_try
        pos_try[13:25] = ico_verts * r2_try
        pos_try[25:55] = edge_mids * r2_try
        
        try:
            res = minimize(fun, pos_try, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
