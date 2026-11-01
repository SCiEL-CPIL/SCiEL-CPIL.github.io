"""Lennard-Jones 55-atom cluster: Parameterized Mackay icosahedron optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55
_START = time.time()
# Safety margin: The evaluator kills at 150s. We target finishing by 145s to be safe.
_DEADLINE = 145.0

def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6) over all pairs (vectorised)."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    return 4.0 * np.sum(inv_r6 * (inv_r6 - 1.0))

def _lj_energy_grad(pos):
    """Energy and gradient for L-BFGS."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    # Avoid division by zero for diagonal, though we sum over all pairs
    # Using a large number or inf is fine as long as we handle it in sum
    # However, for gradient calculation, we need to be careful.
    # Standard trick: set diagonal to 1, then zero out contributions later or handle in sum.
    # Here we use inf for energy, but for gradient we need finite values to avoid nan.
    # Let's use a mask or careful calculation.
    
    # Actually, the standard vectorized gradient formula:
    # grad_i = sum_j 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (r_i - r_j)
    # r_ij^2 = r2[i,j]
    # r_ij^-14 = (r2[i,j])^-7
    # r_ij^-8 = (r2[i,j])^-4
    
    # Let's compute factors
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r2 = 1.0 / r2
        inv_r4 = inv_r2 ** 2
        inv_r7 = inv_r2 ** 3 * inv_r4 # r^-7
        
        # Factor for each pair (i,j) contribution to grad_i
        # dE/dr2 = 4 * (-12 r^-14 + 6 r^-8)
        # dE/dpos_i = 2 * (dE/dr2) * (pos_i - pos_j)
        # Note: The sum is over j.
        
        factor = 4.0 * (-12.0 * inv_r7 * inv_r2 + 6.0 * inv_r4)
        # factor[i,j] corresponds to the coefficient for (pos_i - pos_j) in the sum for grad_i?
        # Let's re-derive.
        # E = sum_{i<j} 4(r^-12 - r^-6)
        # dE/dx_i = sum_{j != i} 4(-12 r^-14 + 6 r^-8) * (x_i - x_j) / r  * (1/r) ? No.
        # d/dr (r^-12) = -12 r^-13.
        # d/dx_i (r^-12) = -12 r^-13 * (x_i - x_j)/r = -12 r^-14 (x_i - x_j).
        # So dE/dx_i = sum_{j != i} 4 * (-12 r^-14 (x_i - x_j) - 6 r^-8 (x_i - x_j))
        #             = sum_{j != i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
        
        # The term (x_i - x_j) is diff[i, j].
        # So we need sum_j factor[i, j] * diff[i, j]
        
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(factor, 0.0)
        
    # grad[i] = sum_j factor[i, j] * diff[i, j]
    # factor[i, j] is a scalar. diff[i, j] is a vector.
    # We can do: factor[:, :, None] * diff, then sum over axis 1.
    
    grad = np.sum(factor[:, :, None] * diff, axis=1)
    
    # Energy calculation
    # E = 0.5 * sum_{i,j} 4(r^-12 - r^-6) = 2 * sum_{i<j} ...
    # Or just sum over upper triangle or all and divide by 2.
    # Using the factor logic:
    # E = 0.5 * sum_{i,j != i} 4(r^-12 - r^-6)
    # r^-12 = inv_r2^6. r^-6 = inv_r2^3.
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    return energy, grad

def _get_icosahedron_geometry():
    """
    Returns unit vectors for:
    12 vertices (V)
    30 edge midpoints (E)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    return v_inner, v_edge

V_INNER, V_EDGE = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: [r1, r2, r3]
    r1: radius for 12 inner vertices
    r2: radius for 30 edge midpoints
    r3: radius for 12 outer vertices (same directions as inner)
    """
    r1, r2, r3 = params
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_INNER * r3
    
    pos = np.vstack([center, shell1, shell2a, shell2b])
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    # Check for minimum distance violation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return _lj_energy(pos)

def _relax_lbfgs(pos, maxiter=2000, gtol=1e-10, ftol=1e-14):
    """Relax using L-BFGS-B."""
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 100}
        )
        return result.x.reshape(-1, 3), float(result.fun)
    except Exception:
        return pos, _lj_energy(pos)

def construct():
    global _START
    best_pos = None
    best_energy = np.inf

    # Candidate radial parameters (r1, r2, r3)
    # Typical optimal values for LJ55 are around r1~1.04, r2~1.76, r3~1.98
    starts = [
        [1.04, 1.76, 1.98],
        [1.05, 1.75, 1.95],
        [1.00, 1.70, 1.90],
        [1.10, 1.80, 2.00],
        [0.95, 1.65, 1.85],
        [1.20, 1.90, 2.10],
        [1.04, 1.70, 1.98],
        [1.04, 1.80, 1.98],
        [1.02, 1.72, 1.96],
        [1.06, 1.78, 2.00],
    ]

    # Step 1: Optimize radial parameters for a few starts
    # We don't need to do this for all starts if time is tight, but Nelder-Mead on 3D is fast.
    optimized_params_list = []
    
    for start in starts:
        if time.time() > _DEADLINE - 10.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < 1e9: # Valid result
                optimized_params_list.append(res.x)
        except Exception:
            continue
            
    # If no optimized params found, use the raw starts
    if not optimized_params_list:
        optimized_params_list = [np.array(s) for s in starts[:3]]

    # Step 2: Full 3D relaxation for the best candidates
    # Sort candidates by energy from radial optimization (approximate)
    # We'll just take the top few unique ones or all if few
    
    # Deduplicate roughly
    unique_params = []
    for p in optimized_params_list:
        is_dup = False
        for up in unique_params:
            if np.linalg.norm(p - up) < 1e-4:
                is_dup = True
                break
        if not is_dup:
            unique_params.append(p)
    
    # Limit number of full relaxations to ensure we finish in time
    # L-BFGS on 165 dims is fast, but let's be safe. 5-10 relaxations should be fine.
    max_relaxations = 8
    
    for i, params in enumerate(unique_params[:max_relaxations]):
        if time.time() > _DEADLINE - 5.0:
            break
            
        pos = _construct_from_params(params)
        
        # Check initial validity
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25 - 1e-9:
            continue
            
        # Relax
        pos_relaxed, e = _relax_lbfgs(pos)
        
        # Validate separation after relaxation
        diff = pos_relaxed[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25 - 1e-9:
            continue
            
        e = _lj_energy(pos_relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()

    # Step 3: Final high-precision polish on the best found
    if best_pos is not None and time.time() < _DEADLINE - 2.0:
        try:
            # Use tighter tolerances
            pos_polished, e_polished = _relax_lbfgs(best_pos, maxiter=5000, gtol=1e-12, ftol=1e-16)
            
            # Validate
            diff = pos_polished[:, np.newaxis, :] - pos[np.newaxis, :, :]
            r2 = np.einsum('ijk,ijk->ij', diff, r2) # Wait, r2 was overwritten? No, local var.
            # Recalculate r2 properly
            diff = pos_polished[:, np.newaxis, :] - pos_polished[np.newaxis, :, :]
            r2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(r2, np.inf)
            
            if np.min(r2) >= 0.25 - 1e-9 and e_polished < best_energy:
                best_pos = pos_polished
                best_energy = e_polished
        except Exception:
            pass

    if best_pos is None:
        # Fallback: seed-like cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
