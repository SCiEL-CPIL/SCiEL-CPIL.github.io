"""Lennard-Jones 55-atom cluster: parameterized Mackay icosahedron manifold optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_icosahedron_directions():
    """Return 12 vertex, 30 edge-midpoint, and 20 face-center unit vectors of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d1, edge_len, atol=1e-10) and np.isclose(d2, edge_len, atol=1e-10) and np.isclose(d3, edge_len, atol=1e-10):
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    return verts, midpoints, face_centers

# Precompute the direction vectors for the 55-atom Mackay structure
_VERTS, _MIDPOINTS, _FACE_CENTERS = _get_icosahedron_directions()
# 1 center + 12 vertices (inner) + 30 midpoints + 12 face centers (outer) = 55
_DIRECTIONS = np.vstack([
    np.zeros((1, 3)),
    _VERTS,
    _MIDPOINTS,
    _FACE_CENTERS[:12]
])

def _make_rotation_matrix(rx, ry, rz):
    """Create rotation matrix from Euler angles (ZYX convention)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def _param_to_pos(params):
    """Convert parameter vector to positions.
    params: [r_inner, r_outer, rx, ry, rz, tx, ty, tz]
    """
    r_inner = params[0]
    r_outer = params[1]
    rx, ry, rz = params[2], params[3], params[4]
    t = params[5:8]
    
    R = _make_rotation_matrix(rx, ry, rz)
    
    # Radial coordinates: center=0, inner shell=r_inner, outer shell=r_outer
    radii = np.zeros(N)
    radii[1:13] = r_inner      # 12 inner shell atoms
    radii[13:] = r_outer        # 42 outer shell atoms
    
    # Base positions: direction * radius
    base = _DIRECTIONS * radii[:, np.newaxis]
    
    # Apply rotation and translation
    pos = (base @ R.T) + t
    return pos

def _param_energy(params):
    """Energy as function of parameters."""
    pos = _param_to_pos(params)
    return lj_energy(pos)

def _param_energy_grad(params):
    """Energy and gradient with respect to parameters using finite differences."""
    pos = _param_to_pos(params)
    e, grad_pos = lj_energy_grad(pos)
    
    # Compute Jacobian of pos w.r.t. params numerically
    # params: [r_inner, r_outer, rx, ry, rz, tx, ty, tz]
    nparams = len(params)
    jac = np.zeros((N, 3, nparams))
    
    # dr_inner/dpos
    jac[1:13, :, 0] = _DIRECTIONS[1:13, :][:, :, np.newaxis] * _make_rotation_matrix(params[2], params[3], params[4]).T
    # dr_outer/dpos
    jac[13:, :, 1] = _DIRECTIONS[13:, :][:, :, np.newaxis] * _make_rotation_matrix(params[2], params[3], params[4]).T
    
    # Rotation derivatives - use finite difference for simplicity
    eps = 1e-7
    for i in range(2, 5):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        pos_plus = _param_to_pos(p_plus)
        pos_minus = _param_to_pos(p_minus)
        jac[:, :, i] = (pos_plus - pos_minus) / (2 * eps)
    
    # Translation
    jac[:, :, 5] = np.eye(3)[:, :, np.newaxis]
    jac[:, :, 6] = np.zeros((N, 3, 1))
    jac[:, :, 7] = np.zeros((N, 3, 1))
    # Actually translation is just adding t, so:
    jac[:, :, 5:8] = np.zeros((N, 3, 3))
    for k in range(3):
        jac[:, k, 5+k] = 1.0
    
    # Chain rule: dE/dparam = sum_ij grad_pos[i,j] * dpos[i,j]/dparam
    grad_params = np.zeros(nparams)
    for p in range(nparams):
        grad_params[p] = np.sum(grad_pos * jac[:, :, p])
    
    return e, grad_params

def relax_full(pos, maxiter=1000, gtol=1e-12, ftol=1e-14):
    """Relax full Cartesian positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def relax_params(params, maxiter=2000, gtol=1e-10, ftol=1e-12):
    """Relax using parameterized manifold optimization."""
    def fun(p):
        e, g = _param_energy_grad(p)
        return e, g
    
    try:
        result = minimize(fun, params, method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x, result.fun
    except Exception:
        pos = _param_to_pos(params)
        return params, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial parameter guess: r_inner=1.04, r_outer=1.96, no rotation/translation
    init_params = np.array([1.04, 1.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Step 1: Optimize on the parameterized manifold
    opt_params, e_opt = relax_params(init_params, maxiter=3000, gtol=1e-12, ftol=1e-14)
    
    best_pos = _param_to_pos(opt_params)
    best_e = e_opt
    
    # Step 2: Full Cartesian relaxation from the parameter-optimized position
    pos_full, e_full = relax_full(best_pos, maxiter=2000, gtol=1e-13, ftol=1e-15)
    if e_full < best_e:
        best_e = e_full
        best_pos = pos_full.copy()
    
    # Step 3: Try a few alternative initial radii on the manifold
    alt_radii = [(1.02, 1.94), (1.06, 1.98), (1.00, 1.92), (1.08, 2.00)]
    for r_in, r_out in alt_radii:
        if time.time() > deadline - 30:
            break
        p_try = np.array([r_in, r_out, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p_opt, e_try = relax_params(p_try, maxiter=1000, gtol=1e-11, ftol=1e-13)
        pos_try = _param_to_pos(p_opt)
        if e_try < best_e:
            best_e = e_try
            best_pos = pos_try.copy()
        # Full relax
        if time.time() < deadline - 20:
            pos_r, e_r = relax_full(pos_try, maxiter=1000, gtol=1e-12, ftol=1e-14)
            if e_r < best_e:
                best_e = e_r
                best_pos = pos_r.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax_full(best_pos, maxiter=3000, gtol=1e-14, ftol=1e-16)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
