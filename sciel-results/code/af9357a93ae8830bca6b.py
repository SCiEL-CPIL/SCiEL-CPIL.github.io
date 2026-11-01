# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

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

def get_mackay_directions():
    """Returns directions for the 55-atom Mackay icosahedron structure."""
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
    
    return verts, midpoints

def make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    verts, mids = get_mackay_directions()
    r1, r2, r3 = radii
    pos = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts * r1,
        mids * r2,
        verts * r3
    ])
    return pos

def relax_positions(pos, maxiter=300, gtol=1e-6):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def optimize_radii_and_relax(radii_init, maxiter_nm=100, maxiter_lbfgs=200, deadline=None):
    """
    Jointly optimize shell radii and relaxed positions.
    """
    def objective(radii):
        radii = np.maximum(radii, 0.5)
        pos = make_mackay_from_radii(radii)
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-5)
        return e
    
    # Optimize radii
    res = minimize(objective, radii_init, method='Nelder-Mead', 
                   options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_radii = np.maximum(res.x, 0.5)
    best_pos = make_mackay_from_radii(best_radii)
    best_pos, best_e = relax_positions(best_pos, maxiter=maxiter_lbfgs, gtol=1e-6)
    
    return best_pos, best_e, best_radii

def make_perturbed_mackay(seed, base_pos, scale=0.1):
    """Generate a perturbed version of a base Mackay structure."""
    rng = np.random.RandomState(seed)
    pos = base_pos + rng.randn(N, 3) * scale
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.0
    return pos

def polish(pos, deadline):
    """Multi-stage high-precision polish: progressively tighten L-BFGS tolerance."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    best_pos = pos.copy()
    best_e = lj_energy(best_pos)
    x0 = pos.reshape(-1)
    
    stages = [
        {'maxiter': 3000, 'gtol': 1e-10, 'maxls': 200, 'ftol': 1e-12},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-14},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-15},
        {'maxiter': 15000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-16},
    ]
    
    for opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, method='L-BFGS-B', jac=True, options=opts)
            if result.fun < best_e - 1e-12:
                best_e = result.fun
                best_pos = result.x.reshape(N, 3)
                x0 = result.x
            else:
                break
        except Exception:
            break
            
    return best_pos, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Joint radius+position optimization with multiple initial guesses
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.15, 1.95, 2.75]),
    ]
    
    best_pos = None
    best_e = np.inf
    
    for radii_init in radius_candidates:
        if time.time() > deadline - 30:
            break
        
        try:
            pos, e, radii = optimize_radii_and_relax(
                radii_init, 
                maxiter_nm=80, 
                maxiter_lbfgs=150
            )
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # If radius optimization didn't give us anything good, fall back to direct construction
    if best_pos is None:
        base = make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_pos, best_e = relax_positions(base, maxiter=300, gtol=1e-6)
    
    # Phase 2: Generate diverse candidates and relax them
    candidates = []
    
    # Perturbed versions of the best structure
    for s in range(8):
        candidates.append((f"pert_{s}", make_perturbed_mackay(s, best_pos, scale=0.05)))
    
    # Larger perturbations to escape local minima
    for s in range(4):
        candidates.append((f"pert_large_{s}", make_perturbed_mackay(s + 100, best_pos, scale=0.15)))
    
    # Cubic lattice
    candidates.append(("cubic", make_cubic_lattice()))
    
    # Random
    for s in range(4):
        candidates.append((f"random_{s}", make_random_cluster(s)))
    
    # Relax candidates
    for name, pos in candidates:
        if time.time() > deadline - 15:
            break
            
        pos_relaxed, e_relaxed = relax_positions(pos, maxiter=200, gtol=1e-5)
        
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = polish(best_pos, deadline)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
            
    return best_pos

# EVOLVE-BLOCK-END
