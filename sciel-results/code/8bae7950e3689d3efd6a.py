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

def min_distance(pos):
    """Compute minimum pairwise distance."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(d, np.inf)
    return np.min(d)

def get_mackay_directions():
    """Returns directions for the Mackay icosahedron shells."""
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

def make_mackay_55(r1, r2, r3):
    """Create a 55-atom cluster based on Mackay icosahedron geometry with given radii."""
    verts, mids = get_mackay_directions()
    pos = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts * r1,
        mids * r2,
        verts * r3
    ])
    return pos

def relax(pos, maxiter=500, gtol=1e-6):
    """Relax using L-BFGS-B."""
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

def optimize_radii_with_relaxation(radii_init, maxiter_nm=80, maxiter_lbfgs=200):
    """
    Joint optimization: optimize shell radii with inner position relaxation.
    This helps find a better Mackay geometry than fixed radii.
    """
    def objective(radii):
        r1, r2, r3 = np.maximum(radii, 0.5)
        pos = make_mackay_55(r1, r2, r3)
        pos_relaxed, e = relax(pos, maxiter=maxiter_lbfgs, gtol=1e-5)
        return e
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-6, 'fatol': 1e-6})
        best_radii = np.maximum(res.x, 0.5)
        best_pos = make_mackay_55(*best_radii)
        best_pos, best_e = relax(best_pos, maxiter=maxiter_lbfgs, gtol=1e-6)
        return best_pos, best_e, best_radii
    except Exception:
        pos = make_mackay_55(*np.maximum(radii_init, 0.5))
        pos_relaxed, e = relax(pos, maxiter=maxiter_lbfgs, gtol=1e-6)
        return pos_relaxed, e, np.maximum(radii_init, 0.5)

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

def make_perturbed_mackay(seed, r1, r2, r3, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_55(r1, r2, r3)
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Standard LJ minimum at r = 2^(1/6) ≈ 1.122
    # For Mackay 55, typical radii: shell1 ~ 1.0-1.1, shell2 ~ 1.7-1.9, shell3 ~ 2.5-2.8
    default_radii = (1.05, 1.80, 2.60)
    
    best_pos = make_mackay_55(*default_radii)
    best_e = lj_energy(best_pos)
    
    # Ensure minimum distance constraint
    if min_distance(best_pos) < 0.5:
        # Scale up to satisfy constraint
        scale_factor = 0.5 / min_distance(best_pos) * 1.01
        best_pos = best_pos * scale_factor
        best_e = lj_energy(best_pos)
    
    candidates = []
    
    # Priority 1: Joint radius optimization from multiple initial guesses
    radius_inits = [
        np.array([1.05, 1.80, 2.60]),
        np.array([1.00, 1.75, 2.55]),
        np.array([1.10, 1.85, 2.65]),
        np.array([0.95, 1.70, 2.50]),
        np.array([1.05, 1.90, 2.70]),
    ]
    
    for i, radii_init in enumerate(radius_inits):
        candidates.append((f"joint_opt_{i}", ("joint", radii_init)))
    
    # Priority 2: Optimized Mackay base with small perturbations
    for s in range(10):
        candidates.append((f"pert_small_{s}", ("perturb", default_radii, s, 0.03)))
    
    # Priority 3: Medium perturbations
    for s in range(8):
        candidates.append((f"pert_med_{s}", ("perturb", default_radii, s, 0.08)))
    
    # Priority 4: Larger perturbations to escape local minima
    for s in range(6):
        candidates.append((f"pert_large_{s}", ("perturb", default_radii, s, 0.15)))
    
    # Priority 5: Even larger perturbations
    for s in range(4):
        candidates.append((f"pert_xl_{s}", ("perturb", default_radii, s, 0.25)))
    
    # Priority 6: Cubic lattice
    candidates.append(("cubic", ("cubic",)))
    
    # Priority 7: Random clusters
    for s in range(8):
        candidates.append((f"random_{s}", ("random", s)))
    
    # Process candidates
    for name, spec in candidates:
        if time.time() > deadline - 25:
            break
        
        if spec[0] == "joint":
            radii_init = spec[1]
            try:
                pos, e, radii = optimize_radii_with_relaxation(radii_init, maxiter_nm=60, maxiter_lbfgs=150)
                if min_distance(pos) >= 0.5 and e < best_e:
                    best_e = e
                    best_pos = pos.copy()
            except Exception:
                continue
        
        elif spec[0] == "perturb":
            r1, r2, r3, seed, scale = spec[1], spec[2], spec[3], spec[4], spec[5]
            pos = make_perturbed_mackay(seed, r1, r2, r3, scale)
            if min_distance(pos) < 0.5:
                continue
            e_raw = lj_energy(pos)
            if time.time() < deadline - 15:
                pos_relaxed, e_relaxed = relax(pos, maxiter=400, gtol=1e-6)
                if e_relaxed < best_e and min_distance(pos_relaxed) >= 0.5:
                    best_e = e_relaxed
                    best_pos = pos_relaxed.copy()
            if e_raw < best_e and min_distance(pos) >= 0.5:
                best_e = e_raw
                best_pos = pos.copy()
        
        elif spec[0] == "cubic":
            pos = make_cubic_lattice()
            e_raw = lj_energy(pos)
            if time.time() < deadline - 15:
                pos_relaxed, e_relaxed = relax(pos, maxiter=400, gtol=1e-6)
                if e_relaxed < best_e and min_distance(pos_relaxed) >= 0.5:
                    best_e = e_relaxed
                    best_pos = pos_relaxed.copy()
            if e_raw < best_e and min_distance(pos) >= 0.5:
                best_e = e_raw
                best_pos = pos.copy()
        
        elif spec[0] == "random":
            seed = spec[1]
            pos = make_random_cluster(seed)
            e_raw = lj_energy(pos)
            if time.time() < deadline - 15:
                pos_relaxed, e_relaxed = relax(pos, maxiter=300, gtol=1e-5)
                if e_relaxed < best_e and min_distance(pos_relaxed) >= 0.5:
                    best_e = e_relaxed
                    best_pos = pos_relaxed.copy()
            if e_raw < best_e and min_distance(pos) >= 0.5:
                best_e = e_raw
                best_pos = pos.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        pos_final, e_final = relax(best_pos, maxiter=3000, gtol=1e-8)
        if min_distance(pos_final) >= 0.5 and e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    # Second polish pass with even tighter tolerance if time permits
    if time.time() < deadline - 3:
        pos_final2, e_final2 = relax(best_pos, maxiter=5000, gtol=1e-10)
        if min_distance(pos_final2) >= 0.5 and e_final2 < best_e:
            best_e = e_final2
            best_pos = pos_final2
    
    # Final safety check: ensure minimum distance
    if min_distance(best_pos) < 0.5:
        # Fallback to a valid configuration
        fallback = make_mackay_55(1.1, 1.9, 2.7)
        if min_distance(fallback) >= 0.5:
            best_pos = fallback
        else:
            # Scale up
            scale_factor = 0.5 / min_distance(fallback) * 1.02
            best_pos = fallback * scale_factor
    
    return best_pos

# EVOLVE-BLOCK-END
