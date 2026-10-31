"""
LJ38 cluster global minimum search.

Approach:
1. Identify the core 19 atoms of the FCC truncated-octahedron (center + 12 nearest + 6 next-nearest).
2. Identify the outer 24 candidate atoms at the next shell distance.
3. Since the global minimum (38 atoms) is a symmetric truncated octahedron, it corresponds to a specific subset of 19 atoms from the 24 outer candidates.
4. We perform multiple local optimizations (L-BFGS) starting from different random subsets of 19 outer atoms.
5. We also include the deterministic "38 closest" subset and an icosahedral starting point.
6. The best energy configuration is returned.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
    core = []
    # Center
    core.append([0.0, 0.0, 0.0])
    
    # 12 atoms: permutations of (1, 1, 0)
    for i in range(3):
        for j in range(3):
            if i != j:
                # Generate permutations of (1, 1, 0)
                pass
    
    # Simpler: generate all FCC points up to dist^2 = 6
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    if d2 <= 6:
                        pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    
    # Separate by distance
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    # core_pts should have 1 + 12 + 6 = 19 atoms
    # outer_pts should have 24 atoms
    return core_pts[:, :3], outer_pts[:, :3]

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic "38 closest" (first 19 outer atoms)
    # This is equivalent to the previous program's approach
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    # Scale
    diff = det_start[:, np.newaxis, :] - det_start[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        det_start *= scale
    starts.append(det_start)
    
    # 2. Random subsets of outer atoms
    # Increase number of restarts to improve probability of finding the global minimum
    # The global minimum is a specific subset of the 24 outer atoms.
    num_random_starts = 100
    for i in range(num_random_starts):
        if time.time() > deadline - 30.0:
            break
        # Pick 19 random outer atoms
        idx = rng.choice(len(outer), size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        # Scale
        diff = start_pos[:, np.newaxis, :] - start_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        nn = np.sqrt(np.min(r2, axis=1))
        avg_nn = np.mean(nn)
        if avg_nn > 1e-12:
            scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            start_pos *= scale
            
        starts.append(start_pos)
        
    # 3. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    for start in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            # Use slightly fewer iterations for the first pass to allow more restarts
            p, e = _relax(start, max_iter=200)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final polish
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-12)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    # Test
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
