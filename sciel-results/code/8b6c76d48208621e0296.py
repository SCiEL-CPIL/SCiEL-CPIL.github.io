"""
LJ38 cluster global minimum search.

Approach:
1. Construct the 19-atom FCC core and generate 24 candidate outer atoms.
2. Use deterministic and random subsets of these to create diverse starting geometries.
3. Include an icosahedral start to explore that funnel.
4. Relax each configuration using L-BFGS.
5. Apply an analytically optimal global scale factor to each relaxed structure.
6. Perform basin hopping from the best configuration to escape local minima.
7. Final high-precision polish on the global best.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
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

def _analytical_optimal_scale(pos):
    """
    Compute the analytically optimal scale factor for minimum LJ energy.
    E(s) = A/s^12 - B/s^6
    dE/ds = -12A/s^13 + 6B/s^7 = 0 => s^6 = 2A/B => s = (2A/B)^(1/6)
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # A = 0.5 * sum(1/r^12), B = 0.5 * sum(1/r^6)
    A = 0.5 * np.sum(1.0 / r12)
    B = 0.5 * np.sum(1.0 / r6)
    
    if B > 1e-15:
        s = (2.0 * A / B) ** (1.0 / 6.0)
    else:
        s = 1.0
    return s

def _scale_pos(pos, target_nn=None):
    """
    Scale positions so that the average nearest neighbor distance is target_nn.
    If target_nn is None, use the analytical optimal scale.
    """
    if target_nn is None:
        s = _analytical_optimal_scale(pos)
        return pos * s
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = target_nn / avg_nn
        return pos * scale
    return pos

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 6.5:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    core_pts = pts[pts[:, 3] <= 4.1]
    outer_pts = pts[(pts[:, 3] > 4.1) & (pts[:, 3] <= 6.1)]
    
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
    
    # Scale to optimal NN
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

def _make_truncated_octahedron_start():
    """
    Create a starting geometry that closely resembles the FCC truncated octahedron.
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale to optimal NN
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
        
    return selected

def _basin_hop(pos, rng, max_iters=100, T=0.1, t0=1.0, t1=0.01, deadline=None):
    """
    Basin hopping: perturb, relax, accept/reject by Metropolis criterion.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Perturbation: add Gaussian noise
        noise_scale = T
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-9)
            # Apply optimal scale after relaxation
            relaxed = _scale_pos(relaxed)
            e_relaxed, _ = _lj_energy_and_grad(relaxed)
        except Exception:
            continue

        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            T_curr = t0 * (t1 / t0) ** (i / max_iters)
            accept = rng.random() < np.exp(-delta_e / T_curr)
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Best guess: truncated octahedron from FCC lattice
    try:
        to_start = _make_truncated_octahedron_start()
        starts.append(to_start)
    except Exception:
        pass
    
    # 2. Deterministic subsets
    if len(outer) >= 19:
        det_start = np.vstack([core, outer[:19]])
        det_start -= np.mean(det_start, axis=0)
        det_start = _scale_pos(det_start, target_nn=2.0 ** (1.0 / 6.0))
        starts.append(det_start)
    elif len(outer) > 0 and len(core) + len(outer) == N:
        det_start = np.vstack([core, outer])
        det_start -= np.mean(det_start, axis=0)
        det_start = _scale_pos(det_start, target_nn=2.0 ** (1.0 / 6.0))
        starts.append(det_start)
    
    # 3. Random subsets
    num_random_starts = 40
    if len(outer) >= 19:
        for i in range(num_random_starts):
            if time.time() > deadline - 60.0:
                break
            idx = rng.choice(len(outer), size=19, replace=False)
            start_pos = np.vstack([core, outer[idx]])
            start_pos -= np.mean(start_pos, axis=0)
            start_pos = _scale_pos(start_pos, target_nn=2.0 ** (1.0 / 6.0))
            starts.append(start_pos)
    else:
        if len(starts) > 0:
            base = starts[0].copy()
            for i in range(num_random_starts):
                if time.time() > deadline - 60.0:
                    break
                perturbed = base + rng.normal(0, 0.1, size=base.shape)
                starts.append(perturbed)
    
    # 4. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    # Phase 1: Quick relaxation of all starts
    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            p, e = _relax(start, max_iter=100)
            p = _scale_pos(p) # Analytical scale
            e, _ = _lj_energy_and_grad(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        if len(starts) > 0:
            best_pos = starts[0]
            best_energy, _ = _lj_energy_and_grad(best_pos)
        else:
            best_pos = rng.normal(0, 1, size=(N, 3))
            best_energy, _ = _lj_energy_and_grad(best_pos)

    # Phase 2: Basin hopping from the best configuration
    if time.time() < deadline - 25.0:
        try:
            p_bh, e_bh = _basin_hop(best_pos, rng, max_iters=50, T=0.15, t0=0.3, t1=0.01, deadline=deadline - 25.0)
            p_bh = _scale_pos(p_bh)
            e_bh, _ = _lj_energy_and_grad(p_bh)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh.copy()
        except Exception:
            pass

    # Phase 3: High-precision polish on best result
    if time.time() < deadline - 15.0:
        try:
            p, e = _relax(best_pos, max_iter=500, gtol=1e-11)
            p = _scale_pos(p)
            e, _ = _lj_energy_and_grad(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # Phase 4: Final ultra-high-precision polish
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-13)
            p = _scale_pos(p)
            e, _ = _lj_energy_and_grad(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # Phase 5: Multi-start refinement around best (if time permits)
    if time.time() < deadline - 2.0 and best_pos is not None:
        for trial in range(3):
            if time.time() > deadline - 1.0:
                break
            try:
                perturbed = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
                p, e = _relax(perturbed, max_iter=300, gtol=1e-12)
                p = _scale_pos(p)
                e, _ = _lj_energy_and_grad(p)
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                continue

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
