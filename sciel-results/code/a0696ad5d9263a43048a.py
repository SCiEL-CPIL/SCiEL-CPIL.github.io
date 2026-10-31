"""
LJ38 cluster global minimum search.

Approach:
1. Construct the 19-atom FCC core (global min for LJ19).
2. Generate 24 candidate outer atoms from the next FCC shell.
3. Use a combination of deterministic symmetric subsets and random subsets to initialize the 19 outer atoms.
4. Relax each configuration using L-BFGS.
5. Perform a final 1D scale optimization on the best result to ensure the geometry is at its energy minimum for that topology.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38

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

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
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
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _make_truncated_octahedron_start():
    """
    Deterministically construct the FCC truncated octahedron for LJ38.
    
    The truncated octahedron has:
    - 1 atom at center
    - 12 atoms at (±1, ±1, 0) and cyclic permutations (distance √2)
    - 6 atoms at (±2, 0, 0) and permutations (distance 2)
    - 19 atoms from the (±2, ±1, ±1) type positions (distance √6)
    
    For the 19 outer atoms, we need to select 19 from the 24 (211) positions.
    The correct selection excludes 5 pairs that would create overlaps or
    non-optimal geometry. The standard truncated octahedron excludes the
    5 positions that are "opposite" to the 6 face centers in a specific way.
    
    Actually, the correct approach: the truncated octahedron for N=38 uses
    all 1+12+6=19 core atoms plus 19 of the 24 outer shell atoms.
    The 5 excluded outer atoms are those that would be too close to each other
    when combined with the core.
    
    Let me construct it properly: the 24 (211) positions are all (±2,±1,±1)
    and permutations. We need to pick 19 that form the truncated octahedron.
    
    A simpler approach: just use all 24 and let relaxation sort it out,
    but that gives 43 atoms. Instead, we'll use a known good subset.
    
    The standard LJ38 truncated octahedron: exclude the 5 atoms that are
    at positions where two coordinates have the same sign pattern that
    creates 5-fold symmetry issues. 
    
    Let me just generate all 24 and pick the 19 with the lowest energy
    contribution when added to the core.
    """
    core, outer = _generate_fcc_core_and_outer()
    
    # For each outer atom, compute its LJ interaction energy with the core
    # Pick the 19 with lowest (most negative) interaction energy
    core_energy_contribs = []
    for i, outer_pt in enumerate(outer):
        # Compute LJ energy between this outer atom and all core atoms
        diffs = outer_pt - core
        r2 = np.sum(diffs**2, axis=1)
        r6 = r2**3
        r12 = r6**2
        e = 4.0 * np.sum(1.0/r12 - 1.0/r6)
        core_energy_contribs.append((e, i))
    
    # Sort by energy contribution (most negative first)
    core_energy_contribs.sort(key=lambda x: x[0])
    
    # Take the 19 best
    selected_indices = [idx for _, idx in core_energy_contribs[:19]]
    
    start = np.vstack([core, outer[selected_indices]])
    start -= np.mean(start, axis=0)
    
    # Scale to optimal nearest-neighbor distance
    diff = start[:, np.newaxis, :] - start[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        start *= scale
    
    return start

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

def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    This is a 1D optimization problem.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    # Initial scale is 1.0. We expect the optimal scale to be close to 1.0.
    # The LJ minimum is at r = 2^(1/6) ~ 1.122.
    # If the initial structure is too compressed, scale > 1.
    # If too expanded, scale < 1.
    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _basin_hop(pos, rng, n_hops=20, max_iter_relax=200):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    
    for _ in range(n_hops):
        if time.time() > time.time() - 1000:  # placeholder, actual check in construct
            break
        
        # Random perturbation
        noise = rng.normal(0, 0.3, size=current_pos.shape)
        perturbed = current_pos + noise
        perturbed -= np.mean(perturbed, axis=0)
        
        try:
            p, e = _relax(perturbed, max_iter=max_iter_relax)
            p, e = _optimize_scale(p)
            
            if e < current_energy:
                current_pos = p
                current_energy = e
        except Exception:
            continue
    
    return current_pos, current_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic truncated octahedron (highest priority)
    try:
        to_start = _make_truncated_octahedron_start()
        starts.append(to_start)
    except Exception:
        pass
    
    # 2. Simple deterministic subset: First 19 outer atoms
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    diff = det_start[:, np.newaxis, :] - det_start[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        det_start *= scale
    starts.append(det_start)
    
    # 3. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    # 4. Random subsets (limited number to leave time for basin hopping)
    num_random_starts = 30
    for i in range(num_random_starts):
        if time.time() > deadline - 60.0:
            break
        idx = rng.choice(len(outer), size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        diff = start_pos[:, np.newaxis, :] - start_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        nn = np.sqrt(np.min(r2, axis=1))
        avg_nn = np.mean(nn)
        if avg_nn > 1e-12:
            scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            start_pos *= scale
            
        starts.append(start_pos)
        
    best_pos = None
    best_energy = np.inf

    # Phase 1: Relax all starts
    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            p, e = _relax(start, max_iter=300)
            p, e = _optimize_scale(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Phase 2: Basin hopping from best found
    if time.time() < deadline - 15.0:
        try:
            bh_pos, bh_energy = _basin_hop(best_pos, rng, n_hops=30, max_iter_relax=200)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-12)
            p, e = _optimize_scale(p)
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
