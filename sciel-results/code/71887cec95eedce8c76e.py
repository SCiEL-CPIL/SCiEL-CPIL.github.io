"""
LJ38 cluster global minimum search.

Strategy:
1. Generate multiple deterministic starting points: FCC truncated octahedron,
   icosahedral structure, and random FCC subsets.
2. Relax each configuration using L-BFGS.
3. Perform basin hopping on the best results to escape local minima.
4. Final high-precision polish with tighter tolerances.
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
    """Generate 19 FCC core points and 24 outer shell points."""
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
    core_pts = pts[pts[:, 3] <= 4]  # 19 atoms (d2 = 0, 2, 4)
    outer_pts = pts[pts[:, 3] == 6]  # 24 atoms (d2 = 6)
    return core_pts[:, :3], outer_pts[:, :3]

def _make_fcc_truncated_octahedron():
    """Construct the 38-atom FCC truncated octahedron start."""
    core_pts, outer_pts = _generate_fcc_core_and_outer()
    start = np.vstack([core_pts, outer_pts[:19]])
    return _center_and_scale(start)

def _make_icosahedral_structure():
    """Construct an icosahedral starting structure for N=38."""
    # Icosahedron vertices: 12 vertices at (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    phi = (1 + np.sqrt(5)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # 12 icosahedron vertices + 1 center + 25 more
    # For N=38: 12 (outer) + 1 (center) + 12 (inner shell) + 13 (more)
    # Use a layered approach: center, inner icosahedron, outer icosahedron
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Scale inner icosahedron
    inner_scale = 0.5
    inner_ico = ico * inner_scale
    
    # Scale outer icosahedron  
    outer_scale = 1.0
    outer_ico = ico * outer_scale
    
    # We have 1 + 12 + 12 = 25, need 13 more
    # Add points on edges of the outer icosahedron
    edge_points = []
    for i in range(12):
        for j in range(i+1, 12):
            mid = (outer_ico[i] + outer_ico[j]) / 2.0
            edge_points.append(mid)
    
    edge_points = np.array(edge_points[:13])
    
    start = np.vstack([center, inner_ico, outer_ico, edge_points])
    return _center_and_scale(start)

def _center_and_scale(pos):
    """Center positions and scale to have average NN distance ~ 2^(1/6)."""
    start = pos.copy()
    start -= np.mean(start, axis=0)
    diff = start[:, np.newaxis, :] - start[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        start *= scale
    return start

def _optimize_scale(pos, n_iter=50):
    """Optimize the overall scale of the cluster to minimize energy."""
    def energy_at_scale(s):
        p = pos * s
        e, _ = _lj_energy_and_grad(p)
        return e
    
    # Initial scale estimate
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    s0 = (2.0 ** (1.0 / 6.0)) / avg_nn if avg_nn > 1e-12 else 1.0
    
    result = minimize_scalar(energy_at_scale, bounds=(0.5, 2.0), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': n_iter})
    return pos * result.x

def _relax(pos, max_iter=2000, gtol=1e-14):
    """L-BFGS-B local relaxation with tight tolerances."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-16},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart."""
    pos = pos.copy()
    for _ in range(100):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(r, np.inf)
        if np.min(r) >= min_dist:
            break
        # Find closest pair and push them apart
        i, j = np.unravel_index(np.argmin(r), r.shape)
        if i >= j:
            continue
        direction = pos[i] - pos[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([1.0, 0, 0])
            norm = 1.0
        direction /= norm
        push = (min_dist - r[i, j]) / 2.0 + 1e-6
        pos[i] += direction * push
        pos[j] -= direction * push
    return pos

def _basin_hop(pos, n_steps=20, step_size=1.0, seed=42):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    rng = np.random.default_rng(seed)
    best_pos = pos.copy()
    best_energy, _ = _lj_energy_and_grad(pos)
    
    for step in range(n_steps):
        # Random perturbation
        delta = rng.normal(0, step_size, size=pos.shape)
        perturbed = pos + delta
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax
        try:
            relaxed, e = _relax(perturbed, max_iter=500, gtol=1e-10)
        except Exception:
            continue
        
        # Metropolis-like acceptance (always accept if better)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
        pos = relaxed
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    best_pos = None
    best_energy = np.inf

    # Generate multiple starting points
    starts = []
    
    # 1. FCC truncated octahedron
    try:
        starts.append(_make_fcc_truncated_octahedron())
    except Exception:
        pass
    
    # 2. Icosahedral structure
    try:
        starts.append(_make_icosahedral_structure())
    except Exception:
        pass
    
    # 3. Random FCC subsets (5 different ones)
    core_pts, outer_pts = _generate_fcc_core_and_outer()
    rng = np.random.default_rng(42)
    for i in range(5):
        idx = rng.choice(len(outer_pts), size=19, replace=False)
        start = np.vstack([core_pts, outer_pts[idx]])
        starts.append(_center_and_scale(start))

    # Relax each starting point
    for i, start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        try:
            p, e = _relax(start, max_iter=1000, gtol=1e-12)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            continue

    # Basin hopping on the best result
    if best_pos is not None and time.time() < deadline - 30.0:
        try:
            bh_pos, bh_energy = _basin_hop(best_pos, n_steps=15, step_size=0.8, seed=123)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass

    # Final polish: scale optimization + tight relaxation
    if best_pos is not None and time.time() < deadline - 10.0:
        try:
            polished = _optimize_scale(best_pos)
            polished, e = _relax(polished, max_iter=3000, gtol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = polished
        except Exception:
            pass

    # Ensure minimum separation
    if best_pos is not None:
        best_pos = _ensure_min_sep(best_pos, min_dist=0.5)

    if best_pos is None:
        # Last resort: return FCC start
        best_pos = _make_fcc_truncated_octahedron()

    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
