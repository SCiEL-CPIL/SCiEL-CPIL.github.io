import time
import numpy as np
import sys
from itertools import combinations

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

def _make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    Structure:
    - 1 atom at origin (0,0,0)
    - 12 atoms at (1,1,0) permutations (12 sites, all present)
    - 24 atoms at (2,1,0) permutations (24 sites, choose 19)
    
    We select 19 from the 24 outer sites to form the truncated octahedron.
    The standard truncated octahedron has 24 vertices, but for LJ38 we need 38 atoms total.
    Actually, the standard LJ38 global minimum is often described as a deformed FCC cluster.
    Let's use the known coordinates or generate a symmetric subset.
    
    A common construction:
    Layer 0: 1 atom (0,0,0)
    Layer 1: 12 atoms (1,1,0) type
    Layer 2: 24 atoms (2,1,0) type -> we need 19 of these.
    
    Wait, 1 + 12 + 24 = 37. We need 38.
    Let's check the shell structure of FCC:
    Shell 0: 1 atom (0,0,0)
    Shell 1: 12 atoms (1,1,0) permutations. Distance sqrt(2).
    Shell 2: 6 atoms (2,0,0) permutations. Distance 2.
    Shell 3: 24 atoms (2,1,0) permutations. Distance sqrt(5).
    
    1 + 12 + 6 + 24 = 43.
    The truncated octahedron usually corresponds to specific shells.
    The "FCC truncated octahedron" for N=38 typically includes:
    - 1 center
    - 12 (1,1,0)
    - 6 (2,0,0)
    - 19 (2,1,0)
    Total: 1+12+6+19 = 38.
    
    Let's construct this.
    """
    pts = []
    # Center
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: (1,1,0) permutations
    for perm in [(1,1,0), (1,0,1), (0,1,1),
                 (-1,1,0), (-1,0,1), (0,-1,1),
                 (1,-1,0), (1,0,-1), (0,1,-1),
                 (-1,-1,0), (-1,0,-1), (0,-1,-1)]:
        pts.append(np.array(perm, dtype=np.float64))
        
    # Shell 2: (2,0,0) permutations
    for perm in [(2,0,0), (-2,0,0), (0,2,0), (0,-2,0), (0,0,2), (0,0,-2)]:
        pts.append(np.array(perm, dtype=np.float64))
        
    # Shell 3: (2,1,0) permutations - 24 sites. We need 19.
    # To maintain symmetry as much as possible, we should remove 5 sites.
    # However, the global minimum is not perfectly symmetric.
    # A common heuristic is to remove sites that are "least favorable" or just pick a set.
    # Let's generate all 24 and then select 19.
    # The 24 sites are permutations of (2,1,0) with signs.
    # (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2), (0,1,2)
    # and their sign variations.
    
    base_perms = [(2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2), (0,1,2)]
    shell3_candidates = []
    for p in base_perms:
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    # Only include if the resulting vector is unique and corresponds to a permutation of (2,1,0)
                    # Actually, simply generating all sign variations of the 6 base perms gives 6*8=48, but many are duplicates or not permutations.
                    # Let's just list them explicitly or use a set.
                    vec = np.array([p[0]*sx, p[1]*sy, p[2]*sz], dtype=np.float64)
                    # Check if it's a permutation of (2,1,0)
                    if sorted(np.abs(vec)) == [0, 1, 2]:
                        shell3_candidates.append(vec)
    
    # Remove duplicates
    unique_shell3 = []
    seen = set()
    for v in shell3_candidates:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_shell3.append(v)
            
    # unique_shell3 should have 24 items
    # We need to pick 19.
    # Let's try picking the first 19. This breaks symmetry but might be a good start.
    # A better strategy: Use the known global minimum coordinates if possible, or relax from this.
    # For now, take 19.
    pts.extend(unique_shell3[:19])
    
    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    
    # Scale to typical LJ distance
    # Min distance in this grid is sqrt(2) ~ 1.414.
    # LJ min is 2^(1/6) ~ 1.122.
    # So we need to scale down.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    scale = target_r / min_r
    pos *= scale
    
    return pos

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    from scipy.optimize import minimize

    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _optimize_scale(pos, n_iter=50):
    """
    1D global search over uniform scale factor s applied to centered positions.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    # 1. Generate initial FCC truncated octahedron geometry
    try:
        pos0 = _make_fcc_truncated_octahedron()
    except Exception:
        # Fallback
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0
    best_energy = np.inf

    # 2. Initial relaxation
    try:
        p, e = _relax(pos0, max_iter=500, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # 3. 1D scale optimization
    if time.time() < deadline - 15.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # 4. High-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-13, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 5. Final ultra-tight polish
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=5000, gtol=1e-14, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 6. Final check and separation enforcement
    best_pos = _ensure_min_sep(best_pos)
    
    # 7. Random restarts with small perturbations
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            try:
                perturbed = best_pos + 0.01 * rng.standard_normal(best_pos.shape)
                perturbed = _ensure_min_sep(perturbed)
                p, e = _relax(perturbed, max_iter=200, gtol=1e-8)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                continue

    # 8. One more scale + polish cycle
    if time.time() < deadline - 8.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-13, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    return best_pos
