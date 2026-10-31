import time
import numpy as np
from scipy.optimize import minimize

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
    
    # Separate by distance
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _analytical_optimal_scale(pos):
    """
    Calculate the optimal global scale factor s such that 
    E(s) = sum 4(s^-12 r^-12 - s^-6 r^-6) is minimized.
    Let u = s^-6. E(u) = 4 u^2 Sum(r^-12) - 4 u Sum(r^-6).
    Min at u = Sum(r^-6) / (2 Sum(r^-12)).
    s = u^(-1/6).
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Sum over all pairs (i<j)
    # To avoid double counting and self-interaction:
    # We can sum over all i,j and divide by 2, but we must exclude i=j.
    # Since we set diagonal to 1, 1/r12 and 1/r6 on diagonal are 1.
    # We subtract the diagonal contributions.
    sum_r12 = np.sum(1.0 / r12) - N # Subtract N diagonal terms (1.0 each)
    sum_r6 = np.sum(1.0 / r6) - N    # Subtract N diagonal terms (1.0 each)
    
    # The energy is 0.5 * sum_{i!=j} ...
    # So the sums in the derivative should be over i!=j.
    # E(s) = 2 * sum_{i<j} 4 (s^-12 r^-12 - s^-6 r^-6)
    # Let S12 = sum_{i<j} r^-12 and S6 = sum_{i<j} r^-6.
    # E(s) = 8 s^-12 S12 - 8 s^-6 S6.
    # dE/ds = -96 s^-13 S12 + 48 s^-7 S6 = 0
    # 2 s^-6 S12 = S6
    # s^-6 = S6 / (2 S12)
    
    # My previous sums sum_r12 and sum_r6 are sum_{i!=j} r^-12.
    # So S12 = sum_r12 / 2.
    # S6 = sum_r6 / 2.
    
    if sum_r12 < 1e-12:
        return 1.0
        
    u = (sum_r6 / 2.0) / (2.0 * (sum_r12 / 2.0))
    u = sum_r6 / (2.0 * sum_r12)
    
    if u <= 0:
        return 1.0
        
    s = u ** (-1.0 / 6.0)
    return s

def _relax(pos, max_iter=500, gtol=1e-10, ftol=1e-14):
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
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxfun": max_iter * 100},
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
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    # Build a list of candidate subsets of outer atoms
    # The 24 outer atoms are (211)-type FCC points.
    # For the truncated octahedron, we want a symmetric subset of 19.
    # Strategy: try subsets that exclude pairs of inversion-related atoms.
    
    # Identify inversion pairs: atom i and atom j are inversion-related if outer[i] = -outer[j]
    n_outer = len(outer)
    inversion_partner = np.full(n_outer, -1, dtype=int)
    for i in range(n_outer):
        for j in range(i+1, n_outer):
            if np.allclose(outer[i], -outer[j], atol=1e-10):
                inversion_partner[i] = j
                inversion_partner[j] = i
                break
    
    # Count how many inversion pairs we have
    pairs = []
    for i in range(n_outer):
        if inversion_partner[i] > i:
            pairs.append((i, inversion_partner[i]))
    
    # We need to exclude 5 atoms. To maintain symmetry, exclude complete pairs.
    # But 5 is odd, so we can't exclude only pairs. 
    # Alternative: exclude 4 atoms as 2 pairs + 1 unpaired atom, or just try random.
    # Better: try all ways to exclude 5 atoms, but prioritize symmetric exclusions.
    
    # Generate candidate start configurations
    starts = []
    
    # 1. Deterministic: first 19 outer atoms
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    s_opt = _analytical_optimal_scale(det_start)
    det_start *= s_opt
    starts.append(det_start)
    
    # 2. Systematic: try excluding different sets of 5 outer atoms
    # Prioritize exclusions that maintain some symmetry
    # Generate all C(24,5) = 42504 combinations is too many.
    # Instead, try a structured subset:
    
    # Strategy A: Exclude 5 atoms that are "most peripheral" or break symmetry least
    # For now, let's try a mix of systematic and random
    
    # 2a. Try excluding each single atom with the 4 "closest" others
    # This gives 24 * C(23,4) which is still too many.
    
    # Let's use a different approach: random restarts with good time management
    # Key insight: we need to ensure each relaxation has enough time.
    
    # Estimate time per relaxation
    # A single L-BFGS relaxation takes ~0.5-2 seconds typically
    
    # Generate starts up to a point where we can relax them all
    # Reserve time for: start generation (fast), relaxations, final polish
    
    # Let's generate starts first, then relax them one by one with time checks
    
    # 2. Random subsets - generate a good number
    num_random_starts = 60
    for i in range(num_random_starts):
        idx = rng.choice(n_outer, size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        s_opt = _analytical_optimal_scale(start_pos)
        start_pos *= s_opt
        
        starts.append(start_pos)
    
    # 3. Also try some structured exclusions
    # Exclude atoms that come in inversion pairs - try excluding 2 pairs + 1 atom
    for pair_idx in range(min(10, len(pairs))):
        i, j = pairs[pair_idx]
        # Exclude i, j, and 3 more random atoms from remaining
        remaining = [k for k in range(n_outer) if k != i and k != j]
        if len(remaining) >= 3:
            extra = rng.choice(remaining, size=3, replace=False)
            exclude_set = set([i, j]) | set(extra.tolist())
            include_set = [k for k in range(n_outer) if k not in exclude_set]
            if len(include_set) == 19:
                start_pos = np.vstack([core, outer[include_set]])
                start_pos -= np.mean(start_pos, axis=0)
                s_opt = _analytical_optimal_scale(start_pos)
                start_pos *= s_opt
                starts.append(start_pos)
    
    best_pos = None
    best_energy = np.inf

    # Relax each start, tracking time carefully
    # Reserve last 10 seconds for final polish
    polish_reserve = 10.0
    
    for start in starts:
        elapsed = time.time() - t0
        if elapsed > deadline - polish_reserve:
            break
        try:
            # Two-stage relaxation: first a quick one, then a more careful one
            p, e = _relax(start, max_iter=300, gtol=1e-8)
            
            # If we have time, do a second stage
            elapsed = time.time() - t0
            if e < best_energy and elapsed < deadline - polish_reserve - 5.0:
                p2, e2 = _relax(p, max_iter=500, gtol=1e-10)
                if e2 < e:
                    p, e = p2, e2
            
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            # Use very tight tolerance for final polish
            p, e = _relax(best_pos, max_iter=10000, gtol=1e-15, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    # One more ultra-tight polish if we have time
    remaining = deadline - time.time()
    if remaining > 2.0:
        try:
            p, e = _relax(best_pos, max_iter=20000, gtol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
