import time
import numpy as np

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    r8 = r2 ** 4
    r14 = r2 ** 7

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

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

def _basin_hopping(pos, t0, deadline, rng):
    """
    Basin hopping with adaptive step size.
    Returns (best_pos, best_energy).
    """
    best_pos, best_energy = pos.copy(), _lj_energy_and_grad(pos)[0]
    current_pos, current_energy = pos.copy(), best_energy
    
    step_size = 0.5
    min_step = 0.05
    max_step = 1.5
    
    # Simple annealing schedule
    # Start with high T, decrease
    T = 0.5
    
    n_iter = 0
    max_iter = 200 # Hard cap on iterations to prevent infinite loops if deadline logic fails
    
    while time.time() < deadline - 5.0 and n_iter < max_iter:
        n_iter += 1
        
        # Perturb
        noise = rng.normal(0, step_size, size=pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-8)
        except Exception:
            relaxed, e_relaxed = perturbed, np.inf
            
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            best_pos = relaxed.copy()
            
        # Metropolis criterion
        delta_e = e_relaxed - current_energy
        if delta_e < 0:
            current_pos = relaxed
            current_energy = e_relaxed
            # If we improved, maybe increase step size slightly or keep it
            # step_size = min(max_step, step_size * 1.05)
        else:
            if np.log(rng.uniform()) < -delta_e / T:
                current_pos = relaxed
                current_energy = e_relaxed
            else:
                # Rejected, maybe decrease step size
                pass
                
        # Cooling
        T *= 0.98
        
        # Adaptive step size: if we've been rejecting a lot, shrink
        # (Simplified: just shrink slightly every few steps)
        if n_iter % 10 == 0:
            step_size = max(min_step, step_size * 0.95)

    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s buffer for 150s limit
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
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

    # 3. High-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 4. Basin Hopping
    # Use multiple seeds to explore different basins
    # The FCC global minimum is known, but BH helps escape icosahedral traps
    # if the initial FCC was slightly off or if we want to be sure.
    # However, for LJ38, FCC is the global min. BH from FCC might wander into
    # icosahedral structures if not careful, but Metropolis + cooling should
    # keep us near the global min if we start there.
    # Actually, starting from FCC and doing BH is risky because FCC is the global min.
    # But if the initial relaxation got stuck in a distorted FCC local min (which is
    # possible if the initial geometry wasn't perfect), BH can help.
    
    # Let's run a few BH chains from slightly perturbed FCCs to ensure we are in the
    # global basin.
    
    rng = np.random.default_rng(42)
    
    # Run 1: From the relaxed FCC
    if time.time() < deadline - 20.0:
        try:
            p_bh, e_bh = _basin_hopping(best_pos, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass

    # Run 2: From a random perturbation of FCC (to escape local distortions)
    if time.time() < deadline - 20.0:
        try:
            perturbed_start = best_pos + 0.1 * rng.standard_normal(best_pos.shape)
            perturbed_start = _ensure_min_sep(perturbed_start)
            p_bh, e_bh = _basin_hopping(perturbed_start, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass
            
    # Run 3: From a different random start (maybe icosahedral-ish?)
    # Just to be safe, try a random cluster start
    if time.time() < deadline - 30.0:
        try:
            # Random cluster within a sphere
            random_pos = rng.uniform(-2, 2, size=(N, 3))
            random_pos = _ensure_min_sep(random_pos)
            p_bh, e_bh = _basin_hopping(random_pos, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass

    # 5. Final check and separation enforcement
    best_pos = _ensure_min_sep(best_pos)
    
    # 6. Final high-precision polish on the absolute best
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
                best_pos = _ensure_min_sep(best_pos)
        except Exception:
            pass

    return best_pos
