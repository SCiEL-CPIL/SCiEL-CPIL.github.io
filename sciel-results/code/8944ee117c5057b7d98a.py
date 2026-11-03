"""Pack 200 equal circles in a unit square via hex-lattice anneal + kinetic eviction local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 150.0


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))


def _hex_lattice_anneal(rng, n_iter=2000):
    """
    Generate a hexagonal lattice packing via simulated anneal on lattice parameters.
    We anneal the lattice spacing and offset to maximize the minimum distance
    of N points sampled from the hex lattice clipped to the square.
    """
    best_c = None
    best_v = 0.0
    
    # Initial parameters: spacing s, offset (ox, oy)
    s = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    ox = 0.0
    oy = 0.0
    
    T = 0.05  # initial temperature
    T_min = 0.001
    
    for it in range(n_iter):
        if time.time() > TIME_LIMIT - 30.0:
            break
        
        # Sample points from current hex lattice
        dx = s
        dy = s * np.sqrt(3.0) / 2.0
        cols = int(np.ceil(1.0 / dx)) + 4
        rows = int(np.ceil(1.0 / dy)) + 4
        xs = np.arange(-3, cols) * dx + ox
        ys = np.arange(-3, rows) * dy + oy
        
        pts = []
        for j, x in enumerate(xs):
            yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
            for i, y in enumerate(ys):
                pts.append((x, y + yoff))
        arr = np.array(pts)
        keep = np.all(np.abs(arr) <= HALF, axis=1)
        arr = arr[keep]
        
        if arr.shape[0] < N:
            continue
        
        # Select N points: use greedy farthest-point
        selected = [int(rng.integers(arr.shape[0]))]
        min_dist = np.full(arr.shape[0], np.inf)
        for _ in range(N - 1):
            last = arr[selected[-1]]
            d = np.linalg.norm(arr - last, axis=1)
            min_dist = np.minimum(min_dist, d)
            j = int(np.argmax(min_dist))
            selected.append(j)
        
        c = arr[selected]
        v = _score(c)
        
        if v > best_v:
            best_v = v
            best_c = c.copy()
        
        # Propose new parameters
        if it % 3 == 0:
            s_new = s * (1.0 + rng.normal(0, T * 0.1))
            s_new = np.clip(s_new, 0.05, 0.2)
            if _score(_quick_hex_score(s_new, ox, oy, rng)) > best_v or rng.random() < np.exp(-T):
                s = s_new
        elif it % 3 == 1:
            ox_new = np.clip(ox + rng.normal(0, T * 0.02), -s, s)
            if _score(_quick_hex_score(s, ox_new, oy, rng)) > best_v or rng.random() < np.exp(-T):
                ox = ox_new
        else:
            oy_new = np.clip(oy + rng.normal(0, T * 0.02), -s, s)
            if _score(_quick_hex_score(s, ox, oy_new, rng)) > best_v or rng.random() < np.exp(-T):
                oy = oy_new
        
        T = max(T_min, T * 0.995)
    
    return best_c, best_v


def _quick_hex_score(s, ox, oy, rng):
    """Quick score estimate for a hex lattice config."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    xs = np.arange(-3, cols) * dx + ox
    ys = np.arange(-3, rows) * dy + oy
    
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x, y + yoff))
    arr = np.array(pts)
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    if arr.shape[0] < N:
        return np.zeros((N, 2))
    
    selected = [int(rng.integers(arr.shape[0]))]
    min_dist = np.full(arr.shape[0], np.inf)
    for _ in range(N - 1):
        last = arr[selected[-1]]
        d = np.linalg.norm(arr - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        selected.append(j)
    
    return arr[selected]


def _eviction_local_search(c, time_budget=60.0):
    """
    Kinetic eviction local search:
    - At each step, identify the circle that is most 'constrained' (smallest clearance)
    - Evict it and re-insert at the position that maximizes the minimum clearance
    - This is a discrete combinatorial move, not a gradient step
    
    Clearance of circle i = min( min_j |c_i - c_j|/2, HALF - max(|c_i|) )
    """
    t0 = time.time()
    c = c.copy().astype(float)
    best_v = _score(c)
    rng = np.random.default_rng(12345)
    
    # Precompute candidate positions for re-insertion
    # Use a fine grid of candidate positions
    n_grid = 30
    grid_ticks = np.linspace(-HALF, HALF, n_grid)
    grid = np.array([(x, y) for y in grid_ticks for x in grid_ticks])
    # Add some random candidates
    n_rand = 200
    rand_cands = rng.uniform(-HALF, HALF, size=(n_rand, 2))
    all_cands = np.vstack([grid, rand_cands])
    
    n_cands = all_cands.shape[0]
    
    # Candidate evaluation: for a candidate position p and existing circles c,
    # compute the new minimum clearance if we place a circle at p
    def _eval_candidate(p, c, exclude_idx=None):
        """Evaluate clearance of placing circle at p, excluding circle at exclude_idx."""
        if exclude_idx is not None:
            others = np.delete(c, exclude_idx, axis=0)
        else:
            others = c
        
        # Distance to all other circles
        d = np.linalg.norm(others - p, axis=1)
        min_d = np.min(d) if d.size > 0 else np.inf
        
        # Wall clearance
        wall = HALF - np.max(np.abs(p))
        
        return min(min_d / 2.0, wall)
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() - t0 < time_budget:
        iteration += 1
        
        # Compute clearance for each circle
        clearances = np.zeros(N)
        for i in range(N):
            others = np.delete(c, i, axis=0)
            d = np.linalg.norm(others - c[i], axis=1)
            min_d = np.min(d) if d.size > 0 else np.inf
            wall = HALF - np.max(np.abs(c[i]))
            clearances[i] = min(min_d / 2.0, wall)
        
        # Evict the most constrained circle (smallest clearance)
        i_evict = int(np.argmin(clearances))
        
        # Find best re-insertion position among candidates
        best_pos = c[i_evict]  # default: keep same position
        best_clear = clearances[i_evict]
        
        # Sample a subset of candidates for speed
        if n_cands > 500:
            sample_idx = rng.choice(n_cands, size=500, replace=False)
            cands_subset = all_cands[sample_idx]
        else:
            cands_subset = all_cands
        
        for p in cands_subset:
            cl = _eval_candidate(p, c, exclude_idx=i_evict)
            if cl > best_clear + 1e-12:
                best_clear = cl
                best_pos = p
        
        # Try the best position
        c_new = c.copy()
        c_new[i_evict] = best_pos
        c_new = np.clip(c_new, -HALF, HALF)
        
        v_new = _score(c_new)
        if v_new > best_v + 1e-12:
            c = c_new
            best_v = v_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, try a random perturbation to escape
            if no_improve_count > 50:
                i_rand = int(rng.integers(N))
                c_new = c.copy()
                c_new[i_rand] += rng.normal(0, 0.02, size=2)
                c_new = np.clip(c_new, -HALF, HALF)
                v_new = _score(c_new)
                if v_new > best_v:
                    c = c_new
                    best_v = v_new
                    no_improve_count = 0
                else:
                    # Accept with some probability (simulated annealing style)
                    if rng.random() < 0.1:
                        c = c_new
                        no_improve_count = 0
    
    return c, best_v


def _gradient_polish(c, time_budget=20.0):
    """
    Gradient-based polish using a smooth penalty objective.
    Minimize: -min_clearance + penalty for violations
    Using a soft-min formulation with L-BFGS-B.
    """
    t0 = time.time()
    c = c.copy().astype(float)
    x0 = c.ravel()
    
    beta = 25.0
    
    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        x = np.clip(x, -HALF, HALF)
        
        diff = x[:, None, :] - x[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1e9)
        pd = d / 2.0
        
        margin = HALF - np.max(np.abs(x), axis=1)
        
        iu = np.triu_indices(N, k=1)
        pd_vals = pd[iu]
        
        all_c = np.concatenate([pd_vals, margin])
        
        # Soft min
        m_c = np.min(all_c)
        exp_c = np.exp(-beta * (all_c - m_c))
        sum_exp = np.sum(exp_c)
        soft_min_val = (m_c + np.log(sum_exp)) / beta
        
        return -soft_min_val
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 2)
        x = np.clip(x, -HALF, HALF)
        
        diff = x[:, None, :] - x[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1e9)
        pd = d / 2.0
        
        margin = HALF - np.max(np.abs(x), axis=1)
        
        iu = np.triu_indices(N, k=1)
        pd_vals = pd[iu]
        
        all_c = np.concatenate([pd_vals, margin])
        
        m_c = np.min(all_c)
        exp_c = np.exp(-beta * (all_c - m_c))
        sum_exp = np.sum(exp_c)
        weights = exp_c / sum_exp
        
        grad_x = np.zeros((N, 2))
        
        # Pairwise gradients
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            if pd[i, j] < 1e-12:
                continue
            diff_ij = x[i] - x[j]
            grad_x[i] += w * diff_ij / (4.0 * pd[i, j])
            grad_x[j] -= w * diff_ij / (4.0 * pd[i, j])
        
        # Margin gradients
        for idx in range(N):
            w = weights[N + idx]
            x_i = x[idx]
            abs_x = np.abs(x_i)
            max_abs = np.max(abs_x)
            if max_abs < 1e-12:
                continue
            k = np.argmax(abs_x)
            sign_val = np.sign(x_i[k])
            grad_x[idx, k] -= w * sign_val
        
        return grad_x.ravel()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        c_opt = result.x.reshape(N, 2)
    except Exception:
        c_opt = c
    
    c_opt = np.clip(c_opt, -HALF, HALF)
    
    v = _score(c_opt)
    if v < _score(c):
        c_opt = c
        v = _score(c)
    
    return c_opt, v


def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 15.0  # safety margin

    rng = np.random.default_rng(2024)

    best_c = None
    best_v = 0.0

    # Phase 1: Hex lattice anneal for diverse initializations
    n_anneal_trials = 8
    for trial in range(n_anneal_trials):
        if time.time() > DEADLINE - 80.0:
            break
        
        # Vary the annealing parameters per trial
        rng_trial = np.random.default_rng(2024 + trial)
        c_trial, v_trial = _hex_lattice_anneal(rng_trial, n_iter=1500 + trial * 200)
        
        if c_trial is not None and v_trial > best_v:
            best_v = v_trial
            best_c = c_trial.copy()

    # Phase 2: Eviction local search on the best configuration
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 40.0:
            polished, pv = _eviction_local_search(best_c, time_budget=min(remaining - 25.0, 60.0))
            if pv > best_v:
                best_c = polished
                best_v = pv

    # Phase 3: Gradient polish for fine-tuning
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 15.0:
            polished2, pv2 = _gradient_polish(best_c, time_budget=min(remaining - 5.0, 20.0))
            if pv2 > best_v:
                best_c = polished2
                best_v = pv2

    # Phase 4: Final quick eviction pass if time permits
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 10.0:
            polished3, pv3 = _eviction_local_search(best_c, time_budget=min(remaining - 5.0, 10.0))
            if pv3 > best_v:
                best_c = polished3
                best_v = pv3

    # Final validity clamp
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    else:
        best_c = np.clip(best_c, -HALF, HALF)

    return best_c
