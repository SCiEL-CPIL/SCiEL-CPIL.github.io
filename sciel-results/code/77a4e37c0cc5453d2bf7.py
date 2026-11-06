"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    wall_min = wall.min()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    return min(wall_min, 0.5 * nn)


def _make_hex(rows, spacing, dy, xoff):
    """Build a hex lattice with the given row lengths."""
    pts = []
    h = spacing * np.sqrt(3.0) / 2.0
    for i, n in enumerate(rows):
        y = -dy + i * h
        base = (0.5 if (n % 2 == 1) else 0.0) * spacing
        xs = np.arange(n) * spacing + base
        xs = xs - (xs.max() + xs.min()) / 2.0 + xoff
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _neg_score(params, rows):
    spacing, dy, xoff = params
    if spacing <= 0:
        return 1.0
    pts = _make_hex(rows, spacing, dy, xoff)
    return -_score(pts)


def _softmin_obj(x, n, beta):
    """
    Epigraph-style smooth objective: minimize -softmin(constraints).
    x is flattened (n,2) positions. Returns scalar to minimize.
    """
    p = x.reshape(n, 2)
    diff = p[:, None, :] - p[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    wall = np.minimum(
        np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
        np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5),
    )
    
    half_d = d / 2.0
    mask = np.isfinite(half_d)
    half_d_flat = half_d[mask]
    all_c = np.concatenate([half_d_flat, wall])
    
    # Soft-min via log-sum-exp for numerical stability
    min_c = np.min(all_c)
    shifted = all_c - min_c
    shifted = np.clip(shifted, 0.0, 50.0)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    softmin_val = min_c - log_sum / beta
    
    # Barrier for out-of-bounds
    barrier = 0.0
    out_lo = np.minimum(0.0, p + 0.495)
    barrier += np.sum(out_lo ** 2) * 100.0
    out_hi = np.minimum(0.0, 0.495 - p)
    barrier += np.sum(out_hi ** 2) * 100.0
    
    return -softmin_val + barrier


def _softmin_grad(x, n, beta):
    """Numerical gradient of _softmin_obj."""
    eps = 1e-7
    g = np.zeros_like(x)
    f0 = _softmin_obj(x, n, beta)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_softmin_obj(xp, n, beta) - f0) / eps
    return g


def _full_optimize(pts, max_iter=300, deadline=None, beta_start=20.0, beta_end=100.0, n_phases=4):
    """
    Multi-phase smooth optimization with increasing beta.
    """
    pts = pts.copy()
    n = len(pts)
    x = pts.flatten()
    
    for phase in range(n_phases):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        beta = beta_start + (beta_end - beta_start) * phase / max(1, n_phases - 1)
        
        for restart in range(2):
            if deadline is not None and time.time() > deadline - 5.0:
                break
            if restart == 0:
                x_init = x.copy()
            else:
                x_init = x + np.random.normal(0, 0.003, len(x))
                x_init = np.clip(x_init, -0.49, 0.49)
            
            try:
                res = minimize(_softmin_obj, x_init, args=(n, beta), method="L-BFGS-B",
                               jac=lambda xx, nn=n, bb=beta: _softmin_grad(xx, nn, bb),
                               options={"maxiter": max_iter, "gtol": 1e-12,
                                        "ftol": 1e-15, "maxls": 40})
                if res.fun < _softmin_obj(x, n, beta):
                    x = res.x
            except Exception:
                pass
    
    return x.reshape(n, 2)


def _refine_free(pts, deadline=None):
    """
    Refine positions freely with multi-phase smooth optimization.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Multiple beta schedules
    schedules = [
        (15.0, 60.0, 4),
        (20.0, 80.0, 4),
        (30.0, 120.0, 5),
    ]
    for beta_s, beta_e, nph in schedules:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        res_pts = _full_optimize(best_pts, max_iter=200, deadline=deadline,
                                 beta_start=beta_s, beta_end=beta_e, n_phases=nph)
        s = _score(res_pts)
        if s > best_score:
            best_score = s
            best_pts = res_pts
    
    return best_pts, best_score


def _hill_climb(pts, max_iter=500, step_size=0.001, deadline=None):
    """Local hill-climbing: perturb one circle at a time to improve min distance."""
    pts = pts.copy()
    current_score = _score(pts)
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        idx = np.random.randint(len(pts))
        orig = pts[idx].copy()
        improved = False
        
        for attempt in range(40):
            ss = step_size * (1.0 if attempt < 25 else 0.5)
            new_pos = orig + np.random.normal(0, ss, 2)
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
            pts[idx] = new_pos
            new_score = _score(pts)
            if new_score > current_score + 1e-11:
                current_score = new_score
                improved = True
                no_improve = 0
                break
            else:
                pts[idx] = orig
        
        if not improved:
            no_improve += 1
            if no_improve > 60:
                step_size *= 0.8
                no_improve = 0
                if step_size < 1e-6:
                    break
        else:
            if it % 150 == 0:
                step_size *= 1.15
    
    return pts, current_score


def _soft_repulsion_force(pts, r_target, k_rep=100.0, k_wall=50.0):
    """Compute repulsion forces between circles and wall forces."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                forces[i] += np.random.normal(0, 1e-4, 2)
                forces[j] -= forces[i]
                continue
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                push = k_rep * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
            elif dist < 2.5 * r_target:
                overlap = 2.5 * r_target - dist
                push = k_rep * 0.1 * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
    
    for i in range(n):
        for k in range(2):
            if pts[i][k] < -0.5 + r_target:
                forces[i][k] += k_wall * (-0.5 + r_target - pts[i][k])
            elif pts[i][k] > 0.5 - r_target:
                forces[i][k] -= k_wall * (pts[i][k] - (0.5 - r_target))
    
    return forces


def _relax(pts, r_target, max_iter=1500, dt=0.001, deadline=None):
    """Relax positions to eliminate overlaps at given radius."""
    pts = pts.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        forces = _soft_repulsion_force(pts, r_target)
        pts += dt * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        max_overlap = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 2 * r_target:
                    max_overlap = max(max_overlap, 2 * r_target - dist)
            for k in range(2):
                if pts[i][k] < -0.5 + r_target:
                    max_overlap = max(max_overlap, -0.5 + r_target - pts[i][k])
                elif pts[i][k] > 0.5 - r_target:
                    max_overlap = max(max_overlap, pts[i][k] - (0.5 - r_target))
        
        if max_overlap < 1e-7:
            break
    
    return pts


def _differential_evolution_lattice(rows, deadline, pop_size=30, max_gen=50):
    """
    Differential evolution over lattice parameters (spacing, dy, xoff).
    """
    max_n = max(rows)
    bounds = [(0.05, 1.0 / (max_n * 0.8)), (-0.15, 0.15), (-0.1, 0.1)]
    
    # Initialize population
    pop = np.random.uniform(bounds[0][0], bounds[0][1], pop_size)
    pop_dy = np.random.uniform(bounds[1][0], bounds[1][1], pop_size)
    pop_xoff = np.random.uniform(bounds[2][0], bounds[2][1], pop_size)
    
    # Add a good initial guess
    pop[0] = 1.0 / max_n
    pop_dy[0] = 0.0
    pop_xoff[0] = 0.0
    
    best_params = None
    best_score = -1.0
    
    for gen in range(max_gen):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        # Evaluate population
        scores = np.zeros(pop_size)
        for i in range(pop_size):
            pts = _make_hex(rows, pop[i], pop_dy[i], pop_xoff[i])
            scores[i] = _score(pts)
        
        # Track best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = [pop[best_idx], pop_dy[best_idx], pop_xoff[best_idx]]
        
        # Mutation and crossover
        new_pop = pop.copy()
        new_dy = pop_dy.copy()
        new_xoff = pop_xoff.copy()
        
        F = 0.8
        CR = 0.7
        
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline - 10.0:
                break
            a, b, c = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            
            # Mutation
            mutant_s = pop[a] + F * (pop[b] - pop[c])
            mutant_dy = pop_dy[a] + F * (pop_dy[b] - pop_dy[c])
            mutant_xoff = pop_xoff[a] + F * (pop_xoff[b] - pop_xoff[c])
            
            # Clip to bounds
            mutant_s = np.clip(mutant_s, bounds[0][0], bounds[0][1])
            mutant_dy = np.clip(mutant_dy, bounds[1][0], bounds[1][1])
            mutant_xoff = np.clip(mutant_xoff, bounds[2][0], bounds[2][1])
            
            # Crossover
            trial_s = mutant_s if np.random.random() < CR else pop[i]
            trial_dy = mutant_dy if np.random.random() < CR else pop_dy[i]
            trial_xoff = mutant_xoff if np.random.random() < CR else pop_xoff[i]
            
            # Evaluate trial
            pts_trial = _make_hex(rows, trial_s, trial_dy, trial_xoff)
            score_trial = _score(pts_trial)
            
            if score_trial > scores[i]:
                new_pop[i] = trial_s
                new_dy[i] = trial_dy
                new_xoff[i] = trial_xoff
                scores[i] = score_trial
                if score_trial > best_score:
                    best_score = score_trial
                    best_params = [trial_s, trial_dy, trial_xoff]
        
        pop = new_pop
        pop_dy = new_dy
        pop_xoff = new_xoff
    
    return best_params, best_score


def _high_precision_polish(pts, deadline):
    """
    Final high-precision polish: alternate between smooth optimization
    with high beta and hill climbing.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for round_i in range(3):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Smooth optimization with high beta
        polished = _full_optimize(best_pts, max_iter=200, deadline=deadline,
                                   beta_start=50.0, beta_end=200.0, n_phases=3)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_pts = polished
        
        # Hill climb polish
        hc_pts, hc_score = _hill_climb(best_pts, max_iter=300, step_size=0.0005, deadline=deadline)
        if hc_score > best_score:
            best_score = hc_score
            best_pts = hc_pts
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    best_pts = None
    best_score = -1.0

    # Candidate row-length patterns that sum to 50.
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [8, 6, 7, 6, 7, 6, 7, 3],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [5, 7, 6, 7, 6, 7, 6, 6],
        [6, 6, 7, 6, 7, 6, 7, 5],
        [7, 6, 6, 7, 6, 7, 6, 5],
        [6, 7, 7, 6, 6, 7, 6, 5],
        [7, 5, 7, 6, 7, 6, 7, 5],
        [6, 8, 6, 7, 6, 7, 6, 4],
    ]

    for rows in patterns:
        if sum(rows) != N:
            continue
        if time.time() > deadline - 20.0:
            break

        max_n = max(rows)
        
        # Phase 1: Nelder-Mead on lattice params
        init = [1.0 / max_n, 0.0, 0.0]
        def obj(p, rows=rows):
            return _neg_score(p, rows)
        
        res = minimize(obj, init, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-10,
                                "adaptive": True})
        
        s = -res.fun
        if s > best_score:
            best_score = s
            best_pts = _make_hex(rows, *res.x)
        
        # Phase 2: Differential evolution for global search
        if time.time() < deadline - 15.0:
            de_params, de_score = _differential_evolution_lattice(rows, deadline=deadline, pop_size=20, max_gen=30)
            if de_params is not None and de_score > best_score:
                best_score = de_score
                best_pts = _make_hex(rows, *de_params)
        
        # Phase 3: Local polish with L-BFGS-B on lattice params
        for xoff0 in (-0.015, 0.0, 0.015):
            for dy0 in (-0.008, 0.0, 0.008):
                if time.time() > deadline - 15.0:
                    break
                base_params = de_params if (de_params is not None and de_score > s) else res.x
                init2 = [base_params[0], dy0, xoff0]
                try:
                    res2 = minimize(obj, init2, method="L-BFGS-B",
                                    options={"maxiter": 300, "gtol": 1e-10})
                    s2 = -res2.fun
                    if s2 > best_score:
                        best_score = s2
                        best_pts = _make_hex(rows, *res2.x)
                except Exception:
                    pass

    # Free-position refinement with multi-phase smooth surrogate
    if best_pts is not None and time.time() < deadline - 30.0:
        refined_pts, refined_score = _refine_free(best_pts, deadline=deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts

    # Physics-based radius growth
    if best_pts is not None and time.time() < deadline - 20.0:
        r = best_score
        pts = best_pts.copy()
        
        for grow_iter in range(50):
            if time.time() > deadline - 15.0:
                break
            r_new = r * 1.002 + 1e-8
            pts_relaxed = _relax(pts, r_new, max_iter=1000, dt=0.0005, deadline=deadline)
            s_new = _score(pts_relaxed)
            if s_new >= r_new - 1e-8:
                r = r_new
                pts = pts_relaxed
                if r > best_score:
                    best_score = r
                    best_pts = pts.copy()
            else:
                r_new = r * 1.001 + 1e-9
                pts_relaxed = _relax(pts, r_new, max_iter=500, dt=0.0005, deadline=deadline)
                s_new = _score(pts_relaxed)
                if s_new >= r_new - 1e-9:
                    r = r_new
                    pts = pts_relaxed
                    if r > best_score:
                        best_score = r
                        best_pts = pts.copy()
                else:
                    break

    # High-precision polish: alternating smooth opt + hill climb
    if best_pts is not None and time.time() < deadline - 10.0:
        final_pts, final_score = _high_precision_polish(best_pts, deadline=deadline)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final hill-climb
    if best_pts is not None and time.time() < deadline - 5.0:
        pts_polished, s_polished = _hill_climb(best_pts, max_iter=400, step_size=0.0008, deadline=deadline)
        if s_polished > best_score:
            best_score = s_polished
            best_pts = pts_polished

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
# EVOLVE-BLOCK-END
