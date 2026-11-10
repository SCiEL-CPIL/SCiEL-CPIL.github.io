# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        if L < 1e-12:
            n[i] = [1.0, 0.0]
        else:
            n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _outer_side_for_points(pts):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 721):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 620.0  # safety margin before 630s kill

    def _eval_state(state):
        """state is (33,): [x0,y0,a0, x1,y1,a1, ...]
        Returns (1/side, inner, outer, side) or (None, inner, outer, side) if invalid.
        """
        x = state[0::3]
        y = state[1::3]
        a = state[2::3]
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        s, a_out = _outer_side_for_points(pts)
        inner = np.column_stack((x, y, a))
        outer = np.array([cx, cy, a_out])
        if _all_valid(inner, outer, s):
            return 1.0 / s, inner, outer, s
        return None, inner, outer, s

    # Objective for differential evolution: minimize side length
    def _objective(state):
        x = state[0::3]
        y = state[1::3]
        a = state[2::3]
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        s, _ = _outer_side_for_points(pts)
        # Check separation; if any violation, add large penalty
        Vs = [_hex_vertices(x[i], y[i], 1.0, a[i]) for i in range(11)]
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                m, _ = _gap_and_margin(Vs[i], Vs[j])
                if m <= 1e-6:
                    pen += (1e-6 - m) ** 2
        # Check containment
        cx_v, cy_v = x.mean(), y.mean()
        # Use the best outer orientation for containment check
        _, a_out = _outer_side_for_points(pts)
        OV = _hex_vertices(cx_v, cy_v, s, a_out)
        for V in Vs:
            for k in range(6):
                p1 = OV[k]
                p2 = OV[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
                if np.any(cross < -1e-6):
                    pen += (cross[cross < -1e-6].min() + 1e-6) ** 2
        return s + 1000.0 * pen

    # Seed arrangements
    def _seed_hex():
        return np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [1.0, 1.7320508, 0.0],
            [-1.0, 1.7320508, 0.0],
            [1.0, -1.7320508, 0.0],
            [-1.0, -1.7320508, 0.0],
            [3.0, 1.7320508, 0.0],
            [-3.0, 1.7320508, 0.0],
            [3.0, -1.7320508, 0.0],
            [-3.0, -1.7320508, 0.0],
        ])

    base = _seed_hex()
    best_val = -np.inf
    best_state = None

    # Evaluate seed
    seed_state = np.concatenate([base[:, 0], base[:, 1], base[:, 2]])
    val, inner, outer, s = _eval_state(seed_state)
    if val is not None:
        best_val = val
        best_state = (inner, outer, s)
    print(f"Seed side: {s:.6f}, 1/s: {1/s:.6f}", flush=True)

    # Try multiple random perturbations as additional starts for DE
    rng = np.random.default_rng(42)
    
    # Define bounds for DE
    # Positions: roughly within [-5, 5] for x and y
    # Angles: [-60, 60] degrees
    bounds = [(-6.0, 6.0)] * 11 + [(-6.0, 6.0)] * 11 + [(-60.0, 60.0)] * 11
    
    # Run differential evolution with multiple random inits
    n_de_runs = 5
    max_time_per_de = (deadline - time.time()) / max(1, n_de_de_runs) if n_de_runs > 0 else deadline - time.time()
    
    for run_idx in range(n_de_runs):
        if time.time() > deadline - 30.0:
            break
        
        # Random initial population
        x0 = base[:, 0] + rng.normal(0, 0.5, size=11)
        y0 = base[:, 1] + rng.normal(0, 0.5, size=11)
        a0 = rng.uniform(-30, 30, size=11)
        init_state = np.concatenate([x0, y0, a0])
        
        # Run DE
        de_result = differential_evolution(
            _objective, bounds, x0=init_state,
            seed=42 + run_idx, maxiter=200, tol=1e-8,
            polish=False, workers=-1, popsize=15,
            mutation=(0.3, 1.2), recombination=0.7,
            init='sobol'
        )
        
        if time.time() > deadline - 15.0:
            break
        
        st = de_result.x
        val, inner, outer, s = _eval_state(st)
        if val is not None and val > best_val:
            best_val = val
            best_state = (inner, outer, s)
            print(f"DE run {run_idx}: side={s:.6f}, 1/s={val:.6f}", flush=True)
        
        # Also try polishing with L-BFGS-B
        if time.time() > deadline - 15.0:
            break
        
        from scipy.optimize import minimize
        
        def _obj_lbfgs(state):
            return _objective(state)
        
        res = minimize(_obj_lbfgs, st, jac=True, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if time.time() > deadline - 10.0:
            break
        
        st2 = res.x
        val2, inner2, outer2, s2 = _eval_state(st2)
        if val2 is not None and val2 > best_val:
            best_val = val2
            best_state = (inner2, outer2, s2)
            print(f"LBFGS after DE run {run_idx}: side={s2:.6f}, 1/s={val2:.6f}", flush=True)
    
    # Try a few more local refinements from the best state
    if best_state is not None:
        inner_best, outer_best, side_best = best_state
        # Convert back to state format
        x_best = inner_best[:, 0] - outer_best[0]  # center-relative
        y_best = inner_best[:, 1] - outer_best[1]
        a_best = inner_best[:, 2]
        
        # Try small perturbations
        for perturb_scale in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 10.0:
                break
            for _ in range(3):
                dx = rng.normal(0, perturb_scale, size=11)
                dy = rng.normal(0, perturb_scale, size=11)
                da = rng.uniform(-perturb_scale * 10, perturb_scale * 10, size=11)
                
                x_try = x_best + dx
                y_try = y_best + dy
                a_try = a_best + da
                
                state_try = np.concatenate([x_try + outer_best[0], y_try + outer_best[1], a_try])
                val_try, inner_try, outer_try, s_try = _eval_state(state_try)
                if val_try is not None and val_try > best_val:
                    best_val = val_try
                    best_state = (inner_try, outer_try, s_try)
                    print(f"Refinement: side={s_try:.6f}, 1/s={val_try:.6f}", flush=True)
    
    if best_state is None:
        # Fallback loose packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    inner, outer, s = best_state
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
