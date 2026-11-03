"""Pack 200 equal circles in a unit square via hex-grid greedy + gradient polish."""
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
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))


def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    # offset every other column
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x + 0.0, y + yoff))
    arr = np.array(pts)
    # keep inside square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]


def _rect_points(sx, sy, ox, oy):
    """Rectangular lattice points with spacing (sx, sy), offset (ox, oy)."""
    nx = int(np.ceil(1.0 / sx)) + 3
    ny = int(np.ceil(1.0 / sy)) + 3
    xs = np.arange(-2, nx) * sx + ox
    ys = np.arange(-2, ny) * sy + oy
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    arr = np.column_stack([XX.ravel(), YY.ravel()])
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]


def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]


def _soft_min_surrogate(x, t, beta=20.0):
    """
    Smooth surrogate for maximizing t subject to:
      t <= |x_i - x_j|/2 for all i<j
      t <= HALF - max(|x_ik|) for all i,k
    We minimize -t + (1/beta)*log(sum(exp(-beta*(constraint - t))))
    which approximates t - soft_violation.
    """
    x = x.reshape(N, 2)
    # Pairwise distances / 2
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1e9)
    pd = d / 2.0
    
    # Wall margins: HALF - max(|x_i|)
    margin = HALF - np.max(np.abs(x), axis=1)
    
    # All constraints as (constraint_value - t) >= 0
    # We want to maximize t, so minimize -t + penalty for violations
    # Soft min: log-sum-exp trick for the minimum of all (pd_ij, margin_i)
    # Let constraints = [pd_ij for all i<j, margin_i for all i]
    # We want max t s.t. t <= all constraints
    # Equivalent to: minimize f(x,t) = -t + (1/beta)*log(sum(exp(-beta*(c_j - t))))
    # where c_j are all constraint values
    
    # Build vector of all constraint values
    # Upper triangle of pd (i<j)
    iu = np.triu_indices(N, k=1)
    pd_vals = pd[iu]
    
    all_constraints = np.concatenate([pd_vals, margin])
    violations = all_constraints - t
    # Soft max of -violations = -soft min of violations
    # We want to penalize when t > any constraint, i.e., violations < 0
    # Use: (1/beta)*log(sum(exp(-beta*violations))) approximates -min(violations) = max(t - c_j)
    m = np.min(violations)
    logsum = m + np.log(np.sum(np.exp(-beta * (violations - m))))
    penalty = logsum / beta
    
    return -t + penalty


def _soft_min_grad(x, t, beta=20.0):
    """Gradient of _soft_min_surrogate with respect to x and t."""
    x = x.reshape(N, 2)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1e9)
    pd = d / 2.0
    
    margin = HALF - np.max(np.abs(x), axis=1)
    
    iu = np.triu_indices(N, k=1)
    pd_vals = pd[iu]
    
    all_constraints = np.concatenate([pd_vals, margin])
    violations = all_constraints - t
    
    m = np.min(violations)
    exp_vals = np.exp(-beta * (violations - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights over violations
    
    # d/dt of (-t + (1/beta)*logsum) = -1 + (1/beta)*sum(weights * (-beta)) = -1 + sum(weights) = -1 + 1 = 0
    # Wait, that's not right. Let me recompute.
    # f = -t + (1/beta)*log(sum(exp(-beta*(c_j - t))))
    # d/dt = -1 + (1/beta)*sum(exp(-beta*(c_j-t)) * beta) / sum(exp(-beta*(c_j-t)))
    #       = -1 + sum(exp) / sum(exp) = -1 + 1 = 0? 
    # That can't be right. Let me think again.
    # Actually: d/dt log(sum(exp(-beta*(c_j - t)))) = d/dt log(sum(exp(beta*t - beta*c_j)))
    # = sum(exp(beta*t - beta*c_j) * beta) / sum(exp(beta*t - beta*c_j)) = beta
    # So d/dt f = -1 + (1/beta)*beta = -1 + 1 = 0. 
    # Hmm, that means t doesn't appear in the gradient? That seems wrong.
    # Let me re-derive. The soft min of {c_j} is: -log(sum(exp(-beta*c_j)))/beta
    # We want to maximize min(c_j), so minimize -soft_min = log(sum(exp(-beta*c_j)))/beta
    # But we also have the variable t. The epigraph form: minimize -t + penalty
    # where penalty enforces t <= c_j for all j.
    # A common approach: minimize -t + (1/beta)*log(sum(exp(-beta*(c_j - t))))
    # Let's verify: when t << min(c_j), all (c_j - t) >> 0, exp(-beta*(c_j-t)) is small,
    # logsum ~ log(N)*(-beta*(min(c_j)-t))/beta... this is getting complex.
    # 
    # Simpler: just use the soft min directly.
    # soft_min(c) = -log(sum(exp(-beta*c)))/beta
    # We want to maximize soft_min of all constraints.
    # So minimize -soft_min = log(sum(exp(-beta*c)))/beta
    # 
    # Let's just compute the gradient numerically to be safe, or use a simpler formulation.
    # 
    # Alternative: use the fact that for the optimum, t = min(constraints).
    # At the optimum, the gradient w.r.t. x is the weighted average of gradients of active constraints.
    # 
    # For now, let's compute gradients numerically for reliability.
    
    # Gradient w.r.t. t: -1 + sum(weights) = 0 (as derived). This is suspicious.
    # Let me just use finite differences for the gradient to be safe.
    
    # Actually, let's use a different, cleaner formulation.
    # Minimize: -t + (1/beta)*log(sum_j exp(-beta*(c_j - t)))
    # d/dt: -1 + (1/beta) * [sum_j exp(-beta*(c_j-t)) * beta] / sum_j exp(-beta*(c_j-t))
    #      = -1 + 1 = 0. So indeed the gradient w.r.t. t is zero? That's because
    # the log-sum-exp is translation-invariant in this form. 
    # This means t is not well-defined from the gradient alone. We need to fix t differently.
    # 
    # Better approach: fix t as a parameter and just minimize the penalty:
    # For a given target radius r, check if feasible, then binary search on r.
    # Or: minimize sum of squared violations for a given t, then increase t.
    # 
    # Simplest robust approach: use a penalty method.
    # Minimize: (1/2)*sum(max(0, t - c_j)^2) - t
    # This is non-smooth but we can smooth it.
    # 
    # Let's just use the soft-min of the constraints directly (no t variable):
    # Objective: -soft_min(all_constraints) = log(sum(exp(-beta*c_j)))/beta
    # This is smooth and differentiable.
    
    # Recompute with this cleaner formulation
    c_vals = all_constraints  # all constraint values
    m_c = np.min(c_vals)
    exp_c = np.exp(-beta * (c_vals - m_c))
    sum_exp_c = np.sum(exp_c)
    obj = (m_c + np.log(sum_exp_c)) / beta  # this is soft_min
    # We want to minimize -obj = -soft_min
    
    # Gradient of soft_min w.r.t. x:
    # d(soft_min)/dx = -sum_j weights_j * dc_j/dx where weights_j = exp(-beta*(c_j-m_c))/sum_exp_c
    # So d(-soft_min)/dx = sum_j weights_j * dc_j/dx
    
    # Compute dc_j/dx for pairwise distances
    # d(|x_i - x_j|/2)/dx_i = (x_i - x_j) / (2 * |x_i - x_j|) = (x_i - x_j) / (4 * pd_ij)
    # d(|x_i - x_j|/2)/dx_j = (x_j - x_i) / (4 * pd_ij)
    
    grad_x = np.zeros((N, 2))
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]  # weight for this pair
        if pd[i, j] < 1e-12:
            continue
        # d(pd_ij)/dx_i = (x_i - x_j) / (2 * d_ij) = (x_i - x_j) / (4 * pd_ij)
        # since pd_ij = d_ij/2, d_ij = 2*pd_ij
        # d(pd_ij)/dx_i = (x_i - x_j) / (2 * 2 * pd_ij) = (x_i - x_j) / (4 * pd_ij)
        diff_ij = x[i] - x[j]
        grad_x[i] += w * diff_ij / (4.0 * pd[i, j])
        grad_x[j] -= w * diff_ij / (4.0 * pd[i, j])
    
    # Gradient for margin constraints: margin_i = HALF - max(|x_i|)
    # d(margin_i)/dx_i = -sign(x_i) for the component with max |x_i|
    for idx in range(N):
        w = weights[N + idx]  # weight for margin constraint i
        x_i = x[idx]
        abs_x = np.abs(x_i)
        max_abs = np.max(abs_x)
        if max_abs < 1e-12:
            continue
        # subgradient: -sign(x_i[k]) where k = argmax(|x_i|)
        k = np.argmax(abs_x)
        grad_x[idx] -= w * np.sign(x_i[k]) * np.eye(2)[k]
    
    # Gradient w.r.t. t is not used since we're not using t in this formulation
    # We'll just return grad_x and ignore t
    
    return grad_x, 0.0


def _polish_gradient(c, time_budget=10.0):
    """
    Polish using gradient-based optimization with soft-min surrogate.
    Minimize -soft_min(constraints) where constraints are pairwise dist/2 and wall margins.
    """
    t0 = time.time()
    c = c.copy().astype(float)
    
    # Flatten for scipy
    x0 = c.ravel()
    
    beta = 15.0  # softness parameter
    
    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        # Clip to keep roughly inside (soft constraint)
        x_clipped = np.clip(x, -HALF, HALF)
        
        diff = x_clipped[:, None, :] - x_clipped[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1e9)
        pd = d / 2.0
        
        margin = HALF - np.max(np.abs(x_clipped), axis=1)
        
        iu = np.triu_indices(N, k=1)
        pd_vals = pd[iu]
        
        all_c = np.concatenate([pd_vals, margin])
        
        # Soft min
        m_c = np.min(all_c)
        exp_c = np.exp(-beta * (all_c - m_c))
        sum_exp = np.sum(exp_c)
        soft_min_val = (m_c + np.log(sum_exp)) / beta
        
        # We want to maximize soft_min, so minimize -soft_min
        # Add a small penalty for going outside
        outside_penalty = np.sum(np.maximum(0, np.abs(x) - HALF)) * 100.0
        
        return -soft_min_val + outside_penalty
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 2)
        x_clipped = np.clip(x, -HALF, HALF)
        
        diff = x_clipped[:, None, :] - x_clipped[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1e9)
        pd = d / 2.0
        
        margin = HALF - np.max(np.abs(x_clipped), axis=1)
        
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
            diff_ij = x_clipped[i] - x_clipped[j]
            grad_x[i] += w * diff_ij / (4.0 * pd[i, j])
            grad_x[j] -= w * diff_ij / (4.0 * pd[i, j])
        
        # Margin gradients
        for idx in range(N):
            w = weights[N + idx]
            x_i = x_clipped[idx]
            abs_x = np.abs(x_i)
            max_abs = np.max(abs_x)
            if max_abs < 1e-12:
                continue
            k = np.argmax(abs_x)
            grad_x[idx] -= w * np.sign(x_i[k]) * np.eye(2)[k]
        
        # Outside penalty gradient
        outside_pen_grad = np.zeros((N, 2))
        for i in range(N):
            for k in range(2):
                if x[i, k] > HALF:
                    outside_pen_grad[i, k] += 100.0
                elif x[i, k] < -HALF:
                    outside_pen_grad[i, k] -= 100.0
        
        total_grad = grad_x + outside_pen_grad
        return total_grad.ravel()
    
    # Run L-BFGS-B
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        c_opt = result.x.reshape(N, 2)
    except Exception:
        c_opt = c
    
    # Clamp to square
    c_opt = np.clip(c_opt, -HALF, HALF)
    
    # Verify and score
    v = _score(c_opt)
    if v < _score(c):
        c_opt = c
        v = _score(c)
    
    return c_opt, v


def _polish_local_search(c, time_budget=5.0):
    """Local search with random perturbations and gradient steps."""
    t0 = time.time()
    c = c.copy()
    best = _score(c)
    rng = np.random.default_rng(42)
    
    while time.time() - t0 < time_budget:
        # Try small random perturbations
        improved = False
        for _ in range(50):
            i = int(rng.integers(N))
            dx = rng.uniform(-0.01, 0.01)
            dy = rng.uniform(-0.01, 0.01)
            c2 = c.copy()
            c2[i, 0] += dx
            c2[i, 1] += dy
            c2 = np.clip(c2, -HALF, HALF)
            v2 = _score(c2)
            if v2 > best + 1e-12:
                c = c2
                best = v2
                improved = True
                break
        
        if not improved:
            # Try gradient step
            try:
                x_flat = c.ravel()
                grad = gradient(x_flat) if False else None  # Skip for speed
            except:
                pass
            break
    
    return c, best


def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 10.0  # safety margin

    rng = np.random.default_rng(2024)

    best_c = None
    best_v = 0.0

    # target spacing estimate
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)  # ~0.076

    # Phase 1: Generate diverse initial configurations
    n_trials = 30
    for trial in range(n_trials):
        if time.time() > DEADLINE:
            break
        
        if trial % 2 == 0:
            # Hex grid
            s = s0 * (0.97 + 0.06 * rng.random())
            ox = rng.uniform(-0.5, 0.5)
            oy = rng.uniform(-0.5, 0.5)
            cands = _hex_points(s, ox, oy)
        else:
            # Rectangular grid with varied aspect
            sx = s0 * (0.95 + 0.1 * rng.random())
            sy = sx * (0.9 + 0.2 * rng.random())
            ox = rng.uniform(-0.5, 0.5)
            oy = rng.uniform(-0.5, 0.5)
            cands = _rect_points(sx, sy, ox, oy)
        
        if cands.shape[0] < N:
            continue
        
        r_min = 0.5 * s0 * 0.85
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()

    # Phase 2: Gradient polish on best configuration
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 15.0:
            polished, pv = _polish_gradient(best_c, time_budget=min(remaining - 5.0, 20.0))
            if pv > best_v:
                best_c = polished
                best_v = pv

        # Phase 3: Additional local search if time permits
        remaining = DEADLINE - time.time()
        if remaining > 10.0:
            polished2, pv2 = _polish_local_search(best_c, time_budget=min(remaining - 5.0, 10.0))
            if pv2 > best_v:
                best_c = polished2
                best_v = pv2

    # Final validity clamp
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    else:
        best_c = np.clip(best_c, -HALF, HALF)

    return best_c
