"""Rewritten: epigraph convex formulation (min t s.t. C_k <= t) + local refinement."""
import time
import numpy as np

N = 100
DEADLINE = 140.0  # safety margin under 150s


def _value(h):
    h = np.asarray(h)
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def _corr_vec(h):
    """Return the full correlation vector C_k = sum_i h_i * (1 - h_{i+k})."""
    h = np.asarray(h, dtype=float)
    return np.correlate(h, 1.0 - h, mode="full")


def _epigraph_objective(z):
    """
    z = [h_0, ..., h_99, t]
    Objective: t (the last element)
    """
    return z[N]


def _epigraph_constraints(z):
    """
    Constraints:
    - h_i in [0, 1] for all i (handled via bounds)
    - sum(h) = 50 (equality)
    - C_k <= t for all lags k (inequality, 2N-1 of them)
    """
    h = z[:N]
    t = z[N]
    c = _corr_vec(h)
    # C_k - t <= 0 for each k
    cons = []
    cons.append({'type': 'eq', 'fun': lambda x: x[:N].sum() - 50.0})
    for k in range(len(c)):
        ck = c[k]
        cons.append({'type': 'ineq', 'fun': lambda x, ck=ck: x[N] - ck})
    return cons


def _local_refine(h, time_limit=18.0):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0).copy()
    target = 50.0
    # fix sum
    diff = target - h.sum()
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    h += (target - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    best = h.copy()
    bestv = _value(best)
    t0 = time.time()
    step = 0.2
    while time.time() - t0 < time_limit and step > 1e-4:
        improved = False
        for _ in range(3000):
            if time.time() - t0 > time_limit:
                break
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            # sum-preserving swap: move delta from i to j
            delta = np.random.uniform(-step, step)
            ni = best[i] - delta
            nj = best[j] + delta
            if ni < 0 or ni > 1 or nj < 0 or nj > 1:
                continue
            trial = best.copy()
            trial[i] = ni
            trial[j] = nj
            v = _value(trial)
            if v < bestv - 1e-12:
                best = trial
                bestv = v
                improved = True
        if not improved:
            step *= 0.7
    return best, bestv


def _epigraph_solve(h0, time_limit=30.0):
    """
    Solve the epigraph formulation:
    min t
    s.t. C_k <= t for all k
         sum(h) = 50
         0 <= h_i <= 1
    Using scipy.optimize.minimize with SLSQP.
    Returns (h, t) or None if it fails.
    """
    from scipy.optimize import minimize
    
    # Initial t: use current max correlation
    c0 = _corr_vec(h0)
    t0 = float(c0.max()) + 0.1
    
    z0 = np.concatenate([h0, [t0]])
    
    bounds = [(0.0, 1.0)] * N + [(0.0, float('inf'))]
    
    # Build constraints
    def sum_constraint(x):
        return x[:N].sum() - 50.0
    
    constraints = [{'type': 'eq', 'fun': sum_constraint}]
    
    # Inequality constraints: t - C_k >= 0 for each lag
    # We need to be careful with lambda closures
    for k in range(2 * N - 1):
        def make_ineq(k):
            def ineq(x):
                h = x[:N]
                t = x[N]
                c = np.correlate(h, 1.0 - h, mode="full")
                return t - c[k]
            return ineq
        constraints.append({'type': 'ineq', 'fun': make_ineq(k)})
    
    t_start = time.time()
    
    try:
        res = minimize(
            _epigraph_objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout")),
        )
        
        if res.success or (hasattr(res, 'x') and res.x is not None):
            h_sol = np.clip(res.x[:N], 0.0, 1.0)
            # Fix sum
            h_sol += (50.0 - h_sol.sum()) / N
            h_sol = np.clip(h_sol, 0.0, 1.0)
            t_sol = float(res.x[N])
            return h_sol, t_sol
    except Exception:
        pass
    
    return None


def _init_kinds():
    inits = []
    # 1) constant
    inits.append(np.full(N, 0.5))
    # 2) alternating 0/1 blocks (balanced)
    a = np.zeros(N)
    a[0::2] = 1.0
    inits.append(a)
    # 3) gradient
    inits.append(np.linspace(0.0, 1.0, N))
    # 4) sawtooth
    x = np.arange(N)
    inits.append((x % 10) / 9.0 * 2.0 - 1.0 + 0.5)
    inits[-1] = np.clip(inits[-1], 0, 1)
    # 5) random
    rng = np.random.default_rng(12345)
    r = rng.uniform(0, 1, N)
    r += (50.0 - r.sum()) / N
    inits.append(np.clip(r, 0, 1))
    # 6) two-tone block
    b = np.zeros(N)
    b[:50] = 1.0
    inits.append(b)
    # 7) sinusoid
    inits.append(0.5 + 0.4 * np.sin(2 * np.pi * np.arange(N) / N * 3))
    inits[-1] = np.clip(inits[-1], 0, 1)
    return inits


def construct():
    t_start = time.time()
    best = None
    bestv = float("inf")

    def consider(h):
        nonlocal best, bestv
        h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
        h += (50.0 - h.sum()) / N
        h = np.clip(h, 0.0, 1.0)
        v = _value(h)
        if v < bestv:
            bestv = v
            best = h.copy()
        return v

    inits = _init_kinds()
    from scipy.optimize import minimize

    for k, h0 in enumerate(inits):
        if time.time() - t_start > DEADLINE - 40.0:
            break
        h0 = np.clip(np.asarray(h0, dtype=float), 0.0, 1.0)
        h0 += (50.0 - h0.sum()) / N
        h0 = np.clip(h0, 0.0, 1.0)
        consider(h0)
        
        # Epigraph convex optimization
        if time.time() - t_start < DEADLINE - 30.0:
            result = _epigraph_solve(h0, time_limit=min(25.0, max(2.0, DEADLINE - 30.0 - (time.time() - t_start))))
            if result is not None:
                h_sol, t_sol = result
                consider(h_sol)
        
        # Refine with local search
        if time.time() - t_start < DEADLINE - 15.0:
            h2, v2 = _local_refine(best if best is not None else h0,
                                   time_limit=min(12.0, max(1.0, DEADLINE - 15.0 - (time.time() - t_start))))
            consider(h2)

    # Final epigraph solve on the best solution found so far
    if best is not None and time.time() - t_start < DEADLINE - 10.0:
        result = _epigraph_solve(best, time_limit=min(20.0, max(2.0, DEADLINE - 10.0 - (time.time() - t_start))))
        if result is not None:
            h_sol, t_sol = result
            consider(h_sol)
    
    # Final local refinement
    if best is not None and time.time() - t_start < DEADLINE - 3.0:
        h2, v2 = _local_refine(best, time_limit=max(1.0, DEADLINE - 3.0 - (time.time() - t_start)))
        consider(h2)

    if best is None:
        best = np.full(N, 0.5)
    return best
