"""MODEL-AND-OPTIMIZE dispatch: fit a linear weight vector over hand-crafted
features by actually simulating the job-shop (shortest makespan = objective)
and optimizing weights via coordinate ascent with simulated-annealing step sizes.
Fallback composite heuristic if time runs out."""
# EVOLVE-BLOCK-START
import time as _T
import numpy as np

_T0 = _T.time()
_BUDGET = 132.0  # explicit deadline guard (150s kill -> margin)


def _remaining(op):
    return op.get("remaining_ops", 0)


def _min_dur(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 1.0
    return min(mo.values())


def _max_dur(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 1.0
    return max(mo.values())


def _mean_dur(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 1.0
    return (sum(mo.values()) + 1e-9) / len(mo)


def _spread(op):
    mo = op.get("machine_options", {})
    if len(mo) < 2:
        return 0.0
    v = np.array(list(mo.values()), dtype=float)
    return float(v.std())


def _n_opt(op):
    return len(op.get("machine_options", {}))


def _feats(op):
    r = float(op.get("remaining_work", _min_dur(op)))
    mn = _min_dur(op)
    mx = _max_dur(op)
    mean = _mean_dur(op)
    spread = _spread(op)
    ro = float(op.get("remaining_ops", 0))
    no = float(_n_opt(op))
    speed = 1.0 / (mn + 1e-9)
    # ratio: how much faster is min vs mean (concentration of options)
    conc = mn / (mean + 1e-9)
    # urgency proxy: remaining_work / (remaining_ops + 1)
    urg_proxy = r / (ro + 1.0)
    return np.array([
        r, mn, mx, mean, speed, spread, ro, no,
        1.0 / (ro + 1.0), 1.0 / (no + 1.0), r * spread, r * speed,
        conc, urg_proxy,
    ])


def _norm(a):
    lo = np.min(a)
    hi = np.max(a)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _urgency(ops, feats):
    """Urgency based on total remaining work per job (sum of min durations of all remaining ops).
    We approximate using the current op's remaining_work as the best available signal,
    but weight by number of remaining ops to capture total job load."""
    jwork = {}
    jops = {}
    for i, op in enumerate(ops):
        j = op.get("job", i)
        w = feats[i, 0]  # remaining_work
        ro = feats[i, 6]  # remaining_ops
        # Use remaining_work as a proxy for total remaining job work
        if j not in jwork or w > jwork[j]:
            jwork[j] = w
        if j not in jops or ro > jops[j]:
            jops[j] = ro
    if not jwork:
        return np.zeros(len(ops))
    # Scale by remaining ops to get a "total load" estimate
    jload = {}
    for j in jwork:
        jload[j] = jwork[j] * (1.0 + 0.3 * jops[j])
    cp = max(jload.values())
    if cp <= 1e-9:
        return np.ones(len(ops))
    return np.array([jload[op.get("job", i)] / cp for i, op in enumerate(ops)])


def _score(ops, feats, w, time):
    n = len(ops)
    if n == 1:
        return 0
    urg = _urgency(ops, feats)
    urg_speed = urg * feats[:, 4]
    # per-feature normalized matrices
    X = feats.copy()
    # normalize each column to [0,1]
    for c in range(X.shape[1]):
        X[:, c] = _norm(X[:, c])
    # build raw score = sum w_k * f_k  where f_k in [0,1] (higher better)
    s = np.zeros(n)
    for k in range(X.shape[1]):
        s += w[k] * X[:, k]
    # add urgency*speed term (a strong composite) as an extra feature
    if len(w) > 14:
        s += w[14] * _norm(urg_speed)
    idx = int(np.argmax(s))
    return idx


def _simulate(ops, w):
    """Event-driven simulation given a full op set. Returns makespan."""
    ops = [dict(o) for o in ops]
    # group by job
    jobs = {}
    for o in ops:
        jobs.setdefault(o.get("job", id(o)), []).append(o)
    for lst in jobs.values():
        lst.sort(key=lambda o: o.get("op", 0))
    # ready = first op of each job
    ready = [lst[0] for lst in jobs.values()]
    pos_in_job = {j: 0 for j in jobs}
    mach_avail = {}
    makespan = 0.0
    total = sum(len(lst) for lst in jobs.values())
    done = 0
    t = 0.0
    while done < total:
        if not ready:
            break
        idx = _score(ready, _all_feats(ready), w, t)
        op = ready.pop(idx)
        job = op.get("job", idx)
        mo = op.get("machine_options", {})
        # choose machine that minimizes end time (max(t, avail) + dur)
        best_m = None
        best_end = float('inf')
        dur = 0.0
        for m, d in mo.items():
            av = mach_avail.get(m, 0.0)
            s = max(t, av)
            end = s + d
            if end < best_end or (end == best_end and av < mach_avail.get(best_m, 0.0)):
                best_m = m
                best_end = end
                dur = d
        mach_avail[best_m] = best_end
        if best_end > makespan:
            makespan = best_end
        t = best_end
        done += 1
        pos_in_job[job] += 1
        lst = jobs[job]
        if pos_in_job[job] < len(lst):
            nxt = lst[pos_in_job[job]]
            ready.append(nxt)
    return makespan


def _all_feats(ops):
    return np.array([_feats(o) for o in ops], dtype=float)


def _fit(ops_list):
    """Nelder-Mead simplex fit of weight vector against mean makespan."""
    global _W
    nfeat = 15  # 14 base features + 1 urgency*speed
    # sensible initial weights
    w0 = np.zeros(nfeat)
    w0[0] = 1.0    # remaining work (MWR)
    w0[4] = 1.5    # speed (1/min_dur)
    w0[1] = 0.3    # min dur
    w0[7] = 0.5    # n options (constrained)
    w0[6] = 0.5    # remaining ops
    w0[14] = 2.0   # urgency*speed

    def obj(wv):
        return np.mean([_simulate(ops, wv) for ops in ops_list])

    best_w = w0.copy()
    best_obj = obj(w0)

    # Nelder-Mead simplex
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.7
    # Initialize simplex: w0 + small perturbations
    simplex = [w0.copy()]
    for k in range(nfeat):
        vi = w0.copy()
        vi[k] += 0.1 if w0[k] == 0 else 0.1 * abs(w0[k])
        simplex.append(vi)
    fvals = [obj(v) for v in simplex]
    best_idx = int(np.argmin(fvals))
    best_obj = fvals[best_idx]
    best_w = simplex[best_idx].copy()

    max_iter = 500
    it = 0
    while it < max_iter and _T.time() - _T0 < _BUDGET - 5.0:
        it += 1
        # Sort simplex by objective value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Update best
        if fvals[0] < best_obj - 1e-12:
            best_obj = fvals[0]
            best_w = simplex[0].copy()

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        f_r = obj(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = obj(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            f_c = obj(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, nfeat + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = obj(simplex[i])
                # Check for convergence
                if max(fvals) - min(fvals) < 1e-8:
                    break

    return best_w


# cached fit
_W = None
_FIT_CACHE = []


def _default_weights():
    w = np.zeros(15)
    w[0] = 1.0
    w[4] = 1.5
    w[1] = 0.3
    w[7] = 0.5
    w[6] = 0.5
    w[14] = 2.0
    return w


def _get_weights(ops):
    global _W
    if _W is not None:
        return _W
    # fit on the first batch of instances we see
    _FIT_CACHE.append(ops)
    # fit only once after first instance (we need full instance to simulate)
    # We'll trigger fit using this ops list as the instance.
    if _T.time() - _T0 < _BUDGET - 5.0:
        try:
            _W = _fit([ops])
        except Exception:
            _W = _default_weights()
    if _W is None:
        _W = _default_weights()
    return _W


def _fallback(ready_operations, time):
    n = len(ready_operations)
    if n <= 1:
        return 0
    feats = _all_feats(ready_operations)
    urg = _urgency(ready_operations, feats)
    urg_speed = urg * feats[:, 4]
    X = feats.copy()
    for c in range(X.shape[1]):
        X[:, c] = _norm(X[:, c])
    w = _default_weights()
    s = np.zeros(n)
    for k in range(X.shape[1]):
        s += w[k] * X[:, k]
    if len(w) > 14:
        s += w[14] * _norm(urg_speed)
    return int(np.argmax(s))


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    if _T.time() - _T0 > _BUDGET:
        return 0
    try:
        w = _get_weights(ready_operations)
        feats = _all_feats(ready_operations)
        return _score(ready_operations, feats, w, time)
    except Exception:
        return _fallback(ready_operations, time)
# EVOLVE-BLOCK-END
