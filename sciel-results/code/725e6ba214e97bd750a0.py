"""Direct, structure-driven heuristic for ACO multiple-constraint knapsack."""
import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0   # hard wall-clock bound from evaluator
MARGIN = 8.0    # safety margin
DEADLINE = T_START + (LIMIT - MARGIN)


def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Normalize shapes -> per-item prize (n,) and per-(constraint,item) weight (m, n)
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2 and weights.shape[0] != weights.shape[1]:
        w = weights
    elif weights.ndim == 2 and weights.shape[1] == p.shape[0]:
        w = weights  # already (m, n)
    else:
        w = weights.reshape(1, -1)

    n = int(p.shape[0])
    m = int(w.shape[0])
    cap = 25.0
    C = np.full(m, cap, dtype=float)

    # Safety: ensure finite, positive output of correct shape
    def _safe(a):
        a = np.where(np.isfinite(a), a, 1.0)
        a = np.maximum(a, 1e-9)
        return a

    if n == 0:
        return np.ones(0)

    # ------------------------------------------------------------------
    # 1) Lagrangian / subgradient: estimate shadow prices lambda for each
    #    constraint. Reduced cost rc_i = p_i - sum_c lambda_c * w_c_i.
    #    Items with high rc are "profitable after paying for resources".
    # ------------------------------------------------------------------
    lam = np.zeros(m, dtype=float)
    best_rc = p.copy()
    best_rc_val = -1.0
    best_packs = []  # (value, [item order])

    # subgradient steps: lambda_{c} <- max(0, lambda_c + step * slack_c)
    n_sub = 25
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 2.0:
            break
        shadow = lam @ w                 # (n,)
        rc = p - shadow
        order = np.argsort(-rc, kind="stable")
        rem = C.copy()
        sel = []
        val = 0.0
        for i in order:
            if rc[i] <= 0.0:
                break
            if np.all(w[:, i] <= rem):
                sel.append(i)
                rem -= w[:, i]
                val += p[i]
        if val > best_rc_val:
            best_rc_val = val
            best_rc = rc.copy()
            best_packs.append((val, sel))
        used = np.zeros(m)
        for i in sel:
            used += w[:, i]
        slack = cap - used
        step = (4.0 / (1.0 + it * 0.15))
        lam = np.maximum(0.0, lam + step * slack)

    # Normalize reduced cost to [0,1]
    rmin = best_rc.min()
    rmax = best_rc.max()
    if rmax > rmin:
        rc_norm = (best_rc - rmin) / (rmax - rmin)
    else:
        rc_norm = np.ones(n)
    rc_norm = np.clip(rc_norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Precompute structural weight metrics
    # ------------------------------------------------------------------
    sum_w = w.sum(axis=0)
    max_w = w.max(axis=0)
    min_w = w.min(axis=0)
    avg_w = w.mean(axis=0)
    safe_max = np.maximum(max_w, 1e-9)

    # Bottleneck slack: min over c of (cap - w_c)/w_c  -> "how easily it fits"
    slack_ratios = (cap - w) / np.maximum(w, 1e-9)
    bottleneck_slack = np.maximum(slack_ratios.min(axis=0), 0.0)

    # ------------------------------------------------------------------
    # 3) Many fast greedy simulations with diverse orderings.
    #    Score each item by its total prize contribution to high-value packs.
    #    This captures the discrete "blocking" effect: an item is valuable
    #    only if it actually fits into good packs, not just a good ratio.
    # ------------------------------------------------------------------
    contrib = np.zeros(n, dtype=float)   # total prize contributed per item
    selection_freq = np.zeros(n, dtype=float)
    best_found = 0.0

    def _greedy(order):
        nonlocal best_found
        rem = C.copy()
        val = 0.0
        sel = []
        for i in order:
            if np.all(w[:, i] <= rem):
                sel.append(i)
                rem -= w[:, i]
                val += p[i]
        if val > 0:
            for i in sel:
                contrib[i] += val
                selection_freq[i] += 1.0
        if val > best_found:
            best_found = val
        return val

    # Build a set of orderings (deterministic ratios + Lagrangian + a few random)
    orderings = []
    orderings.append(np.argsort(-rc_norm, kind="stable"))
    orderings.append(np.argsort(-(p / (sum_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max ** 2 + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (avg_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (min_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p * (bottleneck_slack + 1.0) / (safe_max + 1.0)), kind="stable"))
    orderings.append(np.argsort(-p, kind="stable"))

    # A few deterministic pseudo-random shuffles (fixed seeds) for diversity
    for seed in (7, 13, 42):
        rng = np.random.RandomState(seed)
        orderings.append(np.array(rng.permutation(n)))

    # Run greedy until deadline (these are O(n*m) each, fast for n=100,m=5)
    max_rounds = 200
    for r in range(max_rounds):
        if _deadline_left() < MARGIN:
            break
        for order in orderings:
            if _deadline_left() < MARGIN:
                break
            _greedy(order)

    # ------------------------------------------------------------------
    # 3b) Regret-based scoring: for each item, compute the drop in total
    #     prize of a baseline greedy solution if that item is forced to
    #     be excluded. This explicitly captures blocking effects.
    # ------------------------------------------------------------------
    # Use the best ordering found so far as the baseline
    baseline_order = orderings[0]  # rc_norm ordering
    baseline_rem = C.copy()
    baseline_val = 0.0
    baseline_sel = []
    for i in baseline_order:
        if np.all(w[:, i] <= baseline_rem):
            baseline_sel.append(i)
            baseline_rem -= w[:, i]
            baseline_val += p[i]

    regret = np.zeros(n, dtype=float)
    # For each item, compute regret by excluding it and re-greedy
    for i in range(n):
        if _deadline_left() < MARGIN + 1.0:
            break
        rem = C.copy()
        val = 0.0
        for j in baseline_order:
            if j == i:
                continue
            if np.all(w[:, j] <= rem):
                rem -= w[:, j]
                val += p[j]
        regret[i] = max(0.0, baseline_val - val)

    # ------------------------------------------------------------------
    # 4) Combine signals into the final attractiveness score.
    #    Dominant signal: contribution to good packs (captures blocking).
    #    Secondary: reduced cost, bottleneck slack, raw prize, regret.
    # ------------------------------------------------------------------
    cmax = contrib.max()
    if cmax > 0:
        contrib_norm = contrib / cmax
    else:
        contrib_norm = np.ones(n)

    fmax = selection_freq.max()
    if fmax > 0:
        freq_norm = selection_freq / fmax
    else:
        freq_norm = np.ones(n)

    pmax = p.max()
    if pmax > 0:
        p_norm = p / pmax
    else:
        p_norm = np.ones(n)

    bmax = bottleneck_slack.max()
    if bmax > 0:
        bs_norm = bottleneck_slack / bmax
    else:
        bs_norm = np.ones(n)

    # Normalize regret: items with high regret are "blocking" and should be
    # de-emphasized. Use inverse regret as a positive signal.
    rmax = regret.max()
    if rmax > 0:
        # Items with low regret (non-blocking) get high scores
        regret_norm = 1.0 - (regret / rmax)
    else:
        regret_norm = np.ones(n)

    # Product-form: emphasize contribution & reduced cost strongly,
    # include regret as a penalty for blocking items.
    score = (
        (contrib_norm ** 2.0)
        * (rc_norm ** 1.0)
        * (p_norm ** 1.0)
        * (bs_norm ** 0.5)
        * (freq_norm ** 0.5)
        * (regret_norm ** 1.0)
    )
    score = score + 1e-9

    # ------------------------------------------------------------------
    # 5) Calibrate: the ACO uses heuristic in iteration 1 (pheromone=1).
    #    Scale the heuristic so its own greedy total matches best_found,
    #    i.e. the heuristic's relative ordering is preserved but its
    #    magnitude reflects the achievable pack value. This is a pure
    #    scaling (relative ratios unchanged) so it cannot harm ordering.
    # ------------------------------------------------------------------
    if best_found > 0:
        # greedy total using this score as the ordering proxy
        order = np.argsort(-score, kind="stable")
        rem = C.copy()
        val = 0.0
        for i in order:
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
        if val > 0:
            scale = best_found / val
            score = score * scale

    score = _safe(score)
    smax = score.max()
    if smax > 0:
        score = score / smax
    score = _safe(score)
    return score
