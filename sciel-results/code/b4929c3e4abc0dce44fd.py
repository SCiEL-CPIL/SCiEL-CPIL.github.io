import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # ---------- True (non-smooth) objective ----------
    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-12:
            return 0.0
        return l2_sq / (l1 * max_conv)

    # ---------- Smooth surrogate objective (for optimization) ----------
    # Normalize l1=1. Objective: maximize  sum h^2 / max conv.
    # Use soft-max (log-sum-exp) for the conv max to get a smooth function.
    def smooth_objective(h, beta):
        # h is the amplitude vector on the support (length k), nonnegative
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return -1e6
        hn = h / l1  # normalize to l1=1
        l2_sq = np.sum(hn**2)
        # full convolution
        conv = np.convolve(hn, hn, mode='full')
        # soft max
        cmax = np.max(conv)
        if cmax <= 0:
            return -1e6
        # log-sum-exp
        shifted = conv / cmax  # <=1
        # use a stable soft max: cmax * (1 + (sum(exp(beta*(conv/cmax -1))))^(-1/beta))
        # but conv/cmax-1 <= 0, so exp is in (0,1]; sum >= 1
        ex = np.exp(beta * (conv / cmax - 1.0))
        s = np.sum(ex)
        soft = cmax * (1.0 + s**(-1.0/beta))
        if soft <= 0:
            return -1e6
        ratio = l2_sq / soft
        if ratio <= 0:
            return -1e6
        return -np.log(ratio)

    def smooth_grad(h, beta):
        # finite-difference gradient of smooth_objective
        eps = 1e-6
        g = np.zeros_like(h)
        f0 = smooth_objective(h, beta)
        for i in range(len(h)):
            hp = h.copy(); hp[i] += eps
            hm = h.copy(); hm[i] = max(hm[i]-eps, 0.0)
            g[i] = (smooth_objective(hp, beta) - smooth_objective(hm, beta)) / (2*eps)
        return g

    # ---------- Discrete support helpers ----------
    def max_mult(S):
        if not S:
            return 0
        S_list = list(S)
        sums = {}
        for i in S_list:
            for j in S_list:
                s = i + j
                sums[s] = sums.get(s, 0) + 1
        return max(sums.values())

    def greedy_build(seed_set, max_size=16):
        S = set(seed_set)
        for _ in range(max_size):
            best_x = -1
            best_max_mult = float('inf')
            best_ratio = -1
            for x in range(N):
                if x in S:
                    continue
                S_trial = S | {x}
                m = max_mult(S_trial)
                k = len(S_trial)
                ratio = k / m if m > 0 else 0
                if m < best_max_mult:
                    best_max_mult = m
                    best_x = x
                    best_ratio = ratio
                elif m == best_max_mult:
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_x = x
            if best_x != -1:
                S.add(best_x)
            else:
                break
        return S

    best_h = np.zeros(N)
    best_score = 0.0

    # ---------- Phase 1: find good support sets ----------
    seeds = [
        [0],
        [0, 1],
        [0, 1, 4],
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23, 30],
        [1, 2, 5, 11, 21, 34],
        [0, 1, 3, 7, 12, 20, 31],
        [0, 2, 5, 9, 14, 21, 29, 38],
        [0, 1, 4, 10, 18, 28, 39],
        [0, 3, 7, 12, 18, 25, 33, 42],
        [0, 2, 7, 13, 20, 28, 37, 46],
        [0, 1, 3, 7, 15, 27, 43],
        [0, 1, 4, 10, 19, 31, 44],
        [0, 2, 5, 11, 19, 29, 41],
    ]

    support_candidates = []
    for seed in seeds:
        if time.time() > deadline - 40:
            break
        S = greedy_build(seed)
        if S:
            support_candidates.append(frozenset(S))

    # random sparse sets
    rng = np.random.default_rng(42)
    for _ in range(4000):
        if time.time() > deadline - 40:
            break
        k = int(rng.integers(3, 18))
        S = set(rng.choice(N, k, replace=False).tolist())
        support_candidates.append(frozenset(S))

    # dedupe
    support_candidates = list(set(support_candidates))

    # ---------- Phase 2: intensive continuous optimization per support ----------
    from scipy.optimize import minimize

    for S in support_candidates:
        if time.time() > deadline - 15:
            break
        S = list(S)
        k = len(S)
        if k == 0 or k > 20:
            continue

        # quick uniform score as a baseline
        h_u = np.zeros(N)
        for i in S:
            h_u[i] = 1.0/k
        s_u = score(h_u)
        if s_u > best_score:
            best_score = s_u
            best_h = h_u.copy()

        # Continuous optimization on this support
        # Try several beta values for the soft-max
        best_x = np.ones(k)/k
        best_fun = smooth_objective(best_x, 60.0)

        # diverse initializations
        inits = [np.ones(k)/k]
        # power-law
        t = np.linspace(0.2, 1.0, k)
        inits.append(t/np.sum(t))
        # two-bump
        bump = np.zeros(k); bump[:k//2] = 1.0; bump[k//2:] = 0.3
        inits.append(bump/np.sum(bump))
        # periodic
        per = np.array([1.0 if i%2==0 else 0.4 for i in range(k)])
        inits.append(per/np.sum(per))
        # random
        for _ in range(4):
            r = rng.random(k)
            r = np.maximum(r, 0.05)
            inits.append(r/np.sum(r))

        for x_init in inits:
            if time.time() > deadline - 15:
                break
            for beta in [40.0, 80.0]:
                if time.time() > deadline - 15:
                    break
                try:
                    res = minimize(smooth_objective, x_init.copy(),
                                   args=(beta,),
                                   jac=lambda x, b: smooth_grad(x, b),
                                   method='L-BFGS-B',
                                   bounds=[(1e-10, None)]*k,
                                   options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-13})
                    if res.fun < best_fun:
                        best_fun = res.fun
                        best_x = res.x
                except Exception:
                    pass

        if len(best_x) > 0:
            h_opt = np.zeros(N)
            for i, v in zip(S, best_x):
                h_opt[i] = max(v, 0.0)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # ---------- Phase 3: coordinate-descent polish on true objective ----------
    h = best_h.copy()
    support = list(np.nonzero(h)[0])
    if support:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            cur = score(h)
            for idx in support:
                if time.time() > deadline - 3:
                    break
                h_old = h[idx]
                if h_old < 1e-10:
                    continue
                for factor in np.linspace(0.1, 5.0, 30):
                    h_trial = h.copy()
                    h_trial[idx] = h_old * factor
                    s_trial = score(h_trial)
                    if s_trial > cur + 1e-11:
                        h = h_trial
                        cur = s_trial
                        improved = True
                        break

    # ---------- Phase 4: support swap polish ----------
    if time.time() < deadline - 8:
        S_cur = set(np.nonzero(h)[0])
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            cur = score(h)
            for i in list(S_cur):
                if time.time() > deadline - 8:
                    break
                for j in range(N):
                    if j in S_cur:
                        continue
                    S_trial = S_cur.copy()
                    S_trial.remove(i)
                    S_trial.add(j)
                    h_trial = np.zeros(N)
                    for kk in S_trial:
                        h_trial[kk] = h[kk] if kk in S_cur else 1.0
                    # quick coordinate ascent on amplitudes
                    sup_t = list(S_trial)
                    if sup_t:
                        x_t = np.array([h_trial[kk] for kk in sup_t])
                        for _ in range(4):
                            for idx in sup_t:
                                pos = sup_t.index(idx)
                                ov = x_t[pos]
                                if ov < 1e-10:
                                    continue
                                for fac in [0.5, 0.8, 1.2, 1.5, 2.0]:
                                    x_tn = x_t.copy()
                                    x_tn[pos] = ov*fac
                                    h_t = np.zeros(N)
                                    h_t[sup_t] = np.maximum(x_tn, 0)
                                    if score(h_t) > score(h_trial):
                                        x_t = x_tn
                                        h_trial = h_t
                                        break
                        h_trial[sup_t] = np.maximum(x_t, 0)
                    s_trial = score(h_trial)
                    if s_trial > cur + 1e-10:
                        S_cur = S_trial
                        h = h_trial
                        cur = s_trial
                        improved = True
                        break
                if improved:
                    break
        best_h = h

    return list(best_h)
