import numpy as np
import time

N = 50

def _eval(h):
    """Exact evaluator mirroring the public contract. h length N, nonneg."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _eval_fast(h):
    """Vectorized evaluation using FFT for g = h*h (full)."""
    n = len(h)
    # full convolution length = 2n-1
    size = 2*n - 1
    # pad to at least size; use next fast size
    fsize = 1
    while fsize < size:
        fsize <<= 1
    H = np.fft.rfft(h, n=fsize)
    G = np.fft.irfft(H * H, n=fsize)[:size]
    g = G
    m = size
    # y = [0, g, 0]
    # terms[i] = y[i]^2 + y[i]y[i+1] + y[i+1]^2 for i in 0..m
    # y[0]=0, y[1..m]=g[0..m-1], y[m+1]=0
    # i=0: 0 + 0 + g[0]^2 = g[0]^2
    # i=m: g[m-1]^2 + 0 + 0 = g[m-1]^2
    # i in 1..m-1: g[i-1]^2 + g[i-1]g[i] + g[i]^2
    terms = np.empty(m+1)
    terms[0] = g[0]*g[0]
    terms[m] = g[-1]*g[-1]
    if m > 1:
        terms[1:m] = g[:-1]**2 + g[:-1]*g[1:] + g[1:]**2
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def _make_profile(t):
    # t in [0,1], 0=boxcar, 1=gaussian, 0.5=triangle-ish
    x = np.arange(N)
    c = (N-1)/2.0
    box = np.ones(N)
    gauss = np.exp(-((x - c)**2) / (2 * (N/4.0)**2))
    tri = 1.0 - np.abs(x - c) / (c + 1.0)
    if t <= 0.5:
        a = t/0.5
        h = (1-a)*box + a*tri
    else:
        a = (t-0.5)/0.5
        h = (1-a)*tri + a*gauss
    s = h.sum()
    if s <= 0:
        return np.ones(N)/N
    return h/s

def _eval_batch(hs):
    # hs: (B, N) -> (B,)
    B, n = hs.shape
    size = 2*n - 1
    fsize = 1
    while fsize < size:
        fsize <<= 1
    H = np.fft.rfft(hs, n=fsize, axis=1)
    G = np.fft.irfft(H*H, n=fsize, axis=1)[:size]
    m = size
    g0 = G[:,0]
    g1 = G[:,-1]
    terms = np.empty((B, m+1))
    terms[:,0] = g0*g0
    terms[:,-1] = g1*g1
    if m > 1:
        terms[:,1:m] = G[:,:-1]**2 + G[:,:-1]*G[:,1:] + G[:,1:]**2
    w = 1.0 / (m + 1)
    l2 = (w/3.0) * np.sum(terms, axis=1)
    norm_1 = np.sum(np.abs(G), axis=1) / (m+1)
    norm_inf = np.max(np.abs(G), axis=1)
    den = norm_1*norm_inf
    out = np.where(den>0, l2/np.where(den>0,den,1.0), 0.0)
    return out

def construct():
    start = time.time()
    deadline = start + 138.0
    rng = np.random.default_rng(20240517)

    # Build a diverse initial population of profiles
    profiles = []
    # profile family: boxcar->triangle->gaussian sweep
    for t in np.linspace(0,1,11):
        profiles.append(_make_profile(t))
    # boxcars of various widths centered
    for width in [10,15,20,25,30,35,40,45,50]:
        h = np.zeros(N)
        c = (N-1)/2.0
        lo = int(round(c - width/2)); hi = int(round(c + width/2))
        lo = max(0,lo); hi = min(N,hi)
        if hi>lo:
            h[lo:hi]=1.0
            s=h.sum()
            if s>0:
                profiles.append(h/s)
    # off-center boxcars
    for _ in range(20):
        width = rng.integers(8,45)
        lo = rng.integers(0, max(1,N-width))
        hi = min(N, lo+width)
        h = np.zeros(N); h[lo:hi]=1.0
        s=h.sum()
        if s>0: profiles.append(h/s)
    # gaussians with various centers/widths
    for _ in range(40):
        c = rng.uniform(0,N-1)
        wdt = rng.uniform(3, N)
        x = np.arange(N)
        h = np.exp(-((x-c)**2)/(2*wdt**2))
        s=h.sum()
        if s>0: profiles.append(h/s)
    # random
    for _ in range(60):
        h = rng.random(N)
        s=h.sum()
        if s>0: profiles.append(h/s)
    # random sparse (spikes)
    for _ in range(30):
        h = np.zeros(N)
        k = rng.integers(1,12)
        idx = rng.choice(N, size=k, replace=False)
        h[idx]=1.0
        s=h.sum()
        if s>0: profiles.append(h/s)

    # Evaluate all in batch
    arr = np.vstack(profiles)
    scores = _eval_batch(arr)
    # keep top 30 as population
    order = np.argsort(scores)[::-1]
    pop = [arr[order[i]].copy() for i in range(min(30,len(order)))]
    pop_scores = [float(scores[order[i]]) for i in range(min(30,len(order)))]
    best_i = 0
    best_h = pop[0].copy()
    best_score = pop_scores[0]

    # Local refinement: coordinate descent on simplex using fast eval
    def refine(h, max_iter, deadline):
        h = h.copy()
        cur = _eval_fast(h)
        # coordinate moves: transfer mass between two coordinates
        for it in range(max_iter):
            if time.time() > deadline:
                break
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            # try several transfer amounts
            base = h.copy()
            improved = False
            for frac in (0.05,0.1,0.2,0.4,0.8):
                amt = base[i]*frac
                cand = base.copy()
                cand[i] = base[i] - amt
                cand[j] = base[j] + amt
                if cand[i] < -1e-12 or cand[j] < -1e-12:
                    continue
                s = cand.sum()
                if s <= 0:
                    continue
                cand = cand/s
                sc = _eval_fast(cand)
                if sc > cur + 1e-12:
                    cur = sc
                    h = cand
                    improved = True
                    break
            if not improved:
                # random small noise then project
                noise = rng.normal(0, 0.05, N)
                cand = h + noise
                cand = _project_simplex(cand)
                s = cand.sum()
                if s>0:
                    cand = cand/s
                    sc = _eval_fast(cand)
                    if sc > cur:
                        cur = sc
                        h = cand
        return h, cur

    # Refine top few profiles
    for idx in range(min(8, len(pop))):
        if time.time() > deadline:
            break
        rh, rs = refine(pop[idx], max_iter=400, deadline=deadline)
        if rs > best_score:
            best_score = rs
            best_h = rh.copy()

    # Hill-climbing random walk with restarts
    current = best_h.copy()
    cur_score = best_score
    step = 0.08
    no_improve = 0
    while time.time() < deadline:
        h_new = current.copy()
        k = int(rng.integers(1,4))
        idx = rng.choice(N, size=k, replace=False)
        h_new[idx] += rng.normal(0, step, k)
        h_new = _project_simplex(h_new)
        s = h_new.sum()
        if s > 0:
            h_new = h_new/s
        sc = _eval_fast(h_new)
        if sc > cur_score:
            current = h_new
            cur_score = sc
            no_improve = 0
            if sc > best_score:
                best_score = sc
                best_h = h_new.copy()
        else:
            no_improve += 1
            if no_improve > 30:
                # restart from best or random profile
                if rng.random() < 0.6:
                    current = best_h.copy()
                    cur_score = best_score
                else:
                    current = _make_profile(rng.random())
                    cur_score = _eval_fast(current)
                no_improve = 0
                step = 0.08
        step = max(0.02, step*0.995)

    # Final coordinate refinement of best
    if time.time() < deadline:
        rh, rs = refine(best_h, max_iter=300, deadline=deadline)
        if rs > best_score:
            best_score = rs
            best_h = rh.copy()

    # Ensure nonnegative finite, return as list of floats
    out = np.asarray(best_h, dtype=float)
    out = np.maximum(out, 0.0)
    if not np.all(np.isfinite(out)):
        out = np.ones(N)/N
    return [float(x) for x in out]
