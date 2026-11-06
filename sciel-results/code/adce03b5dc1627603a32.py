import time
import math
import numpy as np

def _eval_score(c):
    """Compute max |P(z)| / sqrt(24) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(c)
    # Dense sampling
    m = 4096
    theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    z = np.exp(1j * theta)
    # P(z) = sum c_k z^k
    vals = np.zeros(m, dtype=complex)
    for k in range(n):
        vals += c[k] * (z ** k)
    mag = np.abs(vals)
    idx = np.argmax(mag)
    # Golden-section refinement around the peak
    best_val = mag[idx]
    best_theta = theta[idx]
    # Refine in a small window
    h = 2*np.pi / m
    lo = best_theta - h
    hi = best_theta + h
    # Golden section
    gr = (math.sqrt(5) - 1) / 2
    def f(t):
        zz = np.exp(1j * t)
        v = 0.0 + 0.0j
        for k in range(n):
            v += c[k] * (zz ** k)
        return abs(v)
    for _ in range(60):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-12:
            break
    mid = (lo + hi) / 2
    best_val = f(mid)
    return best_val / math.sqrt(n)

def _eval_fast(c):
    """Fast approximate evaluation for local search."""
    n = len(c)
    m = 2048
    theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    z = np.exp(1j * theta)
    vals = np.zeros(m, dtype=complex)
    for k in range(n):
        vals += c[k] * (z ** k)
    return np.max(np.abs(vals)) / math.sqrt(n)

def _local_search(c, max_iter=20000, deadline=None):
    """Local search with single and double flips."""
    n = len(c)
    c = list(c)
    cur_score = _eval_fast(c)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        # Single flips
        best_delta = 0
        best_idx = -1
        for i in range(n):
            c[i] = -c[i]
            s = _eval_fast(c)
            c[i] = -c[i]  # restore
            if s < cur_score - 1e-12:
                best_delta = s - cur_score
                best_idx = i
        if best_idx >= 0:
            c[best_idx] = -c[best_idx]
            cur_score += best_delta
            improved = True
            continue
        # Double flips
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return c, _eval_score(c)
            c[i] = -c[i]
            for j in range(i+1, n):
                c[j] = -c[j]
                s = _eval_fast(c)
                c[j] = c[j]  # restore (already -c[j] so restore means set back)
                # Actually c[j] was -c[j], so to restore: c[j] = -c[j]
                c[j] = -c[j]
                if s < cur_score - 1e-12:
                    best_delta = s - cur_score
                    best_idx = i
                    # need to store j too
                    # Let me restructure
                    pass
            c[i] = -c[i]
    return c, _eval_score(c)

def _local_search_v2(c, max_iter=50000, deadline=None):
    """Local search with single and double flips, properly structured."""
    n = len(c)
    c = list(c)
    cur_score = _eval_fast(c)
    iters = 0
    while iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        iters += 1
        improved = False
        # Try single flips
        best_s = cur_score
        best_move = None
        for i in range(n):
            c[i] = -c[i]
            s = _eval_fast(c)
            c[i] = -c[i]
            if s < best_s - 1e-12:
                best_s = s
                best_move = (i, -1)
        if best_move is not None:
            i = best_move[0]
            c[i] = -c[i]
            cur_score = best_s
            improved = True
            continue
        # Try double flips
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            c[i] = -c[i]
            for j in range(i+1, n):
                c[j] = -c[j]
                s = _eval_fast(c)
                c[j] = -c[j]
                if s < best_s - 1e-12:
                    best_s = s
                    best_move = (i, j)
            c[i] = -c[i]
            if best_move is not None:
                break
        if best_move is not None:
            i, j = best_move
            c[i] = -c[i]
            if j >= 0:
                c[j] = -c[j]
            cur_score = best_s
            improved = True
        else:
            break
    return c, _eval_score(c)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    # Start from Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    init = p[:24]
    
    best_c = list(init)
    best_score = _eval_score(best_c)
    
    # Local search from initial
    c, s = _local_search_v2(init, max_iter=30000, deadline=deadline)
    if s < best_score:
        best_c = c
        best_score = s
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        rc = list(np.random.choice([-1, 1], size=24))
        c, s = _local_search_v2(rc, max_iter=10000, deadline=deadline)
        if s < best_score:
            best_c = c
            best_score = s
    
    # Final polish
    if time.time() < deadline:
        c, s = _local_search_v2(best_c, max_iter=5000, deadline=deadline)
        if s < best_score:
            best_c = c
            best_score = s
    
    return best_c
