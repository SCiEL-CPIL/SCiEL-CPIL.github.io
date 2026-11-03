<<<<<<< SEARCH
def _refine_polish(x0, deadline, rng, iters=30000):
    """High-precision polish: very small steps, tight patience."""
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-5
    no_improve = 0
    patience = 5000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        k = int(np.argmax(abs_c)) + 1
        ck = float(c[k])
        g = np.zeros(n)
        sign_ck = np.sign(ck)
        if k < n:
            g[:n - k] += sign_ck * x[k:]
            g[k:] += sign_ck * x[:n - k]
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -1.5, 1.5)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.02, 5e-5)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-8:
                    break
    return bestx, best
=======
def _refine_polish(x0, deadline, rng, iters=30000):
    """High-precision polish: very small steps, tight patience, with soft-max."""
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 5e-6
    beta = 50.0
    no_improve = 0
    patience = 3000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            break
        
        # Use soft-max with high beta for near-exact max gradient
        weights = np.exp(beta * (abs_c - m))
        weights /= weights.sum()
        
        g = np.zeros(n)
        for k in range(1, n):
            wk = weights[k - 1]
            if wk < 1e-10:
                continue
            sign_ck = np.sign(c[k])
            g[:n - k] += wk * sign_ck * x[k:]
            g[k:] += wk * sign_ck * x[:n - k]
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -1.5, 1.5)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.01, 2e-5)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-9:
                    break
    return bestx, best
>>>>>>> REPLACE

