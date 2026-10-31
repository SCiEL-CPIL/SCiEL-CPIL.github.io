def construct():
    t0 = time.time()
    deadline = t0 + 190.0  # safety margin before 210
    x, w, H, Hd = _build()
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_val, _ = _objective_and_grad(seed, x, w, H, Hd)
    best_c = seed.copy()
    best_val = seed_val
    # multistart
    starts = [seed.copy()]
    rng = np.random.default_rng(0)
    for _ in range(40):
        starts.append(rng.normal(0, 1, 3)*0.5 + seed)
    for s in starts:
        if time.time() > deadline: break
        try:
            res = minimize(_objective_and_grad, s, args=(x,w,H,Hd), method='L-BFGS-B', jac=True, options={'maxiter': 200, 'maxfun': 5000, 'gtol':1e-10})
            if res.fun < best_val:
                best_val = res.fun; best_c = res.x.copy()
        except Exception:
            pass
    # rescale to seed's 4-coeff norm
    def norm4(c):
        c12 = -(c[0]+12*c[1]+1680*c[2])/665280.0
        return np.linalg.norm([c[0],c[1],c[2],c12])
    ns = norm4(seed); nb = norm4(best_c)
    if nb > 0:
        best_c = best_c * (ns/nb)
    return np.array(best_c, dtype=float)
