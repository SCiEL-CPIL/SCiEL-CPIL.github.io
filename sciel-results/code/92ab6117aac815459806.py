<<<<<<< SEARCH
def _multi_beta_lse(points, start_time, deadline):
    """
    Multi-stage beta cascade with progressively tighter L-BFGS tolerances.
    """
    best = _normalize(points)
    best_cr = _covering_radius(best)

    stages = [
        (15.0, 1e-10, 1e-8, 200, 300),
        (50.0, 1e-12, 1e-10, 300, 400),
        (150.0, 1e-14, 1e-12, 300, 400),
        (300.0, 1e-15, 1e-13, 300, 400),
        (500.0, 1e-16, 1e-14, 300, 400),
    ]

    for beta, ftol, gtol, maxiter, maxfun in stages:
        if deadline is not None and time.time() >= deadline:
            break
        # time budget per stage
        remaining = max(1.0, deadline - time.time()) if deadline else 10.0
        # allocate up to 12s per stage, but not more than remaining
        stage_time = min(12.0, remaining * 0.6)

        # run L-BFGS with a local deadline
        local_deadline = time.time() + stage_time
        pts, cr = _lbfgs_smooth(
            best,
            start_time=start_time,
            deadline=local_deadline,
            beta=beta,
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            maxfun=maxfun,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

        # optional Riesz polish at lower beta to diversify
        if deadline is not None and time.time() < deadline - 5.0:
            riesz_deadline = time.time() + min(6.0, max(0.5, deadline - time.time() - 5.0))
            pts2, cr2 = _lbfgs_riesz(
                best,
                start_time=start_time,
                deadline=riesz_deadline,
                s=12.0,
                maxiter=150,
                ftol=1e-12,
                gtol=1e-10,
                maxfun=300,
            )
            if cr2 < best_cr:
                best_cr = cr2
                best = pts2.copy()

    return best, best_cr
=======
def _multi_beta_lse(points, start_time, deadline):
    """
    Multi-stage beta cascade with progressively tighter L-BFGS tolerances.
    Allocates more time to later (higher-precision) stages.
    """
    best = _normalize(points)
    best_cr = _covering_radius(best)

    stages = [
        (15.0, 1e-10, 1e-8, 150, 250),
        (50.0, 1e-12, 1e-10, 200, 350),
        (150.0, 1e-14, 1e-12, 250, 400),
        (300.0, 1e-15, 1e-13, 300, 450),
        (600.0, 1e-16, 1e-14, 350, 500),
    ]

    # Allocate time: earlier stages get less, later stages get more
    # Total available time for this function
    total_time = max(1.0, deadline - time.time())
    # Weights: [0.1, 0.15, 0.2, 0.25, 0.3]
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    weights /= weights.sum()
    stage_times = weights * total_time * 0.85  # keep 15% reserve

    for i, (beta, ftol, gtol, maxiter, maxfun) in enumerate(stages):
        if deadline is not None and time.time() >= deadline:
            break
        stage_time = min(stage_times[i], max(1.0, deadline - time.time()))

        local_deadline = time.time() + stage_time
        pts, cr = _lbfgs_smooth(
            best,
            start_time=start_time,
            deadline=local_deadline,
            beta=beta,
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            maxfun=maxfun,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

        # Riesz polish only in middle stages to diversify
        if i in [1, 2] and deadline is not None and time.time() < deadline - 5.0:
            riesz_deadline = time.time() + min(4.0, max(0.5, deadline - time.time() - 5.0))
            pts2, cr2 = _lbfgs_riesz(
                best,
                start_time=start_time,
                deadline=riesz_deadline,
                s=8.0,
                maxiter=100,
                ftol=1e-12,
                gtol=1e-10,
                maxfun=200,
            )
            if cr2 < best_cr:
                best_cr = cr2
                best = pts2.copy()

    return best, best_cr
>>>>>>> REPLACE

