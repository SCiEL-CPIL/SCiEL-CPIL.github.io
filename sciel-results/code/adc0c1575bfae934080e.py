def _de_loop(deadline):
    np.random.seed(0)
    # seed population
    # lattice 4x3 parallelogram
    e1 = np.array([1.5, SQRT3/2])
    e2 = np.array([0.0, SQRT3])
    pts = []
    for a in range(4):
        for b in range(3):
            pts.append(a*e1 + b*e2)
    base = np.array(pts)  # (12,2)
    # center it
    base = base - base.mean(axis=0)
    
    P = 120
    pop = np.zeros((P, 24))
    # first few individuals: base + small noise
    for i in range(min(20, P)):
        c = base + 0.3*np.random.randn(12,2)
        pop[i] = c.flatten()
    # rest: random in bounds
    for i in range(20, P):
        c = np.random.uniform(-5,5,(12,2))
        pop[i] = c.flatten()
    
    F = 0.8
    CR = 0.9
    bounds_lo = -5.0
    bounds_hi = 5.0
    
    best_c = None
    best_score = np.inf
    best_s = None
    best_pen = None
    
    # evaluate initial pop
    scores = np.zeros(P)
    for i in range(P):
        C = pop[i].reshape(12,2)
        s_, s_req, pen = fitness(C)
        scores[i] = s_
        if s_ < best_score:
            best_score = s_
            best_c = C.copy()
            best_s = s_req
            best_pen = pen
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        for i in range(P):
            # pick r1,r2,r3 distinct from i
            r1, r2, r3 = np.random.choice([x for x in range(P) if x != i], 3, replace=False)
            mutant = pop[r1] + F*(pop[r2]-pop[r3])
            # binomial crossover
            trial = pop[i].copy()
            jrand = np.random.randint(24)
            for j in range(24):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = mutant[j]
            # clip
            np.clip(trial, bounds_lo, bounds_hi, out=trial)
            C = trial.reshape(12,2)
            s_, s_req, pen = fitness(C)
            if s_ <= scores[i]:
                pop[i] = trial
                scores[i] = s_
                if s_ < best_score:
                    best_score = s_
                    best_c = C.copy()
                    best_s = s_req
                    best_pen = pen
    return best_c, best_s, best_pen
