<<<<<<< SEARCH
def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Polish each initialization with NM
    for C in inits:
        if time.time() > deadline - 60:
            break
        Cn = optimize_nm(C, deadline, maxiter=3000)
        Sn = outer_side(Cn)
        if valid(Cn, Sn):
            if Sn < best_side:
                best_side = Sn
                best = Cn
        # Also check initial
        Si = outer_side(C)
        if valid(C, Si) and Si < best_side:
            best_side = Si
            best = C

    # 2. Rigid body optimization: treat cluster as single object, optimize rotation+translation
    if best is not None:
        if time.time() < deadline - 20:
            Cn = optimize_rigid(best, deadline, maxiter=5000)
            Sn = outer_side(Cn)
            if valid(Cn, Sn) and Sn < best_side:
                best_side = Sn
                best = Cn

    # 3. If nothing valid, fallback
    if best is None:
        C = init_flower()
        Cn = optimize_nm(C, deadline, maxiter=30000)
        Sn = outer_side(Cn)
        if not valid(Cn, Sn):
            # fallback: spread out widely (valid but large)
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            Sn = outer_side(Cn)
        best = Cn

    Cn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
=======
def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Polish each initialization with NM
    for C in inits:
        if time.time() > deadline - 60:
            break
        Cn = optimize_nm(C, deadline, maxiter=3000)
        Sn = outer_side(Cn)
        if valid(Cn, Sn):
            if Sn < best_side:
                best_side = Sn
                best = Cn
        # Also check initial
        Si = outer_side(C)
        if valid(C, Si) and Si < best_side:
            best_side = Si
            best = C

    # 2. Try differential evolution from best found so far
    if best is not None:
        if time.time() < deadline - 30:
            Cn = optimize_de(best, deadline, popsize=15, maxiter=50)
            Sn = outer_side(Cn)
            if valid(Cn, Sn) and Sn < best_side:
                best_side = Sn
                best = Cn
            # Polish the DE result with NM
            if time.time() < deadline - 10:
                Cn2 = optimize_nm(Cn, deadline, maxiter=2000)
                Sn2 = outer_side(Cn2)
                if valid(Cn2, Sn2) and Sn2 < best_side:
                    best_side = Sn2
                    best = Cn2

    # 3. Rigid body optimization: treat cluster as single object, optimize rotation+translation
    if best is not None:
        if time.time() < deadline - 20:
            Cn = optimize_rigid(best, deadline, maxiter=5000)
            Sn = outer_side(Cn)
            if valid(Cn, Sn) and Sn < best_side:
                best_side = Sn
                best = Cn

    # 4. If nothing valid, fallback
    if best is None:
        C = init_flower()
        Cn = optimize_nm(C, deadline, maxiter=30000)
        Sn = outer_side(Cn)
        if not valid(Cn, Sn):
            # fallback: spread out widely (valid but large)
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            Sn = outer_side(Cn)
        best = Cn

    Cn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
>>>>>>> REPLACE

