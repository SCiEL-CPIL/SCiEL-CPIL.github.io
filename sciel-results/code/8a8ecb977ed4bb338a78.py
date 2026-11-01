def _optimize(pos0, r0, wall, deadline):
    k = 100.0
    dt = 0.005
    gamma = 2.0
    T0 = 0.005
    inner_steps = 150
    grow_frac = 0.005
    pos = pos0.copy().astype(float)
    vel = np.zeros_like(pos)
    r_feasible = r0
    best_r = r0
    best_pos = pos0.copy()
    stall = 0
    while time.time() < deadline:
        r_try = r_feasible * (1.0 + grow_frac)
        # small perturbation to escape local minima
        pos = pos + 0.003*np.random.randn(*pos.shape)
        # inner dynamics
        for s in range(inner_steps):
            T = T0*(1.0 - s/inner_steps)
            F = _forces(pos, r_try, wall, k)
            vel = (1.0 - gamma*dt)*vel + F*dt
            if T > 1e-9:
                vel += np.sqrt(2.0*gamma*T*dt)*np.random.randn(*pos.shape)
            pos = pos + vel*dt
        # measure
        r_ach = _min_margin(pos)
        if r_ach >= r_try*0.999:
            r_feasible = r_try
            if r_ach > best_r:
                best_r = r_ach
                best_pos = pos.copy()
            stall = 0
        else:
            stall += 1
            if stall > 20:
                # stuck: reset to best and bigger perturbation
                pos = best_pos.copy()
                vel = np.zeros_like(pos)
                pos = pos + 0.02*np.random.randn(*pos.shape)
                stall = 0
    return best_pos, best_r
