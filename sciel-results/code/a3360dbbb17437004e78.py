"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)


def _forces(c, R):
    """Compute pairwise and wall forces for target radius R."""
    n = len(c)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, 1e-12)
    mag = np.maximum(2.0 * R - d, 0.0) / d
    np.fill_diagonal(mag, 0.0)
    f = (mag[:, :, None] * diff).sum(axis=1)
    # Wall forces: push inward if too close to boundary
    f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
    f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
    return f


def _langevin_relax(c, R, iters, eta0, eta1, temp0, temp1, rng, t0, deadline):
    """
    Langevin dynamics relaxation with time-dependent temperature and step size.
    Particles experience deterministic packing forces + stochastic thermal kicks.
    Temperature decays from temp0 to temp1, step size from eta0 to eta1.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if (time.time() - t0) > deadline - 3:
            break
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        temp = temp0 + (temp1 - temp0) * t
        
        f = _forces(c, R)
        
        # Langevin update: deterministic force + stochastic kick
        if temp > 1e-10:
            kick = rng.standard_normal((n, 2)) * temp
            c += eta * f + kick
        else:
            c += eta * f
        
        # Clip to square (hard boundary)
        c = np.clip(c, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return c


def _grow_langevin(start, R0, R1, step, iters_per_step, eta0, eta1, temp0, temp1, rng, t0, deadline):
    """
    Grow radius from R0 to R1 using Langevin relaxation at each step.
    Keeps track of best configuration found.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius(c)
    R = R0
    while R <= R1:
        if (time.time() - t0) > deadline - 5:
            break
        c = _langevin_relax(c, R, iters_per_step, eta0, eta1, temp0, temp1, rng, t0, deadline)
        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _init_7x7_plus():
    """7x7 grid (49 points) plus one at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jittered_grid(rng, m=8, jitter=0.02):
    """m x m grid with jitter, pick N points."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-jitter, jitter, c.shape)
    return np.clip(c, -0.48, 0.48)


def _init_hexagonal(rng):
    """Hexagonal packing initial guess, perturbed."""
    # Hexagonal lattice spacing ~ 2*R for optimal packing
    # For 50 circles, roughly 7x8 hex grid
    pts = []
    row = 0
    while len(pts) < N:
        for col in range(10):
            x = (col - 4.5) * 0.14
            y = (row - 3.5) * 0.122
            if col % 2 == 1:
                y += 0.061
            if abs(x) < 0.48 and abs(y) < 0.48:
                pts.append([x, y])
        row += 1
    c = np.array(pts[:N])
    c += rng.uniform(-0.01, 0.01, c.shape)
    return np.clip(c, -0.48, 0.48)


def _init_random(rng):
    """Random uniform initialization."""
    return rng.uniform(-0.45, 0.45, (N, 2))


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = -1.0
    
    # Multiple diverse initializations with Langevin growth
    inits = [
        ("7x7+center", _init_7x7_plus()),
        ("jitter8", _init_jittered_grid(rng, m=8, jitter=0.02)),
        ("hex", _init_hexagonal(rng)),
        ("rand", _init_random(rng)),
        ("jitter7", _init_jittered_grid(rng, m=7, jitter=0.015)),
    ]
    
    for name, start in inits:
        if (time.time() - t0) > deadline - 20:
            break
        
        # Growth with Langevin dynamics
        # Start at moderate radius, grow toward optimal
        c, r = _grow_langevin(
            start, 
            R0=0.05, R1=0.0715, step=0.0002,
            iters_per_step=500,
            eta0=0.02, eta1=0.001,
            temp0=0.005, temp1=0.0,
            rng=rng, t0=t0, deadline=deadline
        )
        if r > best_r:
            best_r = r
            best_c = c
    
    if best_c is None:
        best_c = _init_7x7_plus()
        best_r = _radius(best_c)
    
    # Final annealing polish: very low temperature, fine steps
    if (time.time() - t0) < deadline - 5:
        r_now = _radius(best_c)
        c_polished = _langevin_relax(best_c, r_now, 3000, 0.005, 0.0002, 0.001, 0.0, rng, t0, deadline)
        r_polished = _radius(c_polished)
        if r_polished >= r_now:
            best_c = c_polished
            best_r = r_polished
    
    # Ultra-fine final pass
    if (time.time() - t0) < deadline - 2:
        r_now = _radius(best_c)
        c_final = _langevin_relax(best_c, r_now, 2000, 0.002, 0.0001, 0.0003, 0.0, rng, t0, deadline)
        r_final = _radius(c_final)
        if r_final >= r_now:
            best_c = c_final
            best_r = r_final
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(best_c)) or _radius(best_c) <= 0:
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
