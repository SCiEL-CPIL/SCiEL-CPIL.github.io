"""
Strategy: Stochastic many-body repulsion Molecular Dynamics (soft-sphere / Lennard-Jones)
annealing to pack 50 equal spheres in the unit cube [-1/2,1/2]^3, followed by a short
L-BFGS-B polish on a log-sum-exp surrogate for final precision.

This is a materially different design family from the seed:
- The seed uses deterministic Nesterov projected gradient ascent on an exact min-distance
  subgradient, then an LSE L-BFGS polish.
- Here we model the spheres as particles in a smooth many-body potential (Lennard-Jones
  core + linear wall repulsion) and run a Langevin/velocity-Verlet molecular dynamics
  simulation with a cooling schedule. The dynamics naturally explore the configuration
  space via stochastic thermal motion + strong repulsive forces, which is a different
  search mechanism (physics-based stochastic exploration) rather than a smooth-surrogate
  gradient method.
- The objective of the dynamics is a smooth, differentiable energy (not the non-smooth
  min), so the system can flow through the landscape and escape shallow local minima that
  trap pure gradient methods.
- A final short LSE polish refines the best MD configuration.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centres):
    """Exact maximum common radius for centres in [-0.5, 0.5]^3."""
    n = len(centres)
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    if n < 2:
        return min_face
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_face, min_pair / 2.0)


# ------------------------- Molecular Dynamics -------------------------

def _lj_energy_and_force(centres, sigma=0.24, epsilon=1.0, wall_softness=5.0):
    """
    Lennard-Jones-like repulsive energy and forces + soft wall repulsion.
    We use only the repulsive (1/r^12 - 1/r^6) core, but to keep it smooth and
    purely repulsive we use a softened repulsive potential:
        U_pair = epsilon * ( (sigma/r)^12 - 2*(sigma/r)^6 )   [min at r=sigma, U=0]
    Actually we want spheres of radius r to be non-overlapping, so the natural
    "contact" distance is 2r. We set sigma = 2r_target so the LJ minimum sits at
    the target contact distance. The repulsive core pushes apart when r < sigma.
    Wall: U_wall = wall_softness * sum_i (max(0, |c_ik| - 0.5 + margin))^2  (quadratic)
    But to keep it smooth we use a soft hinge: w * (0.5 - |c|)^{-p} style is singular.
    Instead use U_wall = K * sum_i sum_k max(0, |c_ik| - (0.5 - pad))^2  with pad small.
    We'll use a quadratic penalty that turns on near the wall.
    """
    n = len(centres)
    energy = 0.0
    force = np.zeros_like(centres)

    # Pairwise LJ (repulsive core)
    for i in range(n):
        for j in range(i + 1, n):
            d = centres[i] - centres[j]
            r2 = np.dot(d, d)
            r = np.sqrt(r2) if r2 > 1e-12 else 1e-6
            sr = sigma / r
            sr6 = sr ** 6
            sr12 = sr6 * sr6
            # LJ: 4*eps*((sig/r)^12 - (sig/r)^6); force magnitude = 24*eps*(2*sr12 - sr6)/r
            # We use a pure repulsive form to avoid attraction: U = eps*sr12 - 2*eps*sr6
            # dU/dr = -12*eps*sr12/r + 12*eps*sr6/r = 12*eps*sr6*(sr6 - sr12)/r ... 
            # Let's just use standard LJ and rely on repulsive core at short range.
            u = 4.0 * epsilon * (sr12 - sr6)
            # force on i from j: F_i = -dU/dc_i ; dU/dr * d r/d c_i ; r = |c_i - c_j|, d r/d c_i = d/r
            # dU/dr = 4*eps*(-12*sr12 + 6*sr6)/r = 24*eps*(sr6 - 2*sr12)/r
            dUdr = 24.0 * epsilon * (sr6 - 2.0 * sr12) / r
            fvec = dUdr * (d / r)  # force on i (repulsive when r<sigma gives positive push apart)
            force[i] += fvec
            force[j] -= fvec
            energy += u

    # Wall quadratic penalty
    pad = 0.0
    for i in range(n):
        c = centres[i]
        for k in range(3):
            a = np.abs(c[k])
            if a > 0.5 - pad:
                over = a - (0.5 - pad)
                energy += wall_softness * over * over
                # dU/dc_ik = wall_softness * 2*over * sign(c_ik)
                force[i, k] -= wall_softness * 2.0 * over * np.sign(c[k])

    return energy, force


def _md_anneal(centres, time_deadline, sigma=0.24, epsilon=1.0, wall_softness=5.0,
               dt=0.002, T0=0.05, Tend=0.0005, nsteps_target=20000, mass=1.0,
               damping=0.0):
    """
    Velocity-Verlet MD with Langevin thermostat and linear cooling.
    Returns best configuration by radius seen during the run.
    """
    c = centres.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _compute_radius(c)

    E, F = _lj_energy_and_force(c, sigma, epsilon, wall_softness)
    a = F / mass

    nsteps = nsteps_target
    cooling = np.linspace(T0, Tend, nsteps)

    rng = np.random.RandomState(12345)

    for step in range(nsteps):
        if time.time() > time_deadline:
            break
        T = cooling[step]

        # Velocity Verlet (half kick, drift, half kick)
        v += 0.5 * dt * a
        c += dt * v
        # project into cube (hard wall) to keep valid
        c = np.clip(c, -0.4999, 0.4999)

        E2, F2 = _lj_energy_and_force(c, sigma, epsilon, wall_softness)
        a2 = F2 / mass
        v += 0.5 * dt * a2

        # Langevin thermostat: v = v*(1 - gamma*dt) + sqrt(2*T*dt/mass)*noise
        if damping > 0:
            v *= (1.0 - damping * dt)
        noise_scale = np.sqrt(2.0 * T * dt / mass)
        v += noise_scale * rng.standard_normal(c.shape)

        a = a2

        # Track best by radius
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    return best_c, best_r


# ------------------------- LSE polish -------------------------

def _lse_objective_and_grad(x, beta=800.0):
    c = x.reshape(N, 3)
    n = N
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    wall_vals = 0.5 - np.max(np.abs(c), axis=1)
    all_vals = np.concatenate([pair_vals, wall_vals])

    max_val = np.max(all_vals)
    exp_vals = np.exp(beta * (all_vals - max_val))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    lse = max_val + np.log(sum_exp) / beta
    val = -lse

    grad_c = np.zeros_like(c)
    # pair
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = weights[idx]
        u = (c[i] - c[j]) / d
        grad_c[i] -= w * u / 2.0
        grad_c[j] += w * u / 2.0
    # wall
    for idx in range(n):
        w = weights[len(pair_vals) + idx]
        i = idx
        cc = c[i]
        axis = np.argmax(np.abs(cc))
        g = np.zeros(3)
        g[axis] = -np.sign(cc[axis])
        grad_c[i] -= w * g

    return val, grad_c.reshape(-1)


def _lbgfgs_polish(centres, beta=800.0, maxiter=150, time_deadline=None):
    def obj_and_grad(x):
        return _lse_objective_and_grad(x, beta=beta)

    x0 = centres.reshape(-1)
    result = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12})
    if time_deadline is not None and time.time() > time_deadline:
        return centres
    x_opt = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
    return x_opt


# ------------------------- Initial configurations -------------------------

def _grid_start(m=4, seed=0):
    rng = np.random.RandomState(seed)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng.shuffle(grid)
    return grid[:N].copy()


def _fcc_start(seed=0):
    rng = np.random.RandomState(seed)
    a = 1.0 / 4.0
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) * a - 0.5
                y = (j + 0.5) * a - 0.5
                z = (k + 0.5) * a - 0.5
                pts.append([x, y, z])
                if i < 3:
                    pts.append([x + a/2, y + a/2, z])
                    pts.append([x + a/2, y, z + a/2])
                    pts.append([x, y + a/2, z + a/2])
    pts = np.array(pts)
    rng.shuffle(pts)
    return pts[:N].copy()


def _random_start(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin

    starts = [
        _grid_start(4, 1),
        _grid_start(5, 2),
        _fcc_start(3),
        _random_start(4),
        _random_start(5),
    ]

    best_centres = None
    best_r = 0.0

    # MD annealing parameters (tuned for 50 spheres, target radius ~0.136 => sigma ~0.272)
    # We'll try a couple of sigma values around the expected contact distance.
    sigma_schedule = [0.272, 0.26, 0.28]

    for si, init in enumerate(starts):
        if time.time() > deadline - 40.0:
            break
        for sigma in sigma_schedule:
            if time.time() > deadline - 20.0:
                break
            # Run MD anneal
            c_md, r_md = _md_anneal(
                init, time_deadline=deadline - 15.0,
                sigma=sigma, epsilon=1.0, wall_softness=5.0,
                dt=0.002, T0=0.04, Tend=0.0004, nsteps_target=15000,
                mass=1.0, damping=0.5
            )
            if r_md > best_r:
                best_r = r_md
                best_centres = c_md.copy()

            # Short LSE polish on the MD result
            if time.time() < deadline - 5.0:
                c_pol = _lbgfgs_polish(c_md, beta=800.0, maxiter=80,
                                       time_deadline=deadline - 3.0)
                r_pol = _compute_radius(c_pol)
                if r_pol > best_r:
                    best_r = r_pol
                    best_centres = c_pol.copy()

    # Final high-precision polish on the overall best
    if best_centres is not None and time.time() < deadline - 3.0:
        final = _lbgfgs_polish(best_centres, beta=1200.0, maxiter=200,
                               time_deadline=deadline - 1.0)
        r_final = _compute_radius(final)
        if r_final > best_r:
            best_r = r_final
            best_centres = final

    if best_centres is None:
        best_centres = starts[0]

    return best_centres
