"""LJ38 global minimum: FCC-truncated-octahedron seed + basin hopping + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 120.0
SAFETY = 10.0


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for N atoms (reduced units)."""
    p = pos.reshape(N, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # energy = 4 * sum_{i<j} (r^-12 - r^-6)
    eij = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(eij, 0.0)
    energy = 0.5 * eij.sum()
    # force on i from j: 4*(12 r^-12 - 6 r^-6) * (r_i - r_j)
    # grad_i = -sum_j F_ij where F_ij = 4*(12 r^-12 - 6 r^-6)*(r_i-r_j)/r^2... 
    # Use: dE/dr = 4*(-12 r^-13 + 6 r^-7); grad_i = -dE/dr * (r_i-r_j)/r
    # = 4*(12 r^-12 - 6 r^-6) * (r_i - r_j) / r^2  ... careful sign
    # Standard: F_ij (on i due to j) = 24*(2 r^-12 - r^-6) * (r_i - r_j)/r^2  (attractive/repulsive)
    # grad_i = -F_ij sum. Let's derive cleanly.
    # E_pair = 4*(r^-12 - r^-6). dE_pair/d r_vec_i = 4*(-12 r^-14 + 6 r^-8) * (r_i - r_j)/r
    # = 4*( -12 r^-14 + 6 r^-8 ) * (r_i-r_j) / r
    # = 4*(12 r^-12 - 6 r^-6) * (r_j - r_i) / r^2  ... let's just use chain rule:
    # dE/dr2 = 4*(-6 r^-14 + 3 r^-8)  (since r^-12=(r2)^-6, d/dr2 = -6 r2^-7; r^-6=r2^-3, d=-3 r2^-4)
    # dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4) = 4*(-6*inv_r2^7 + 3*inv_r2^4)
    # grad_i = dE/dr2 * d(r2)/d(r_i) = dE/dr2 * 2*(r_i - r_j)
    de_dr2 = 4.0 * (-6.0 * inv_r2 ** 7 + 3.0 * inv_r2 ** 4)
    # de_dr2 shape (N,N); grad_i = sum_j 2*de_dr2[i,j]*(r_i - r_j)
    # = 2*sum_j de_dr2[i,j]*r_i - 2*sum_j de_dr2[i,j]*r_j
    grad = 2.0 * (de_dr2.sum(axis=1, keepdims=True) * p - (de_dr2 * p.T).sum(axis=1))
    return energy, grad


def _fcc_seed():
    """Build a face-centred-cubic truncated-octahedron of 38 atoms (LJ minimum geometry).
    FCC lattice with cubic close-packed layers; take 38 nearest atoms to origin."""
    a = 2.0 ** 0.25  # LJ equilibrium spacing = 2^(1/6) ~ 1.12246
    # FCC basis vectors
    b1 = a * np.array([1, 0, 0])
    b2 = a * np.array([0, 1, 0])
    b3 = a * np.array([0, 0, 1])
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ], dtype=float) * a
    # generate FCC lattice points in a box
    pts = []
    R = 4
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for b in basis:
                    pts.append(i * b1 + j * b2 + k * b3 + b)
    pts = np.array(pts)
    # centre on origin
    pts -= pts.mean(axis=0)
    # sort by distance from origin, take N
    dist = np.linalg.norm(pts, axis=1)
    order = np.argsort(dist)
    pts = pts[order[:N]]
    # centre again
    pts -= pts.mean(axis=0)
    return pts


def _valid(pos):
    p = pos.reshape(N, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    return (r2.min() >= 0.5 ** 2)


def _local_min(pos):
    res = minimize(_lj_energy_and_grad, pos, method='L-BFGS-B',
                   jac=True, options={'maxiter': 200, 'gtol': 1e-8})
    if res.success or res.fun < 1e9:
        return res.x, res.fun
    return pos, _lj_energy_and_grad(pos)[0]


def _perturb(pos, scale):
    """Random displacement preserving validity (retry a few times)."""
    rng = np.random.default_rng()
    for _ in range(20):
        cand = pos + rng.normal(0.0, scale, size=pos.shape)
        if _valid(cand):
            return cand
    return pos


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # start from FCC seed
    pos0 = _fcc_seed()
    best_pos, best_e = _local_min(pos0)

    def elapsed():
        return time.time() - t0

    # basin hopping
    scales = [0.3, 0.5, 0.7, 1.0]
    it = 0
    while elapsed() < DEADLINE - SAFETY:
        scale = scales[it % len(scales)]
        cand = _perturb(best_pos, scale)
        new_pos, new_e = _local_min(cand)
        if new_e < best_e:
            best_pos, best_e = new_pos, new_e
        it += 1
        # occasionally try a fresh FCC seed perturbed
        if it % 5 == 0:
            alt = _fcc_seed() + rng.normal(0.0, 0.2, size=(N, 3))
            if _valid(alt):
                ap, ae = _local_min(alt)
                if ae < best_e:
                    best_pos, best_e = ap, ae

    # final tight polish with more iterations
    if elapsed() < DEADLINE - 2.0:
        res = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                       jac=True, options={'maxiter': 500, 'gtol': 1e-10})
        if res.fun < best_e:
            best_pos, best_e = res.x, res.fun

    return best_pos.reshape(N, 3)
