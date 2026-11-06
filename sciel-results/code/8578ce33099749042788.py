"""Gradient-based antireflection coating design with analytic derivatives."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _wavelengths(problem):
    b0, b1 = problem["band_nm"]
    return np.linspace(b0, b1, 401)


def _reflectance_and_grad_v2(th, nvec, n0, ns, wls):
    """Return (mean reflectance, gradient d(mean)/d(th)) via chain rule."""
    n = len(th)
    M = len(wls)
    i_ = 1j

    # phase delta[m,j] = pi * nvec[j] * th[j] / wls[m]
    delta = np.zeros((M, n))
    for j in range(n):
        delta[:, j] = np.pi * nvec[j] * th[j] / wls

    c = np.cos(delta)
    s = np.sin(delta)

    a = c
    b = i_ * s / nvec[None, :]
    cc = i_ * nvec[None, :] * s
    d = c

    # forward products F_j = M_0...M_{j-1}
    F = np.zeros((M, 2, 2, n + 1), dtype=complex)
    F[:, :, :, 0] = np.eye(2)
    for j in range(n):
        Mj = np.zeros((M, 2, 2), dtype=complex)
        Mj[:, 0, 0] = a[:, j]
        Mj[:, 0, 1] = b[:, j]
        Mj[:, 1, 0] = cc[:, j]
        Mj[:, 1, 1] = d[:, j]
        Fj = F[:, :, :, j]
        F[:, :, :, j + 1] = Fj[:, :, 0:1, :] @ Mj[:, 0:1, :] + Fj[:, :, 1:2, :] @ Mj[:, 1:2, :]

    # backward products B_j = M_j...M_{n-1}
    B = np.zeros((M, 2, 2, n + 1), dtype=complex)
    B[:, :, :, n] = np.eye(2)
    for j in range(n - 1, -1, -1):
        Mj = np.zeros((M, 2, 2), dtype=complex)
        Mj[:, 0, 0] = a[:, j]
        Mj[:, 0, 1] = b[:, j]
        Mj[:, 1, 0] = cc[:, j]
        Mj[:, 1, 1] = d[:, j]
        Bj1 = B[:, :, :, j + 1]
        B[:, :, :, j] = Mj[:, 0:1, :] @ Bj1[:, 0:1, :] + Mj[:, 1:2, :] @ Bj1[:, 1:2, :]

    m11 = F[:, 0, 0, n]
    m12 = F[:, 0, 1, n]
    m21 = F[:, 1, 0, n]
    m22 = F[:, 1, 1, n]
    num = m11 + m12 * ns
    den = m21 + m22 * ns
    Y = num / den
    R = np.abs(Y) ** 2

    g11 = 1.0 / den
    g12 = ns / den
    g21 = -num / (den * den)
    g22 = -ns * num / (den * den)

    dRdY = 2.0 * np.conjugate(Y)
    G = np.zeros((M, 2, 2), dtype=complex)
    G[:, 0, 0] = dRdY * g11
    G[:, 0, 1] = dRdY * g12
    G[:, 1, 0] = dRdY * g21
    G[:, 1, 1] = dRdY * g22

    da_d = -s
    db_d = i_ * c / nvec[None, :]
    dc_d = i_ * nvec[None, :] * c
    dd_d = -s

    grad = np.zeros(n)
    for j in range(n):
        L = F[:, :, :, j]
        Rr = B[:, :, :, j + 1]
        H = np.zeros((M, 2, 2), dtype=complex)
        for a_ in range(2):
            for b_ in range(2):
                H[:, a_, b_] = np.sum(
                    L[:, :, a_][None, :] * G[:, :, :] * Rr[:, :, b_][None, :], axis=1
                )
        dRdd = (H[:, 0, 0] * da_d[:, j] + H[:, 0, 1] * db_d[:, j]
                + H[:, 1, 0] * dc_d[:, j] + H[:, 1, 1] * dd_d[:, j])
        chain = np.pi * nvec[j] / wls
        grad[j] = np.sum(np.real(dRdd) * chain)

    mean_R = np.mean(R)
    return mean_R, grad


def _objective(th, nvec, n0, ns, wls):
    return _reflectance_and_grad_v2(th, nvec, n0, ns, wls)[0]


def _make_starts(problem):
    starts = []
    band0, band1 = problem["band_nm"]
    centre = 0.5 * (band0 + band1)
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    max_layers = problem["max_layers"]

    def qw_stack(labels):
        ths = []
        for lab in labels:
            n = nL if lab == "L" else nH
            th = centre / (4.0 * n)
            th = min(max(th, 1.0), max_layer)
            ths.append(th)
        return labels, ths

    for k in range(1, min(max_layers, 12) + 1):
        labels = ["L" if i % 2 == 0 else "H" for i in range(k)]
        starts.append(qw_stack(labels))
        if k > 1:
            labels2 = ["H" if i % 2 == 0 else "L" for i in range(k)]
            starts.append(qw_stack(labels2))

    starts.append(qw_stack(["L"]))
    starts.append(qw_stack(["H"]))

    rng = np.random.default_rng(12345)
    for _ in range(24):
        k = int(rng.integers(1, max_layers + 1))
        labels = [str(rng.choice(["L", "H"])) for _ in range(k)]
        ths = [float(rng.uniform(1.0, max_layer)) for _ in range(k)]
        tot = sum(ths)
        if tot > max_total:
            scale = max_total / tot
            ths = [t * scale for t in ths]
        starts.append((labels, ths))

    for _ in range(24):
        k = int(rng.integers(2, max_layers + 1))
        labels = [str(rng.choice(["L", "H"])) for _ in range(k)]
        ths = []
        for lab in labels:
            n = nL if lab == "L" else nH
            th = centre / (4.0 * n) * float(rng.uniform(0.5, 1.5))
            th = min(max(th, 1.0), max_layer)
            ths.append(th)
        tot = sum(ths)
        if tot > max_total:
            scale = max_total / tot
            ths = [t * scale for t in ths]
        starts.append((labels, ths))

    return starts


def design(problem):
    deadline = time.time() + 130.0
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    wls = _wavelengths(problem)
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    starts = _make_starts(problem)

    best_R = None
    best_labels = None
    best_ths = None

    for labels, ths in starts:
        if time.time() > deadline - 5.0:
            break
        nvec = np.array([materials[c] for c in labels], dtype=float)
        x0 = np.array(ths, dtype=float)
        try:
            def obj(x):
                return _objective(x, nvec, n0, ns, wls)

            res = minimize(
                obj, x0, method="L-BFGS-B",
                bounds=[(0.0, max_layer)] * len(x0),
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
            )
            xopt = np.clip(res.x, 0.0, max_layer)
            tot = xopt.sum()
            if tot > max_total:
                xopt = xopt * (max_total / tot)
            R = _objective(xopt, nvec, n0, ns, wls)
            if best_R is None or R < best_R:
                best_R = R
                best_labels = list(labels)
                best_ths = [float(t) for t in xopt]
        except Exception:
            continue

    if best_labels is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_labels = ["L"]
        best_ths = [centre / (4.0 * materials["L"])]

    return best_labels, best_ths

# EVOLVE-BLOCK-END
