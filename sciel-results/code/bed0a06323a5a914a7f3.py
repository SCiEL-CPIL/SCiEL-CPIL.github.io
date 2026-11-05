import time
import numpy as np

_DEADLINE = None
_DEADLINE_BUDGET = 150.0
_DEADLINE_MARGIN = 12.0


def _now():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_BUDGET - _DEADLINE_MARGIN
    return time.time()


def _deadline_hit():
    return _now() > _DEADLINE


def _reflectance_matrix(mat, thicknesses, lambdas, n0, ns):
    """Return mean reflectance in percent over the band (401 points, 400..800 nm)."""
    w = 2.0 * np.pi / lambdas
    m11 = np.ones(len(lambdas), dtype=complex)
    m12 = np.zeros(len(lambdas), dtype=complex)
    m21 = np.zeros(len(lambdas), dtype=complex)
    m22 = np.ones(len(lambdas), dtype=complex)
    for n, d in zip(mat, thicknesses):
        beta = n * d * w
        c = np.cos(beta)
        s = np.sin(beta)
        t11 = m11 * c + m12 * (1j * s / n)
        t12 = m11 * (1j * n * s) + m12 * c
        t21 = m21 * c + m22 * (1j * s / n)
        t22 = m21 * (1j * n * s) + m22 * c
        m11, m12, m21, m22 = t11, t12, t21, t22
    y = (ns * m11 + m12) / (ns * m21 + m22)
    r = (y - n0) / (y + n0)
    return np.mean(np.abs(r) ** 2) * 100.0


def _make_lambdas():
    return np.linspace(400.0, 800.0, 401)


def _valid(mat, thicknesses, problem):
    ml = problem["max_layers"]
    mt = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    if len(mat) > ml:
        return False
    for d in thicknesses:
        if d < -1e-6 or d > max_layer + 1e-6:
            return False
    if sum(thicknesses) > mt + 1e-6:
        return False
    return True


def _clamp(thicknesses, max_layer):
    return np.clip(thicknesses, 0.0, max_layer)


def _objective(theta, mat, lambdas, n0, ns, problem):
    thicknesses = _clamp(theta, problem["max_layer_nm"])
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return 1e9
    return _reflectance_matrix(mat, thicknesses, lambdas, n0, ns)


def _nm_initial(mat, lambdas, n0, ns, problem, x0):
    """Nelder-Mead with a hard deadline guard."""
    n = len(x0)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    x = np.array(x0, dtype=float)
    x = _clamp(x, max_layer)

    # initial simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.05 if x[i] == 0 else 0.05 * abs(x[i])
        xi[i] += step
        xi = _clamp(xi, max_layer)
        simplex.append(xi)

    fvals = []
    for i in range(n + 1):
        if sum(simplex[i]) > max_total + 1e-6:
            fvals.append(1e9)
        else:
            fvals.append(_reflectance_matrix(mat, simplex[i], lambdas, n0, ns))

    best_i = int(np.argmin(fvals))
    best_x = simplex[best_i].copy()
    best_f = fvals[best_i]

    max_iter = 600
    for it in range(max_iter):
        if _deadline_hit():
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()

        centroid = np.mean(simplex[:-1], axis=0)
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _clamp(xr, max_layer)
        fr = 1e9 if sum(xr) > max_total + 1e-6 else _reflectance_matrix(mat, xr, lambdas, n0, ns)

        if fr < fvals[0]:
            if fr < fvals[-2]:
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = _clamp(xe, max_layer)
                fe = 1e9 if sum(xe) > max_total + 1e-6 else _reflectance_matrix(mat, xe, lambdas, n0, ns)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < best_f:
                        best_f = fe
                        best_x = simplex[-1].copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                simplex[-1] = xr
                fvals[-1] = fr
                if fr < best_f:
                    best_f = fr
                    best_x = simplex[-1].copy()
        else:
            if fr < fvals[-1]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                rho = 0.5
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = _clamp(xc, max_layer)
                fc = 1e9 if sum(xc) > max_total + 1e-6 else _reflectance_matrix(mat, xc, lambdas, n0, ns)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
                        simplex[i] = _clamp(simplex[i], max_layer)
                        fvals[i] = 1e9 if sum(simplex[i]) > max_total + 1e-6 else _reflectance_matrix(mat, simplex[i], lambdas, n0, ns)
                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

        if np.max(fvals) - np.min(fvals) < 1e-6 * (1.0 + np.min(fvals)):
            break
    return best_x, best_f


def design(problem):
    global _DEADLINE
    _DEADLINE = time.time() + _DEADLINE_BUDGET - _DEADLINE_MARGIN

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    lambdas = _make_lambdas()

    # Bare glass baseline
    r0 = (n0 - ns) / (n0 + ns)
    best_f = r0 ** 2 * 100.0
    best_mat = []
    best_th = []

    # Try increasing layer counts with alternating L/H, quarter-wave-ish starts
    for n in range(1, min(max_layers, 8) + 1):
        if _deadline_hit():
            break
        # alternating starting with L (low index near air)
        for start in ([0], [1]):
            if _deadline_hit():
                break
            mat = []
            th = []
            for k in range(n):
                idx = (start + k) % 2
                if idx == 0:
                    mat.append(nL)
                    th.append(600.0 / (4.0 * nL))
                else:
                    mat.append(nH)
                    th.append(600.0 / (4.0 * nH))
            # scale to fit budget if needed
            s = sum(th)
            if s > max_total:
                scale = max_total / s
                th = [t * scale for t in th]
            th = _clamp(np.array(th), max_layer)
            f = _reflectance_matrix(mat, th, lambdas, n0, ns)
            if f < best_f:
                best_f = f
                best_mat = mat[:]
                best_th = list(th)

            # Nelder-Mead local refinement
            x0 = np.array(th)
            xf, ff = _nm_initial(mat, lambdas, n0, ns, problem, x0)
            if ff < best_f:
                best_f = ff
                best_mat = mat[:]
                best_th = list(xf)

    # Convert materials to labels
    label_map = {nL: "L", nH: "H"}
    labels = [label_map[m] for m in best_mat]
    thicknesses = [float(t) for t in best_th]

    # Final validity check
    if not _valid(labels, thicknesses, problem):
        # fall back to empty (bare glass)
        return [], []

    return labels, thicknesses
