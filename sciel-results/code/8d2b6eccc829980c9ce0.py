"""Combined strategy: robust NumPy-only Hamiltonian construction, fast matrix
exponentiation via eigendecomposition, and a manual Nelder-Mead optimizer with
symmetric starting points and time-budget management."""
import numpy as np
import time

_DEADLINE = time.time() + 135.0  # safety margin before 150s wall-clock kill
N_SITES = 6
DIM = 2 ** N_SITES
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12
EVAL_TIMES = (0.25, 0.5, 1.0)

# Precompute Hamiltonians A and B
def _build_hamiltonians():
    n = N_SITES
    dim = DIM
    delta = ANISOTROPY
    field = FIELD
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        ops = []
        for s in range(n):
            ops.append(op if s == site else I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = []
        for s in range(n):
            if s == s1:
                ops.append(op1)
            elif s == s2:
                ops.append(op2)
            else:
                ops.append(I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # A: odd bonds (1-2, 3-4) + half field
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    # B: even bonds (0-1, 2-3, 4-5) + half field
    B += xxz_bond(0, 1)
    B += xxz_bond(2, 3)
    B += xxz_bond(4, 5)
    for s in range(n):
        B += (field / 2.0) * site_op(sz, s)

    return A, B

_A = None
_B = None
_EXACT = {}
_EIG_A = None
_EIG_B = None

def _get_AB():
    global _A, _B, _EIG_A, _EIG_B
    if _A is None:
        _A, _B = _build_hamiltonians()
        _EIG_A = np.linalg.eigh(_A)
        _EIG_B = np.linalg.eigh(_B)
    return _A, _B

def _get_exact():
    global _EXACT
    if not _EXACT:
        A, B = _get_AB()
        H = A + B
        eigvals, eigvecs = np.linalg.eigh(H)
        for t in EVAL_TIMES:
            phase = np.exp(-1j * eigvals * t)
            _EXACT[t] = (eigvecs * phase) @ eigvecs.conj().T
    return _EXACT

def _matrix_exp_from_eig(eigvals, eigvecs, t):
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def _eval_error(a, b):
    """Compute mean spectral-norm error."""
    A, B = _get_AB()
    exact = _get_exact()
    total = 0.0
    for t in EVAL_TIMES:
        U = np.eye(DIM, dtype=complex)
        for k in range(STAGES):
            if b[k] != 0.0:
                Eb = _matrix_exp_from_eig(_EIG_B[0], _EIG_B[1], b[k] * t)
                U = Eb @ U
            if a[k] != 0.0:
                Ea = _matrix_exp_from_eig(_EIG_A[0], _EIG_A[1], a[k] * t)
                U = Ea @ U
        diff = U - exact[t]
        total += np.linalg.norm(diff, 2)
    return total / len(EVAL_TIMES)

def _clamp(x):
    return np.clip(x, -50.0, 50.0)

def _nelder_mead(objective, x0, max_iter=800, tol=1e-6, deadline=None):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        if abs(point[i]) > 1e-10:
            point[i] *= 1.05
        else:
            point[i] += 0.05
        simplex.append(point)

    values = [objective(p) for p in simplex]
    best_val = min(values)
    best_point = simplex[values.index(best_val)].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline - 8:
            break

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if abs(values[-1] - values[0]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)

        if fr < values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])

        current_best = min(values)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[values.index(current_best)].copy()

    return best_point, best_val

def _make_objective():
    def obj(theta):
        a_free = theta[:11]
        b_free = theta[11:22]
        a = np.empty(12)
        a[:11] = a_free
        a[11] = 1.0 - np.sum(a_free)
        b = np.empty(12)
        b[:11] = b_free
        b[11] = 1.0 - np.sum(b_free)
        a = _clamp(a)
        b = _clamp(b)
        try:
            return _eval_error(a, b)
        except Exception:
            return 1e10
    return obj

def coefficients(stages):
    """Return optimized coefficients for the 12-stage product formula."""
    start_time = time.time()

    # Ensure Hamiltonians are built
    _get_AB()
    _get_exact()

    obj = _make_objective()

    # Start from uniform (Strang-like)
    theta0 = np.ones(22) / 12.0
    best_theta = theta0.copy()
    best_val = obj(theta0)

    # Run Nelder-Mead
    try:
        res_theta, res_val = _nelder_mead(obj, theta0, max_iter=1500, tol=1e-7, deadline=_DEADLINE)
        if res_val < best_val:
            best_val = res_val
            best_theta = res_theta
    except Exception:
        pass

    # If we have time, try a second round from the best point with smaller initial simplex
    if time.time() < _DEADLINE - 20:
        theta1 = best_theta.copy()
        # Slightly perturb
        np.random.seed(123)
        theta1 += np.random.normal(0, 0.01, 22)
        try:
            res_theta2, res_val2 = _nelder_mead(obj, theta1, max_iter=800, tol=1e-7, deadline=_DEADLINE)
            if res_val2 < best_val:
                best_val = res_val2
                best_theta = res_theta2
        except Exception:
            pass

    # Extract final coefficients
    a_free = best_theta[:11]
    b_free = best_theta[11:22]
    a = np.empty(12)
    a[:11] = a_free
    a[11] = 1.0 - np.sum(a_free)
    b = np.empty(12)
    b[:11] = b_free
    b[11] = 1.0 - np.sum(b_free)
    a = _clamp(a)
    b = _clamp(b)

    # Ensure exact sum = 1
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb

    a = a.tolist()
    b = b.tolist()

    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]

    return a, b
