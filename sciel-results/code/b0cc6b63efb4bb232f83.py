# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 115.0  # seconds; evaluator kills at 120, keep safety margin
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_cz(psi, n, i):
    dim = 1 << n
    stride = 1 << i
    for base in range(0, dim, stride * 2):
        for off in range(stride):
            a = base + off
            b = a + stride
            if psi[b] != 0:
                psi[b] = -psi[b]
    return psi


def _circuit_state(thetas, n):
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for layer in range(thetas.shape[0]):
        for i in range(n):
            M = _ry_matrix(thetas[layer, i])
            stride = 1 << i
            for base in range(0, dim, stride * 2):
                for off in range(stride):
                    a = base + off
                    b = a + stride
                    na = M[0, 0] * psi[a] + M[0, 1] * psi[b]
                    nb = M[1, 0] * psi[a] + M[1, 1] * psi[b]
                    psi[a] = na
                    psi[b] = nb
        if layer < thetas.shape[0] - 1:
            for i in range(n - 1):
                _apply_cz(psi, n, i)
    return psi


def _hamiltonian(coupling, field, n):
    dim = 1 << n
    Z = np.diag([1.0 if (k >> i) & 1 == 0 else -1.0 for k in range(dim)])
    X = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        X[k ^ 1, k] = 1.0
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        Zi = np.kron(Z, np.eye(2)) if i < n - 1 else np.eye(2)  # placeholder
    # build ZZ and X terms properly
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        Zi = np.eye(2)
        Zj = np.eye(2)
        # construct operator for qubit i (bit i) and i+1
        ops = []
        for q in range(n):
            if q == i:
                ops.append(np.diag([1.0, -1.0]))
            elif q == i + 1:
                ops.append(np.diag([1.0, -1.0]))
            else:
                ops.append(np.eye(2))
        # tensor in order q=0..n-1
        op = ops[0]
        for q in range(1, n):
            op = np.kron(op, ops[q])
        H = H - coupling * op
    for i in range(n):
        ops = []
        for q in range(n):
            if q == i:
                ops.append(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
            else:
                ops.append(np.eye(2))
        op = ops[0]
        for q in range(1, n):
            op = np.kron(op, ops[q])
        H = H - field * op
    return H


def _ground_energy(coupling, field, n):
    H = _hamiltonian(coupling, field, n)
    vals = np.linalg.eigvalsh(H)
    return float(vals[0])


def _energy(psi, coupling, field, n):
    dim = 1 << n
    Z = np.zeros((dim, dim), dtype=complex)
    X = np.zeros((dim, dim), dtype=complex)
    # Z_i
    Zs = []
    Xs = []
    for i in range(n):
        ops = []
        for q in range(n):
            if q == i:
                ops.append(np.diag([1.0, -1.0]))
            else:
                ops.append(np.eye(2))
        op = ops[0]
        for q in range(1, n):
            op = np.kron(op, ops[q])
        Zs.append(op)
        ops = []
        for q in range(n):
            if q == i:
                ops.append(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
            else:
                ops.append(np.eye(2))
        op = ops[0]
        for q in range(1, n):
            op = np.kron(op, ops[q])
        Xs.append(op)
    E = 0.0
    for i in range(n - 1):
        ZZ = Zs[i] @ Zs[i + 1]
        E += -coupling * (psi.conj() @ (ZZ @ psi)).real
    for i in range(n):
        E += -field * (psi.conj() @ (Xs[i] @ psi)).real
    return float(E)


def _objective(params, n, coupling, field, ge):
    thetas = params.reshape(3, n)
    psi = _circuit_state(thetas, n)
    e = _energy(psi, coupling, field, n)
    return (e - ge) / n


def angles(n, coupling, field, layers):
    global _START
    _START = time.time()
    dim = 1 << n
    ge = _ground_energy(coupling, field, n)

    # candidate initial guesses
    guesses = []
    # all pi/2 (product |+>)
    guesses.append(np.full((layers, n), np.pi / 2))
    # all zero (product |0>)
    guesses.append(np.zeros((layers, n)))
    # alternating
    g = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n):
            g[l, i] = np.pi / 2 * (1 if (i % 2 == 0) else -1)
    guesses.append(g)
    # small random
    rng = np.random.RandomState(12345)
    for _ in range(4):
        guesses.append(rng.uniform(-np.pi / 2, np.pi / 2, size=(layers, n)))

    def obj(p):
        return _objective(p, n, coupling, field, ge)

    best = None
    best_val = np.inf
    for guess in guesses:
        if _remaining() < 8.0:
            break
        try:
            r = minimize(obj, guess.ravel(), method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            v = r.fun
            if v < best_val:
                best_val = v
                best = r.x
        except Exception:
            pass

    # L-BFGS-B polish
    if best is not None and _remaining() > 10.0:
        try:
            r = minimize(obj, best, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if r.fun < best_val:
                best_val = r.fun
                best = r.x
        except Exception:
            pass

    if best is None:
        best = np.zeros(layers * n)

    return best.reshape(layers, n)
# EVOLVE-BLOCK-END
