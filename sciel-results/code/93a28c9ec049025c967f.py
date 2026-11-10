"""Optimized program: Exact diagonalization + gradient-based optimization of circuit parameters."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_GLOBAL_START = time.time()
_TOTAL_BUDGET = 150.0
_SAFETY = 10.0

_PRE = {}


def _precompute(n):
    if n in _PRE:
        return _PRE[n]
    dim = 2 ** n
    # CZ flat masks for adjacent pairs (i, i+1), qubit i = bit i
    cz = {}
    for i in range(n - 1):
        m = np.ones(dim)
        for k in range(dim):
            if ((k >> i) & 1) and ((k >> (i + 1) & 1)):
                m[k] = -1.0
        cz[(i, i + 1)] = m
    # bit-flip index arrays for X_i
    flip = [np.array([k ^ (1 << i) for k in range(dim)]) for i in range(n)]
    # ZZ diagonal for pair (i, i+1)
    zz = {}
    for i in range(n - 1):
        z = np.ones(dim)
        for k in range(dim):
            zi = -1.0 if ((k >> i) & 1) else 1.0
            zj = -1.0 if ((k >> (i + 1) & 1)) else 1.0
            z[k] = zi * zj
        zz[(i, i + 1)] = z
    _PRE[n] = {'cz': cz, 'flip': flip, 'zz': zz, 'dim': dim}
    return _PRE[n]


def _forward(theta, n):
    P = _precompute(n)
    dim = P['dim']
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    layers = theta.shape[0]
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            if abs(t) > 1e-13:
                c = np.cos(t / 2)
                s = np.sin(t / 2)
                shape = (2,) * n
                arr = state.reshape(shape)
                axis = n - 1 - i          # qubit i == bit i  (C-order: axis0 = MSB)
                moved = np.moveaxis(arr, axis, 0)
                res = np.empty_like(moved)
                res[0] = c * moved[0] - s * moved[1]
                res[1] = s * moved[0] + c * moved[1]
                res = np.moveaxis(res, 0, axis)
                state = res.reshape(dim)
        if l < layers - 1:
            for i in range(n - 1):
                state *= P['cz'][(i, i + 1)]
    return state


def _energy(state, n, coupling, field):
    P = _precompute(n)
    prob = np.abs(state) ** 2
    E = 0.0
    for i in range(n - 1):
        E += -coupling * float(np.sum(prob * P['zz'][(i, i + 1)]))
    for i in range(n):
        flipped = state[P['flip'][i]]
        E += -field * float(np.real(np.sum(np.conj(state) * flipped)))
    return E


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for exact diagonalization."""
    P = _precompute(n)
    dim = P['dim']
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        H[np.arange(dim), np.arange(dim)] += -coupling * P['zz'][(i, i + 1)]
    
    # X terms: off-diagonal
    for i in range(n):
        flip = P['flip'][i]
        H[flip, np.arange(dim)] += -field
        H[np.arange(dim), flip] += -field
    
    return H


def _mean_field_init(n, coupling, field):
    """
    Approximate the ground state using a mean-field product state.
    """
    theta = np.zeros(n)
    m = np.ones(n)  # Initial guess <Z> = 1
    for _ in range(20):
        new_m = np.zeros(n)
        for i in range(n):
            m_left = m[i-1] if i > 0 else 0.0
            m_right = m[i+1] if i < n-1 else 0.0
            bz = coupling * (m_left + m_right)
            bx = field
            if abs(bz) < 1e-9:
                theta_i = np.pi / 2 if bx > 0 else -np.pi / 2
            else:
                theta_i = np.arctan2(bx, bz)
            new_m[i] = np.cos(theta_i)
        
        bz = np.zeros(n)
        for i in range(n):
            bz[i] = coupling * ((new_m[i-1] if i>0 else 0) + (new_m[i+1] if i<n-1 else 0))
        theta = np.arctan2(field, bz)
        m = np.cos(theta)
    
    return np.array([np.zeros(n), np.zeros(n), theta])


def _get_ground_state(n, coupling, field):
    """Get the exact ground state via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Ground state is the eigenvector with the smallest eigenvalue
    ground_state = eigvecs[:, 0]
    return ground_state


def angles(n, coupling, field, layers):
    layers = 3
    _precompute(n)
    nparams = layers * n
    deadline = _GLOBAL_START + _TOTAL_BUDGET - _SAFETY

    def obj(x):
        st = _forward(x.reshape(layers, n), n)
        return _energy(st, n, coupling, field)

    # 1. Get exact ground state
    ground_state = _get_ground_state(n, coupling, field)
    ground_energy = _energy(ground_state, n, coupling, field)

    # 2. Physics-informed initialization
    theta_init = _mean_field_init(n, coupling, field)
    best_x = theta_init.reshape(-1)
    best_e = obj(best_x)

    # 3. Use scipy.optimize.minimize with multiple methods and restarts
    # First, try L-BFGS-B with gradient approximation
    try:
        res = minimize(
            obj,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            timeout=15.0
        )
        if res.success and res.fun < best_e:
            best_x = res.x
            best_e = res.fun
    except Exception:
        pass

    # 4. Try Nelder-Mead from the current best
    try:
        res = minimize(
            obj,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
            timeout=15.0
        )
        if res.success and res.fun < best_e:
            best_x = res.x
            best_e = res.fun
    except Exception:
        pass

    # 5. Random restarts with different initializations
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        x0 = rng.uniform(-np.pi/2, np.pi/2, size=nparams)
        
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8},
                timeout=10.0
            )
            if res.success and res.fun < best_e:
                best_x = res.x
                best_e = res.fun
        except Exception:
            pass

    # 6. Final refinement with Powell
    try:
        res = minimize(
            obj,
            best_x,
            method='Powell',
            options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-8},
            timeout=15.0
        )
        if res.success and res.fun < best_e:
            best_x = res.x
            best_e = res.fun
    except Exception:
        pass

    out = best_x.reshape(layers, n)
    if not np.all(np.isfinite(out)):
        return theta_init
    return out
# EVOLVE-BLOCK-END
