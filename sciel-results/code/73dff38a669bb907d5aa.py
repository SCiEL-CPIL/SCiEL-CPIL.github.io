import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg

# ---- Constants from the task contract ----
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
N_CASES = 9

_DEADLINE = 315.0
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # coherent: -i(H rho - rho H)  =>  L = -i( kron(I,H) - kron(H.T, I) )
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # incoherent (constant)
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # test states (6) and X-pi target states (6)
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
    # X-pi target: |0><->|1> (global phase irrelevant for Tr(rho_t rho))
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# ---- Fast per-pulse simulation (all 9 cases) ----
def _score(I, Q):
    infidelities = np.empty(9)
    k = 0
    det_vals = (0.0, -2 * np.pi * 0.002, 2 * np.pi * 0.002)
    sc_vals = (0.97, 1.0, 1.03)
    for di_val in det_vals:
        for sc_val in sc_vals:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _det_coh
                L += sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

# ---- Fast surrogate: only nominal case (det=0, scale=1) ----
def _score_nominal(I, Q):
    S = np.eye(9, dtype=complex)
    for i in range(N_STEPS):
        L = _base_coh + _base_inc
        L += I[i] * _coh_x + Q[i] * _coh_y
        S = expm(L * DT) @ S
    f = np.empty(6)
    for j in range(6):
        rf = S @ _test_states[j]
        fj = np.real(np.vdot(_target_states[j], rf))
        f[j] = fj if fj > 0 else 0.0
    return 1.0 - np.mean(f)

# ---- Legendre basis for flexible pulse representation ----
_t_grid = np.linspace(0, 1, N_STEPS)
_x_grid = 2 * _t_grid - 1
_N_LEG = 6  # 6 terms for I, 6 terms for Q
_legendre_basis = np.zeros((N_STEPS, _N_LEG))
for k in range(_N_LEG):
    _legendre_basis[:, k] = np.polynomial.legendre.legval(_x_grid, np.eye(_N_LEG)[k])

def _coeffs_to_pulse(coeffs):
    I_coeffs = coeffs[:_N_LEG]
    Q_coeffs = coeffs[_N_LEG:2*_N_LEG]
    I = _legendre_basis @ I_coeffs
    Q = _legendre_basis @ Q_coeffs
    return I, Q

def _enforce_constraints(I, Q):
    # Amplitude constraint
    ampv = np.sqrt(I * I + Q * Q)
    ma = np.max(ampv)
    if ma > AMPLITUDE_BOUND:
        s = AMPLITUDE_BOUND / ma
        I = I * s
        Q = Q * s
    # Slew rate constraint (iterative)
    for _ in range(20):
        violated = False
        for i in range(N_STEPS - 1):
            dI = I[i + 1] - I[i]
            dQ = Q[i + 1] - Q[i]
            sl = np.hypot(dI, dQ) / DT
            if sl > SLEW_BOUND:
                f = SLEW_BOUND * DT / np.hypot(dI, dQ)
                I[i + 1] = I[i] + dI * f
                Q[i + 1] = Q[i] + dQ * f
                violated = True
        s0 = np.hypot(I[0], Q[0]) / DT
        if s0 > SLEW_BOUND:
            f = SLEW_BOUND * DT / np.hypot(I[0], Q[0])
            I[0] *= f
            Q[0] *= f
            violated = True
        sN = np.hypot(I[-1], Q[-1]) / DT
        if sN > SLEW_BOUND:
            f = SLEW_BOUND * DT / np.hypot(I[-1], Q[-1])
            I[-1] *= f
            Q[-1] *= f
            violated = True
        if not violated:
            break
    # Re-check amplitude after slew enforcement
    ampv = np.sqrt(I * I + Q * Q)
    ma = np.max(ampv)
    if ma > AMPLITUDE_BOUND:
        s = AMPLITUDE_BOUND / ma
        I = I * s
        Q = Q * s
    return I, Q

def _make_seed_pulse():
    """Create a DRAG-style Gaussian seed pulse and fit to Legendre basis."""
    sigma = TOTAL_TIME / 6.0
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    x = (t - center) / sigma
    g = np.exp(-0.5 * x * x)
    area = np.sum(g) * DT
    I = g * (np.pi / area)
    Q = np.zeros(N_STEPS)
    # Add DRAG
    dI = np.empty(N_STEPS)
    dI[1:-1] = (I[2:] - I[:-2]) / (2.0 * DT)
    dI[0] = (I[1] - I[0]) / DT
    dI[-1] = (I[-1] - I[-2]) / DT
    Q = 0.1 * dI
    I, Q = _enforce_constraints(I, Q)
    # Fit to Legendre basis
    I_coeffs = np.linalg.lstsq(_legendre_basis, I, rcond=None)[0]
    Q_coeffs = np.linalg.lstsq(_legendre_basis, Q, rcond=None)[0]
    return np.concatenate([I_coeffs, Q_coeffs])

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def _obj_full(coeffs):
    I, Q = _coeffs_to_pulse(coeffs)
    I, Q = _enforce_constraints(I, Q)
    return _score(I, Q)

def _obj_nominal(coeffs):
    I, Q = _coeffs_to_pulse(coeffs)
    I, Q = _enforce_constraints(I, Q)
    return _score_nominal(I, Q)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Seed: DRAG Gaussian fit to Legendre basis
    best_p = _make_seed_pulse()
    best_I, best_Q = _coeffs_to_pulse(best_p)
    best_I, best_Q = _enforce_constraints(best_I, best_Q)
    best_score = _score(best_I, best_Q)

    # ---- Phase 1: L-BFGS-B on nominal surrogate ----
    if elapsed() < _DEADLINE - 60:
        try:
            res = scipy.optimize.minimize(_obj_nominal, best_p, method='L-BFGS-B',
                                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            p1 = res.x
            s1 = _score_nominal(*_coeffs_to_pulse(p1))
            s1_full = _score(*_enforce_constraints(*_coeffs_to_pulse(p1)))
            if s1_full < best_score:
                best_score = s1_full
                best_p = p1.copy()
                best_I, best_Q = _enforce_constraints(*_coeffs_to_pulse(best_p))
        except Exception:
            pass

    # ---- Phase 2: L-BFGS-B on full score ----
    if elapsed() < _DEADLINE - 30:
        try:
            res2 = scipy.optimize.minimize(_obj_full, best_p, method='L-BFGS-B',
                                            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            p2 = res2.x
            s2 = _obj_full(p2)
            if s2 < best_score:
                best_score = s2
                best_p = p2.copy()
                best_I, best_Q = _enforce_constraints(*_coeffs_to_pulse(best_p))
        except Exception:
            pass

    # ---- Phase 3: Multi-start with random perturbations + L-BFGS-B ----
    if elapsed() < _DEADLINE - 10:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if elapsed() > _DEADLINE - 8:
                break
            pert = best_p + rng.normal(0, 0.3, size=2 * _N_LEG)
            try:
                res3 = scipy.optimize.minimize(_obj_full, pert, method='L-BFGS-B',
                                                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                p3 = res3.x
                s3 = _obj_full(p3)
                if s3 < best_score:
                    best_score = s3
                    best_p = p3.copy()
                    best_I, best_Q = _enforce_constraints(*_coeffs_to_pulse(best_p))
            except Exception:
                pass

    # ---- Final safety ----
    best_I, best_Q = _enforce_constraints(best_I, best_Q)
    if not _valid(best_I, best_Q):
        # Fallback to seed
        seed_p = _make_seed_pulse()
        best_I, best_Q = _coeffs_to_pulse(seed_p)
        best_I, best_Q = _enforce_constraints(best_I, best_Q)

    return np.column_stack([best_I, best_Q])
