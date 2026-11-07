import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

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

# ---- Precompute expm for all 9 cases for faster scoring ----
# We'll precompute the base Liouvillian for each case
_case_base_L = []
for det_val in DE_TUNINGS:
    for sc_val in SCALES:
        L_base = _base_coh + _base_inc + det_val * _det_coh
        _case_base_L.append(L_base)

def _score_fast(I, Q):
    """Score using precomputed base Liouvillians."""
    infidelities = np.empty(9)
    k = 0
    for ci in range(9):
        sc_val = SCALES[ci % 3]
        S = np.eye(9, dtype=complex)
        for i in range(N_STEPS):
            L = _case_base_L[ci] + sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
            S = expm(L * DT) @ S
        f = np.empty(6)
        for j in range(6):
            rf = S @ _test_states[j]
            fj = np.real(np.vdot(_target_states[j], rf))
            f[j] = fj if fj > 0 else 0.0
        infidelities[k] = 1.0 - np.mean(f)
        k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _score_nominal(I, Q):
    """Score only the nominal case (det=0, scale=1)."""
    S = np.eye(9, dtype=complex)
    for i in range(N_STEPS):
        L = _case_base_L[4] + I[i] * _coh_x + Q[i] * _coh_y  # case index 4 is det=0, scale=1
        S = expm(L * DT) @ S
    f = np.empty(6)
    for j in range(6):
        rf = S @ _test_states[j]
        fj = np.real(np.vdot(_target_states[j], rf))
        f[j] = fj if fj > 0 else 0.0
    return 1.0 - np.mean(f)

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

def _enforce_constraints(I, Q):
    """
    Project I, Q to satisfy amplitude and slew rate constraints.
    """
    I = I.copy()
    Q = Q.copy()
    
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I *= scale
        Q *= scale
    
    # Slew rate constraint - iterative
    for _ in range(10):
        # From 0 to I[0], Q[0]
        dist0 = np.hypot(I[0], Q[0])
        if dist0 / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / dist0
            I[0] *= scale
            Q[0] *= scale
        
        # Between steps
        for i in range(N_STEPS - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = np.hypot(dI, dQ)
            if dist / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / dist
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
        
        # From last to 0
        distN = np.hypot(I[-1], Q[-1])
        if distN / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / distN
            I[-1] *= scale
            Q[-1] *= scale
            
    # Re-check amplitude after slew adjustments
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I *= scale
        Q *= scale
        
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Seed: DRAG-style Gaussian pulse
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    sigma = TOTAL_TIME / 6.0
    I_seed = np.exp(-0.5 * ((t - center) / sigma)**2)
    area = np.sum(I_seed) * DT
    I_seed *= (np.pi / area)
    Q_seed = np.zeros(N_STEPS)
    
    I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
    
    best_score = _score_fast(I_seed, Q_seed)
    best_I, best_Q = I_seed.copy(), Q_seed.copy()

    # Try several different parameterizations and optimization strategies
    # Strategy 1: Direct optimization of all 80 I and 80 Q values with constraints
    # Strategy 2: Gaussian family with DRAG correction
    # Strategy 3: Legendre basis
    
    # Strategy 2: Gaussian + DRAG
    def _gaussian_drag_pulse(params):
        """
        params: [amp, sigma, drag_coeff, q_amp]
        I(t) = amp * exp(-0.5*((t-center)/sigma)^2) * (1 + drag_coeff * (t-center)/sigma * exp(-0.5*((t-center)/sigma)^2))
        Q(t) = q_amp * exp(-0.5*((t-center)/sigma)^2) * (t-center)/sigma
        """
        amp, sigma, drag_coeff, q_amp = params
        if sigma < 1.0:
            sigma = 1.0
        I = amp * np.exp(-0.5 * ((t - center) / sigma)**2)
        # DRAG correction: derivative of Gaussian
        drag = -drag_coeff * (t - center) / (sigma**2) * np.exp(-0.5 * ((t - center) / sigma)**2)
        I = I + drag
        Q = q_amp * (t - center) / sigma * np.exp(-0.5 * ((t - center) / sigma)**2)
        return I, Q

    # Optimize Gaussian + DRAG
    def _objective_gaussian_drag(params):
        I, Q = _gaussian_drag_pulse(params)
        I, Q = _enforce_constraints(I, Q)
        return _score_nominal(I, Q)

    if elapsed() < _DEADLINE - 20:
        try:
            # Initial guess
            amp0 = np.pi / TOTAL_TIME
            sigma0 = TOTAL_TIME / 6.0
            drag0 = 0.0
            q0 = 0.0
            res = scipy.optimize.minimize(_objective_gaussian_drag, [amp0, sigma0, drag0, q0],
                                          method='Nelder-Mead',
                                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if _valid(*_gaussian_drag_pulse(res.x)):
                I_new, Q_new = _gaussian_drag_pulse(res.x)
                I_new, Q_new = _enforce_constraints(I_new, Q_new)
                s_new = _score_fast(I_new, Q_new)
                if s_new < best_score:
                    best_score = s_new
                    best_I, best_Q = I_new, Q_new
        except Exception:
            pass

    # Strategy 3: Legendre basis
    _t_grid = np.linspace(0, 1, N_STEPS)
    _x_grid = 2 * _t_grid - 1
    _legendre_basis = np.array([np.polynomial.legendre.legval(_x_grid, [1 if i == k else 0 for i in range(10)]) for k in range(10)]).T

    def _coeffs_to_pulse(coeffs):
        I_coeffs = coeffs[:10]
        Q_coeffs = coeffs[10:]
        I = _legendre_basis @ I_coeffs
        Q = _legendre_basis @ Q_coeffs
        return I, Q

    def _pulse_to_coeffs(I, Q):
        I_coeffs = np.linalg.lstsq(_legendre_basis, I, rcond=None)[0]
        Q_coeffs = np.linalg.lstsq(_legendre_basis, Q, rcond=None)[0]
        return np.concatenate([I_coeffs, Q_coeffs])

    def _objective_legendre(coeffs):
        I, Q = _coeffs_to_pulse(coeffs)
        I, Q = _enforce_constraints(I, Q)
        return _score_nominal(I, Q)

    if elapsed() < _DEADLINE - 20:
        try:
            coeffs_seed = _pulse_to_coeffs(best_I, best_Q)
            bounds = [(-3, 3)] * 20
            res = scipy.optimize.minimize(_objective_legendre, coeffs_seed, method='L-BFGS-B',
                                          bounds=bounds,
                                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            I_new, Q_new = _coeffs_to_pulse(res.x)
            I_new, Q_new = _enforce_constraints(I_new, Q_new)
            s_new = _score_fast(I_new, Q_new)
            if s_new < best_score:
                best_score = s_new
                best_I, Q_new = I_new, Q_new
        except Exception:
            pass

    # Strategy 4: Direct optimization with penalty for constraints
    def _objective_direct(params):
        I = params[:N_STEPS]
        Q = params[N_STEPS:]
        I, Q = _enforce_constraints(I, Q)
        return _score_nominal(I, Q)

    if elapsed() < _DEADLINE - 30:
        try:
            res = scipy.optimize.minimize(_objective_direct, np.concatenate([best_I, best_Q]),
                                          method='L-BFGS-B',
                                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            I_new = res.x[:N_STEPS]
            Q_new = res.x[N_STEPS:]
            I_new, Q_new = _enforce_constraints(I_new, Q_new)
            s_new = _score_fast(I_new, Q_new)
            if s_new < best_score:
                best_score = s_new
                best_I, best_Q = I_new, Q_new
        except Exception:
            pass

    # Final validation
    if not _valid(best_I, best_Q):
        best_I, best_Q = I_seed, Q_seed
        best_score = _score_fast(best_I, best_Q)

    return np.column_stack([best_I, best_Q])
