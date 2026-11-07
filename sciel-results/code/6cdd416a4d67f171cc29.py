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

# Precompute base Liouvillians for the 9 cases
_case_base_L = []
for det_val in DE_TUNINGS:
    for sc_val in SCALES:
        # Note: Scale affects the drive term, not the base (detuning/anharmonicity)
        # Wait, the Hamiltonian is H = detuning n + ... + 0.5 scale (drive)
        # So base_L includes detuning and anharmonicity and dissipation.
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
    I = I.copy()
    Q = Q.copy()
    
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I *= scale
        Q *= scale
    
    # Slew rate constraint - iterative
    for _ in range(20):
        dist0 = np.hypot(I[0], Q[0])
        if dist0 / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / dist0
            I[0] *= scale
            Q[0] *= scale
        
        for i in range(N_STEPS - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = np.hypot(dI, dQ)
            if dist / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / dist
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
        
        distN = np.hypot(I[-1], Q[-1])
        if distN / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / distN
            I[-1] *= scale
            Q[-1] *= scale
            
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

    # Time grid
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0

    # Strategy 1: Parametric DRAG Optimization
    # I(t) = A * exp(-(t-c)^2 / 2 sigma^2)
    # Q(t) = A * beta * (t-c)/sigma^2 * exp(-(t-c)^2 / 2 sigma^2)
    # Area constraint: A * sigma * sqrt(2*pi) = pi  => A = sqrt(pi/2) / sigma
    
    def _drag_pulse(sigma, beta):
        if sigma < 1.0: sigma = 1.0
        if sigma > 20.0: sigma = 20.0
        A = np.sqrt(np.pi / 2.0) / sigma
        I = A * np.exp(-0.5 * ((t - center) / sigma)**2)
        Q = A * beta * ((t - center) / sigma**2) * np.exp(-0.5 * ((t - center) / sigma)**2)
        return I, Q

    def _objective_drag(params):
        sigma, beta = params
        I, Q = _drag_pulse(sigma, beta)
        if not _valid(I, Q):
            return 1e6
        return _score_fast(I, Q)

    best_I, best_Q = None, None
    best_score = 1e9

    # Multi-start for DRAG optimization
    starts = [
        [6.0, 0.0],   # Standard
        [5.0, 0.0],   # Narrower
        [8.0, 0.0],   # Broader
        [6.0, 0.5],   # Positive DRAG
        [6.0, -0.5],  # Negative DRAG
        [6.0, 1.0],   # Stronger DRAG
        [6.0, -1.0],  # Stronger DRAG neg
        [7.0, 0.2],
        [5.5, -0.2],
    ]

    for start in starts:
        if elapsed() > _DEADLINE - 30:
            break
        try:
            res = scipy.optimize.minimize(_objective_drag, start, method='Nelder-Mead',
                                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                I_new, Q_new = _drag_pulse(res.x[0], res.x[1])
                if _valid(I_new, Q_new):
                    best_I, best_Q = I_new, Q_new
        except Exception:
            pass

    # If no valid pulse found yet, use a seed
    if best_I is None:
        I_seed, Q_seed = _drag_pulse(6.0, 0.0)
        I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
        best_I, best_Q = I_seed, Q_seed
        best_score = _score_fast(best_I, best_Q)

    # Strategy 2: Local refinement of the best pulse
    # Optimize all 160 parameters directly
    if elapsed() < _DEADLINE - 10:
        try:
            # Flatten I and Q
            x0 = np.concatenate([best_I, best_Q])
            
            def _objective_direct(params):
                I = params[:N_STEPS]
                Q = params[N_STEPS:]
                I, Q = _enforce_constraints(I, Q)
                return _score_fast(I, Q)

            # Use L-BFGS-B with bounds if possible, or Nelder-Mead
            # Bounds: I, Q in [-0.4, 0.4]
            bounds = [(-0.4, 0.4)] * (2 * N_STEPS)
            
            res = scipy.optimize.minimize(_objective_direct, x0, method='L-BFGS-B',
                                          bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                I_new = res.x[:N_STEPS]
                Q_new = res.x[N_STEPS:]
                I_new, Q_new = _enforce_constraints(I_new, Q_new)
                if _valid(I_new, Q_new):
                    best_I, best_Q = I_new, Q_new
        except Exception:
            pass

    # Final validation
    if not _valid(best_I, best_Q):
        I_seed, Q_seed = _drag_pulse(6.0, 0.0)
        I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
        best_I, best_Q = I_seed, Q_seed

    return np.column_stack([best_I, best_Q])
