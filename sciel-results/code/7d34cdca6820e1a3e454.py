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

_DEADLINE = 290.0  # Safety margin before 330s kill
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # Coherent part: -i(H rho - rho H)
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    # H = detuning * n + 0.5 * anh * n(n-1) + 0.5 * scale * (I (a + a^dag) + i Q (a - a^dag))
    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # Incoherent part
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # Test states and targets
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
    
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
        
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

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

def _score(I, Q):
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
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

def _constraint_penalty(I, Q):
    amp = np.sqrt(I**2 + Q**2)
    amp_viol = np.sum(np.maximum(0, amp - AMPLITUDE_BOUND)**2)
    
    slew = np.zeros(N_STEPS + 1)
    slew[0] = np.hypot(I[0], Q[0]) / DT
    slew[-1] = np.hypot(I[-1], Q[-1]) / DT
    if N_STEPS > 1:
        dI = np.diff(I)
        dQ = np.diff(Q)
        slew[1:-1] = np.hypot(dI, dQ) / DT
    slew_viol = np.sum(np.maximum(0, slew - SLEW_BOUND)**2)
    
    return 100.0 * (amp_viol + slew_viol)

# Precompute Legendre polynomials on the time grid
def _init_legendre():
    t = (np.arange(N_STEPS) + 0.5) * DT
    # Map to [-1, 1]
    t_norm = (t - 20.0) / 20.0  # center at 20ns, half-width 20ns
    # Clip to avoid numerical issues
    t_norm = np.clip(t_norm, -1.0, 1.0)
    # Compute Legendre polynomials P_0 through P_5
    P = np.zeros((6, N_STEPS))
    P[0, :] = 1.0
    if N_STEPS > 1:
        P[1, :] = t_norm
    for k in range(2, 6):
        P[k, :] = ((2*k - 1) * t_norm * P[k-1, :] - (k-1) * P[k-2, :]) / k
    return t, t_norm, P

_t_grid, _t_norm, _legendre_P = _init_legendre()

def _coeffs_to_pulse(coeffs):
    """Convert 12 Legendre coefficients (6 for I, 6 for Q) to pulse arrays."""
    cI = coeffs[:6]
    cQ = coeffs[6:12]
    I = _legendre_P @ cI
    Q = _legendre_P @ cQ
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def param_obj(params):
        I, Q = _coeffs_to_pulse(params)
        score = _score(I, Q)
        pen = _constraint_penalty(I, Q)
        return score + pen

    # Initial guess: Gaussian-like shape approximated by Legendre coefficients
    # A Gaussian centered at 20ns with sigma ~ 6.67ns
    # Approximate with first few Legendre terms
    # For a simple X pulse: I should be positive hump, Q should be small (DRAG)
    # Start with coefficients that approximate a Gaussian
    sigma_seed = 40.0 / 6.0
    A_seed = 1.0 / (sigma_seed * np.sqrt(2.0))
    
    # Generate seed pulse and project onto Legendre basis
    I_seed = A_seed * np.exp(-0.5 * ((_t_grid - 20.0) / sigma_seed)**2)
    Q_seed = np.zeros(N_STEPS)
    
    # Least squares projection onto Legendre basis
    cI_seed = np.linalg.lstsq(_legendre_P, I_seed, rcond=None)[0]
    cQ_seed = np.zeros(6)
    
    x0 = np.concatenate([cI_seed, cQ_seed])
    
    # Bounds: keep coefficients moderate
    bounds = [(-0.5, 0.5)] * 12
    
    best_I = np.zeros(N_STEPS)
    best_Q = np.zeros(N_STEPS)
    best_score = np.inf
    
    # L-BFGS-B optimization
    if elapsed() < _DEADLINE - 30:
        try:
            res = scipy.optimize.minimize(param_obj, x0, method='L-BFGS-B', bounds=bounds,
                                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            if res.fun < best_score:
                best_I, best_Q = _coeffs_to_pulse(res.x)
                best_score = _score(best_I, best_Q)
        except Exception:
            pass

    # Second round with slightly different initial conditions if time permits
    if elapsed() < _DEADLINE - 15:
        try:
            # Perturb initial guess
            x1 = x0 + np.random.normal(0, 0.05, 12)
            res2 = scipy.optimize.minimize(param_obj, x1, method='L-BFGS-B', bounds=bounds,
                                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-8})
            if res2.fun < best_score:
                best_I, best_Q = _coeffs_to_pulse(res2.x)
                best_score = _score(best_I, best_Q)
        except Exception:
            pass

    # Fallback if invalid
    if not _valid(best_I, best_Q) or best_score == np.inf:
        I_seed = A_seed * np.exp(-0.5 * ((_t_grid - 20.0) / sigma_seed)**2)
        Q_seed = np.zeros(N_STEPS)
        best_I, best_Q = I_seed, Q_seed
        best_score = _score(best_I, best_Q)

    # Final check and clamp
    if not _valid(best_I, best_Q):
        amp = np.sqrt(best_I**2 + best_Q**2)
        mask = amp > AMPLITUDE_BOUND
        if np.any(mask):
            best_I[mask] *= AMPLITUDE_BOUND / amp[mask]
            best_Q[mask] *= AMPLITUDE_BOUND / amp[mask]
        
        # Check slew rate
        if not _valid(best_I, best_Q):
            # Reduce amplitude to satisfy slew
            for i in range(N_STEPS - 1):
                d = np.hypot(best_I[i+1] - best_I[i], best_Q[i+1] - best_Q[i])
                if d / DT > SLEW_BOUND:
                    scale = SLEW_BOUND * DT / d
                    best_I[i+1] = best_I[i] + (best_I[i+1] - best_I[i]) * scale
                    best_Q[i+1] = best_Q[i] + (best_Q[i+1] - best_Q[i]) * scale
            
            if not _valid(best_I, best_Q):
                return np.zeros((80, 2))

    return np.column_stack([best_I, best_Q])
