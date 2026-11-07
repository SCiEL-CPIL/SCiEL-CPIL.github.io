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

_DEADLINE = 320.0
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

# ---- Precompute Liouvillian for each case (without drive) ----
# L_case = base_coh + base_inc + det_coh*delta + (scale*I*coh_x + scale*Q*coh_y)
# For scoring we need to apply the full Liouvillian each step.
# We can precompute the constant part per case: L_const_case = base_coh + base_inc + delta*det_coh
# Then each step: L = L_const_case + scale*(I[i]*coh_x + Q[i]*coh_y)

_L_const_cases = []
for delta in DE_TUNINGS:
    L_const_cases.append(_base_coh + _base_inc + delta * _det_coh)
_L_const_cases = np.array(_L_const_cases)  # shape (9, 9, 9)

def _score_fast(I, Q):
    """Score using precomputed constant Liouvillians."""
    infidelities = np.empty(9)
    k = 0
    for ci in range(9):
        sc = SCALES[ci % 3]
        L_const = _L_const_cases[ci]
        S = np.eye(9, dtype=complex)
        for i in range(N_STEPS):
            L = L_const + sc * (I[i] * _coh_x + Q[i] * _coh_y)
            S = expm(L * DT) @ S
        f = np.empty(6)
        for j in range(6):
            rf = S @ _test_states[j]
            fj = np.real(np.vdot(_target_states[j], rf))
            f[j] = fj if fj > 0 else 0.0
        infidelities[k] = 1.0 - np.mean(f)
        k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _score_nominal_fast(I, Q):
    L_const = _L_const_cases[4]  # delta=0, scale=1.0 -> index 4
    S = np.eye(9, dtype=complex)
    for i in range(N_STEPS):
        L = L_const + I[i] * _coh_x + Q[i] * _coh_y
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
    Iterative clipping/projection.
    """
    I = I.copy()
    Q = Q.copy()
    for _ in range(20):
        # Amplitude constraint
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / np.max(amp)
            I *= scale
            Q *= scale
        
        # Slew rate constraint
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
    
    # Final amplitude check
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I *= scale
        Q *= scale
        
    return I, Q

# ---- Parameterization: use a smooth basis to ensure slew rate is naturally satisfied ----
# We parameterize the pulse as a sum of smooth basis functions that are zero at boundaries
# and have bounded derivatives. This avoids the projection step during optimization.

# Use a set of basis functions that are smooth and vanish at t=0 and t=TOTAL_TIME
# with bounded derivatives. We can use scaled Legendre polynomials on [0,1] multiplied
# by a smooth window that vanishes at the edges.

# Actually, let's use a simpler approach: parameterize with a small number of parameters
# that directly define a smooth pulse shape. We'll use a Gaussian-like shape with
# adjustable center, width, and a DRAG-like skew, plus a few harmonics.

# Better: use a Fourier-like basis with sine/cosine terms that vanish at boundaries.
# Let t_norm = t / TOTAL_TIME in [0, 1]. Use basis: sin(k * pi * t_norm) for k=1..K
# These are zero at t=0 and t=TOTAL_TIME, and their derivatives are bounded.

def _make_basis(K=6):
    """Create basis functions: sin(k*pi*t) for k=1..K, t in [0,1]."""
    t = (np.arange(N_STEPS) + 0.5) * DT / TOTAL_TIME  # midpoints in [0,1]
    basis = np.zeros((N_STEPS, 2*K))
    for k in range(1, K+1):
        basis[:, 2*(k-1)] = np.sin(k * np.pi * t)
        basis[:, 2*(k-1)+1] = np.cos(k * np.pi * t) - 1.0  # shift so it's zero at edges? 
        # Actually cos(k*pi*t) - 1 is zero at t=0 but not at t=1.
        # Let's just use sin(k*pi*t) for both I and Q.
    # Redo: use sin(k*pi*t) for I and sin(k*pi*t) for Q separately
    basis = np.zeros((N_STEPS, 2*K))
    for k in range(1, K+1):
        basis[:, 2*(k-1)] = np.sin(k * np.pi * t)
        basis[:, 2*(k-1)+1] = np.sin(k * np.pi * t)
    return basis

# Actually, let me reconsider. The slew rate constraint is ||p[i+1]-p[i]||/dt <= 0.12.
# For a smooth function, the maximum slope is bounded. If we use sin(k*pi*t_norm) with t_norm in [0,1],
# the derivative is k*pi*cos(k*pi*t_norm)/TOTAL_TIME * DT... wait.
# t = (i+0.5)*DT, so t_norm = (i+0.5)*DT/TOTAL_TIME.
# d/dt of sin(k*pi*t_norm) = k*pi*cos(k*pi*t_norm) * DT/TOTAL_TIME = k*pi*cos(k*pi*t_norm)/80.
# For k=6, max derivative = 6*pi/80 ≈ 0.236 per ns. Times DT=0.5 gives 0.118 per step.
# That's close to the slew bound of 0.12! So k=6 might be tight.
# Let's use K=4 to be safe: max derivative = 4*pi/80 = 0.157 per ns, *0.5 = 0.0785 per step. Safe.

K_BASIS = 4
_basis_funcs = np.zeros((N_STEPS, 2*K_BASIS))
_t_norm = (np.arange(N_STEPS) + 0.5) * DT / TOTAL_TIME
for k in range(1, K_BASIS+1):
    _basis_funcs[:, 2*(k-1)] = np.sin(k * np.pi * _t_norm)
    _basis_funcs[:, 2*(k-1)+1] = np.sin(k * np.pi * _t_norm)

def _coeffs_to_pulse(params):
    """params: (8,) array. First 4 for I, next 4 for Q."""
    I = _basis_funcs[:, :4] @ params[:4]
    Q = _basis_funcs[:, 4:] @ params[4:]
    return I, Q

def _score_params(params):
    I, Q = _coeffs_to_pulse(params)
    I, Q = _enforce_constraints(I, Q)
    return _score_fast(I, Q)

def _score_nominal_params(params):
    I, Q = _coeffs_to_pulse(params)
    I, Q = _enforce_constraints(I, Q)
    return _score_nominal_fast(I, Q)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Seed: simple Gaussian-like pulse
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    sigma = TOTAL_TIME / 6.0
    I_seed = np.exp(-0.5 * ((t - center) / sigma)**2)
    area = np.sum(I_seed) * DT
    I_seed *= (np.pi / area)
    Q_seed = np.zeros(N_STEPS)
    
    I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
    
    # Fit to our basis
    # Least squares: I ≈ B_I @ c_I, Q ≈ B_Q @ c_Q
    B_I = _basis_funcs[:, :4]
    B_Q = _basis_funcs[:, 4:]
    c_I, _, _, _ = np.linalg.lstsq(B_I, I_seed, rcond=None)
    c_Q, _, _, _ = np.linalg.lstsq(B_Q, Q_seed, rcond=None)
    params_seed = np.concatenate([c_I, c_Q])
    
    best_score = _score_fast(I_seed, Q_seed)
    best_params = params_seed.copy()
    best_I, best_Q = I_seed.copy(), Q_seed.copy()

    # Optimization using L-BFGS-B
    bounds = [(-1.5, 1.5)] * (2*K_BASIS)
    
    if elapsed() < _DEADLINE - 40:
        try:
            res = scipy.optimize.minimize(_score_params, params_seed, method='L-BFGS-B',
                                          bounds=bounds,
                                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                best_I, best_Q = _coeffs_to_pulse(best_params)
                best_I, best_Q = _enforce_constraints(best_I, best_Q)
        except Exception:
            pass

    # Second pass: polish on nominal case
    if elapsed() < _DEADLINE - 20:
        try:
            res2 = scipy.optimize.minimize(_score_nominal_params, best_params, method='L-BFGS-B',
                                           bounds=bounds,
                                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            I2, Q2 = _coeffs_to_pulse(res2.x)
            I2, Q2 = _enforce_constraints(I2, Q2)
            s2 = _score_fast(I2, Q2)
            if s2 < best_score:
                best_score = s2
                best_params = res2.x.copy()
                best_I, best_Q = I2, Q2
        except Exception:
            pass

    # Third pass: final polish on full score if time permits
    if elapsed() < _DEADLINE - 10:
        try:
            res3 = scipy.optimize.minimize(_score_params, best_params, method='L-BFGS-B',
                                           bounds=bounds,
                                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            I3, Q3 = _coeffs_to_pulse(res3.x)
            I3, Q3 = _enforce_constraints(I3, Q3)
            s3 = _score_fast(I3, Q3)
            if s3 < best_score:
                best_score = s3
                best_params = res3.x.copy()
                best_I, best_Q = I3, Q3
        except Exception:
            pass

    # Final validation
    if not _valid(best_I, best_Q):
        best_I, best_Q = I_seed, Q_seed
        best_score = _score_fast(best_I, best_Q)

    return np.column_stack([best_I, best_Q])
