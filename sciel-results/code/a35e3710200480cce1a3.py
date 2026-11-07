import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _precompute_super_terms(a, adag, n):
    """Precompute all constant terms of the Liouvillian superoperator."""
    I3 = np.eye(3, dtype=complex)
    # H0 terms (constant part of Hamiltonian)
    anharmonicity = -2 * np.pi * 0.25
    H0 = anharmonicity * 0.5 * (n @ (n - I3))
    H0_super = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
    
    # Drive terms: I*(a+adag) and iQ*(a-adag)
    drive_I = 0.5 * (a + adag)
    drive_Q = 0.5j * (a - adag)
    drive_I_super = -1j * (np.kron(drive_I, I3) - np.kron(I3, drive_I.T))
    drive_Q_super = -1j * (np.kron(drive_Q, I3) - np.kron(I3, drive_Q.T))
    
    # Detuning term
    det_super = -1j * (np.kron(n, I3) - np.kron(I3, n.T))
    
    # Dissipation terms (constant)
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    diss_super = L1_super + L2_super
    
    return H0_super, drive_I_super, drive_Q_super, det_super, diss_super

def _simulate_pulse_fast(IQ, detuning, scale, super_terms):
    """Fast simulation using precomputed superoperator terms."""
    H0_super, drive_I_super, drive_Q_super, det_super, diss_super = super_terms
    dt = 0.5
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [1/np.sqrt(2), 1j/np.sqrt(2), 0],
        [1/np.sqrt(2), -1j/np.sqrt(2), 0]
    ], dtype=complex)
    
    # Precompute all 80 step superoperators
    supers = []
    for i in range(80):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        L = H0_super + detuning * det_super + scale * (I_val * drive_I_super + Q_val * drive_Q_super) + diss_super
        supers.append(expm(L * dt))
    
    # Simulate all 6 states
    fids = []
    for k in range(6):
        psi = states[k]
        rho = np.outer(psi, psi.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho = rho.reshape(3, 3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fids.append(np.real(np.trace(rho_t @ rho)))
    return np.mean(fids)

def _score_fast(IQ, super_terms):
    infs = []
    for det in [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse_fast(IQ, det, sc, super_terms)
            infs.append(1 - fid)
    return np.mean(infs) + np.max(infs)

def _valid(IQ):
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _make_pulse_from_params(params, a, adag, n):
    """
    Generate pulse from low-dimensional parameters:
    params[0] = sigma (Gaussian width)
    params[1] = peak amplitude
    params[2] = DRAG alpha (Q = alpha * dI/dt)
    params[3] = tilt (quadratic correction to envelope)
    """
    sigma = np.clip(params[0], 2.0, 15.0)
    peak = np.clip(params[1], 0.05, 0.39)
    alpha = np.clip(params[2], -0.5, 0.5)
    tilt = np.clip(params[3], -0.02, 0.02)
    
    t = np.arange(80) * 0.5
    center = 20.0
    env = peak * np.exp(-(t - center) ** 2 / (2 * sigma ** 2))
    
    # Add quadratic tilt for flexibility
    env += tilt * (t - center) ** 2 / (sigma ** 2)
    
    # Ensure non-negative and zero at edges
    env = np.maximum(env, 0.0)
    env[0] = 0.0
    env[-1] = 0.0
    
    # Normalize area to pi
    area = np.sum(env) * 0.5
    if area > 1e-10:
        env *= np.pi / area
    
    # Cap amplitude
    env = np.minimum(env, 0.39)
    
    I = env
    Q = alpha * np.gradient(I, 0.5)
    Q[0] = 0.0
    Q[-1] = 0.0
    
    IQ = np.column_stack([I, Q])
    
    # Enforce constraints
    IQ = _enforce_constraints(IQ)
    return IQ

def _enforce_constraints(IQ):
    # Amplitude constraint
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4):
        scale = 0.4 / np.max(amps) * 0.999
        IQ = IQ * scale
    
    # Slew rate constraint
    max_d = 0.12 * 0.5 * 0.999
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        diff = np.linalg.norm(IQ[i] - prev)
        if diff > max_d:
            if diff > 1e-15:
                IQ[i] = prev + (max_d / diff) * (IQ[i] - prev)
            else:
                IQ[i] = prev.copy()
    
    # Final step to zero
    if np.linalg.norm(IQ[-1]) > max_d:
        if np.linalg.norm(IQ[-1]) > 1e-15:
            IQ[-1] *= max_d / np.linalg.norm(IQ[-1])
        else:
            IQ[-1] = np.array([0.0, 0.0])
    
    return IQ

def _make_seed_pulse():
    # DRAG-style Gaussian: sigma=40/6, area=pi
    t = np.arange(80) * 0.5
    sigma = 40.0 / 6.0
    env = np.exp(-(t - 20) ** 2 / (2 * sigma ** 2))
    env = np.maximum(env - env[0] - env[-1], 0.0)
    area = np.sum(env) * 0.5
    if area > 1e-10:
        env *= np.pi / area
    I = env
    Q = -0.3 * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    return _enforce_constraints(IQ)

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s kill
    
    a, adag, n = _build_matrices()
    super_terms = _precompute_super_terms(a, adag, n)
    
    # Start with seed pulse
    best_IQ = _make_seed_pulse()
    best_score = _score_fast(best_IQ, super_terms)
    
    # Low-dimensional parameter optimization
    # params: [sigma, peak, alpha, tilt]
    initial_params = np.array([40.0/6.0, 0.35, -0.3, 0.0])
    
    def objective(params):
        if time.time() > deadline - 10:
            return 1e6
        IQ = _make_pulse_from_params(params, a, adag, n)
        if not _valid(IQ):
            return 1e6
        return _score_fast(IQ, super_terms)
    
    # Nelder-Mead on 4 parameters
    try:
        res = minimize(objective, initial_params, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_IQ = _make_pulse_from_params(res.x, a, adag, n)
            if not _valid(best_IQ):
                best_IQ = _enforce_constraints(best_IQ)
    except Exception:
        pass
    
    # Second round: fine-tune with different initializations
    if time.time() < deadline - 60:
        rng = np.random.default_rng(123)
        for attempt in range(4):
            if time.time() > deadline - 30:
                break
            p_try = initial_params + rng.normal(0, 0.5, 4)
            p_try[0] = np.clip(p_try[0], 3.0, 12.0)
            p_try[1] = np.clip(p_try[1], 0.1, 0.39)
            p_try[2] = np.clip(p_try[2], -0.5, 0.5)
            p_try[3] = np.clip(p_try[3], -0.02, 0.02)
            
            try:
                res2 = minimize(objective, p_try, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_IQ = _make_pulse_from_params(res2.x, a, adag, n)
                    if not _valid(best_IQ):
                        best_IQ = _enforce_constraints(best_IQ)
            except Exception:
                pass
    
    # Final: local free-form refinement on the best pulse
    if time.time() < deadline - 40:
        x0 = best_IQ.flatten()
        
        def obj_freeform(x):
            if time.time() > deadline - 10:
                return 1e6
            IQ = x.reshape(80, 2)
            IQ = _enforce_constraints(IQ)
            if not _valid(IQ):
                return 1e6
            return _score_fast(IQ, super_terms)
        
        try:
            res3 = minimize(obj_freeform, x0, method='Nelder-Mead',
                          options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
            if res3.fun < best_score:
                best_score = res3.fun
                best_IQ = _enforce_constraints(res3.x.reshape(80, 2))
        except Exception:
            pass
    
    # Final validity check
    if not _valid(best_IQ):
        best_IQ = _enforce_constraints(best_IQ)
    
    return best_IQ
