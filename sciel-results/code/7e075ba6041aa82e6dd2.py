"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = time.time() + 300  # safety margin before 330s kill

# Hardware parameters
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
N_STEPS = 80
DT = 0.5

# Precompute ladder operators
_a = np.zeros((3, 3), dtype=complex)
_a[1, 0] = np.sqrt(1.0)
_a[2, 1] = np.sqrt(2.0)
_a_dag = _a.conj().T
_n_op = _a_dag @ _a

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n_op * np.sqrt(2.0 / Tphi)
_L1_dag = _L1.conj().T
_L2_dag = _L2.conj().T

_I3 = np.eye(3)

# Precompute constant parts of Liouvillian
_H_anharm = 0.5 * ANHARMONICITY * (_n_op @ (_n_op - _I3))
_H_anharm_t = _H_anharm.T

_L_diss_base = np.zeros((9, 9), dtype=complex)
for Lk, Lk_dag in [(_L1, _L1_dag), (_L2, _L2_dag)]:
    Lk_dag_Lk = Lk_dag @ Lk
    _L_diss_base += np.kron(Lk_dag.T, Lk) - 0.5 * (np.kron(_I3, Lk_dag_Lk.T) + np.kron(Lk_dag_Lk, _I3))

# Precompute kron structures for drive terms
_kron_I3_a = np.kron(_I3, _a)
_kron_I3_adag = np.kron(_I3, _a_dag)
_kron_a_I3 = np.kron(_a, _I3)
_kron_adag_I3 = np.kron(_a_dag, _I3)


def _build_liouvillian_fast(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one step (fast version)."""
    # H = detuning * n + H_anharm + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    # H_t = detuning * n_t + H_anharm_t + 0.5*scale*(I*(a_t+a_dag_t) + i*Q*(a_t-a_dag_t))
    # Since n is diagonal real, n_t = n
    # a_t has a[0,1]=1, a[1,2]=sqrt(2); a_dag_t = a (since a_dag = a^T for real a)
    # Actually a is real, so a_dag = a.T, a_dag.T = a
    # a.T: [0,0]=0, [0,1]=0, [0,2]=0, [1,0]=1, [1,1]=0, [1,2]=0, [2,0]=0, [2,1]=sqrt(2), [2,2]=0
    # So a + a_dag is symmetric real, (a+a_dag).T = a + a_dag
    # a - a_dag is antisymmetric real, (a-a_dag).T = -(a-a_dag)
    
    # L_ham = -1j*(kron(I3, H_t) - kron(H_t, I3))
    # = -1j*(kron(I3, det*n + H_anharm_t + 0.5*s*I*(a+a_dag) + 0.5*s*i*Q*(a-a_dag)) 
    #          - kron(det*n + H_anharm_t + 0.5*s*I*(a+a_dag) + 0.5*s*i*Q*(a-a_dag), I3))
    
    H_t = detuning * _n_op + _H_anharm_t + 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag))
    
    L_ham = -1j * (np.kron(_I3, H_t) - np.kron(H_t, _I3))
    
    return L_ham + _L_diss_base


def _simulate_pulse_fast(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix (fast)."""
    rho = np.zeros(9, dtype=complex)
    rho[0] = 1.0  # |0><0| in column-major
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _build_liouvillian_fast(detuning, scale, I_val, Q_val)
        L_mat = expm(L * DT)
        rho = L_mat @ rho
    
    return rho


def _fidelity_from_vec(rho_vec, target_state):
    """Compute fidelity from vectorized density matrix and pure target state."""
    rho = rho_vec.reshape(3, 3)
    val = np.vdot(target_state, rho @ target_state).real
    return max(0.0, min(1.0, val))


def _check_validity(pulse):
    """Check amplitude and slew rate constraints."""
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _compute_score_fast(pulse):
    """Compute the full score for a pulse (fast version)."""
    if not _check_validity(pulse):
        return None
    
    target_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scl in scales:
            for init_state in target_states:
                rho_vec = _simulate_pulse_fast(pulse, det, scl)
                target = X_gate @ init_state
                f = _fidelity_from_vec(rho_vec, target)
                infidelities.append(1.0 - f)
    
    infidelities = np.array(infidelities)
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf


def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    pulse = pulse.copy()
    
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    # Slew rate bound (forward pass)
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        delta = pulse[i] - prev
        norm = np.linalg.norm(delta) / DT
        if norm > SLEW_BOUND:
            pulse[i] = prev + SLEW_BOUND * DT * delta / (norm * DT)
        prev = pulse[i]
    
    # Final step to zero
    norm_final = np.linalg.norm(prev) / DT
    if norm_final > SLEW_BOUND:
        pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
    
    return pulse


def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = 40.0 / 6.0
    
    envelope = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    envelope = envelope * (np.pi / raw_area)
    
    drag_coeff = -0.12
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope
    Q_pulse = drag_coeff * envelope_deriv
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _param_pulse(params):
    """Generate pulse from parameters: [amp, drag, sigma, q_offset, i_offset]."""
    amp, drag, sigma, q_offset, i_offset = params
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    
    envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    if raw_area > 1e-10:
        envelope = envelope * (np.pi / raw_area)
    
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope + i_offset
    Q_pulse = drag * envelope_deriv + q_offset
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _direct_optimize(pulse, n_iter=200, step_size=0.02):
    """Direct coordinate descent on pulse samples with constraint projection."""
    best_pulse = pulse.copy()
    best_score = _compute_score_fast(best_pulse)
    if best_score is None:
        return best_pulse, 1.0
    
    for iteration in range(n_iter):
        if time.time() > _DEADLINE - 15:
            break
        
        improved = False
        # Try perturbing random samples
        for _ in range(5):
            idx = np.random.randint(0, N_STEPS)
            for comp in range(2):
                for sign in [1, -1]:
                    trial = best_pulse.copy()
                    trial[idx, comp] += sign * step_size
                    trial = _enforce_constraints(trial)
                    score = _compute_score_fast(trial)
                    if score is not None and score < best_score:
                        best_pulse = trial
                        best_score = score
                        improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    return best_pulse, best_score


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Start with seed
    best_pulse = _seed_pulse()
    best_score = _compute_score_fast(best_pulse)
    if best_score is None:
        best_pulse = np.zeros((N_STEPS, 2))
        best_score = 1.0
    
    # Phase 1: Grid search over DRAG parameters
    if time.time() < _DEADLINE - 60:
        drags = np.linspace(-0.25, 0.0, 11)
        sigmas = np.linspace(40.0/6.0 - 1.0, 40.0/6.0 + 1.0, 7)
        
        for drag in drags:
            if time.time() > _DEADLINE - 60:
                break
            for sigma in sigmas:
                if time.time() > _DEADLINE - 60:
                    break
                p = _param_pulse([1.0, drag, sigma, 0.0, 0.0])
                s = _compute_score_fast(p)
                if s is not None and s < best_score:
                    best_score = s
                    best_pulse = p
    
    # Phase 2: Direct optimization on the best pulse found
    if time.time() < _DEADLINE - 20:
        best_pulse, best_score = _direct_optimize(best_pulse, n_iter=150, step_size=0.015)
    
    # Phase 3: Fine-tune with smaller steps
    if time.time() < _DEADLINE - 10:
        best_pulse, best_score = _direct_optimize(best_pulse, n_iter=100, step_size=0.005)
    
    return best_pulse
