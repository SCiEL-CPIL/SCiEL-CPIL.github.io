import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _legendre_basis(n_terms, n_steps=N_STEPS):
    """Build Legendre polynomial basis matrix (n_steps x n_terms)."""
    t = np.linspace(-1, 1, n_steps)
    basis = np.zeros((n_steps, n_terms))
    for k in range(n_terms):
        Pk = np.polynomial.legendre.legval(t, np.zeros(k+1) if k == 0 else 
                                           np.zeros(k+1) + np.eye(k+1)[k])
        basis[:, k] = Pk
    return basis

def _pulse_from_coefficients(coeffs, n_terms=10):
    """Generate pulse from Legendre coefficients for I and Q channels."""
    t = np.linspace(-1, 1, N_STEPS)
    I_vals = np.zeros(N_STEPS)
    Q_vals = np.zeros(N_STEPS)
    for k in range(n_terms):
        Pk = np.polynomial.legendre.legval(t, np.zeros(k+1) + np.eye(k+1)[k])
        I_vals += coeffs[k] * Pk
        Q_vals += coeffs[k + n_terms] * Pk
    return np.column_stack([I_vals, Q_vals])

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    return True

def _simulate_state_vector(pulse, detuning, scale, a, a_dag, n):
    """Simulate using state vectors for all 6 inputs simultaneously."""
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Stack into 3x6 matrix
    states = np.column_stack([psi_0, psi_1, psi_2, psi_3, psi_4, psi_5])
    
    # Evolve each state
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        H = (detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) +
             0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag)))
        
        # Compute U = expm(-1j * H * DT)
        U = expm(-1j * H * DT)
        states = U @ states
    
    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    fidelities = []
    for j in range(6):
        psi_target = X_mat @ inputs[j]
        psi_out = states[:, j]
        fidelity = np.abs(np.vdot(psi_target, psi_out))**2
        fidelity = max(0.0, min(1.0, np.real(fidelity)))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _compute_score(pulse, a, a_dag, n):
    """Compute score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            fid = _simulate_state_vector(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    return np.mean(infidelities) + np.max(infidelities)

def _compute_score_smooth(pulse, a, a_dag, n, beta=50.0):
    """Smooth surrogate of mean + max using log-sum-exp."""
    infidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            fid = _simulate_state_vector(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    infidelities = np.array(infidelities)
    mean_infid = np.mean(infidelities)
    max_infid = beta * np.log(np.sum(np.exp(infidelities / beta)))
    return mean_infid + max_infid

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # Leave 30s safety margin
    
    a, a_dag, n = _build_matrices()
    n_terms = 8  # Number of Legendre terms per channel
    
    # Initial guess: DRAG-style Gaussian
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    sigma = 40.0 / 6.0
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    beta_drag = 0.2
    I_seed = gauss
    Q_seed = beta_drag * dgauss
    
    pulse_seed = np.column_stack([I_seed, Q_seed])
    # Enforce amplitude bound
    amps = np.sqrt(pulse_seed[:, 0]**2 + pulse_seed[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        pulse_seed *= AMPLITUDE_BOUND / max_amp
    
    best_pulse = pulse_seed.copy()
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Optimization using Legendre coefficients
    def coeffs_to_pulse(coeffs):
        t = np.linspace(-1, 1, N_STEPS)
        I_vals = np.zeros(N_STEPS)
        Q_vals = np.zeros(N_STEPS)
        for k in range(n_terms):
            Pk = np.polynomial.legendre.legval(t, np.zeros(k+1) + np.eye(k+1)[k])
            I_vals += coeffs[k] * Pk
            Q_vals += coeffs[k + n_terms] * Pk
        return np.column_stack([I_vals, Q_vals])
    
    # Extract initial coefficients by least squares
    t = np.linspace(-1, 1, N_STEPS)
    basis = np.zeros((N_STEPS, n_terms))
    for k in range(n_terms):
        basis[:, k] = np.polynomial.legendre.legval(t, np.zeros(k+1) + np.eye(k+1)[k])
    
    I_init, Q_init = np.linalg.lstsq(basis, I_seed, rcond=None)[0]
    I_init, Q_init = np.linalg.lstsq(basis, Q_seed, rcond=None)[0]
    coeffs_init = np.concatenate([I_init, Q_init])
    
    # Phase 1: Coarse optimization with smooth surrogate
    def objective_smooth(coeffs):
        p = coeffs_to_pulse(coeffs)
        if not _check_validity(p):
            return 1.0
        return _compute_score_smooth(p, a, a_dag, n)
    
    starts = [
        coeffs_init.copy(),
        coeffs_init + np.random.RandomState(1).normal(0, 0.05, 2*n_terms),
        coeffs_init + np.random.RandomState(2).normal(0, 0.1, 2*n_terms),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(objective_smooth, x_start, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5})
            p_candidate = coeffs_to_pulse(res.x)
            if _check_validity(p_candidate):
                score_candidate = _compute_score(p_candidate, a, a_dag, n)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_pulse = p_candidate.copy()
        except Exception:
            pass
    
    # Phase 2: Fine polish with exact objective
    if time.time() < deadline - 30.0:
        def objective_exact(coeffs):
            p = coeffs_to_pulse(coeffs)
            if not _check_validity(p):
                return 1.0
            return _compute_score(p, a, a_dag, n)
        
        # Re-extract coefficients from best_pulse
        I_best, Q_best = np.linalg.lstsq(basis, best_pulse[:, 0], rcond=None)[0]
        I_best, Q_best = np.linalg.lstsq(basis, best_pulse[:, 1], rcond=None)[0]
        coeffs_best = np.concatenate([I_best, Q_best])
        
        for x_start in [coeffs_best, coeffs_best + np.random.RandomState(3).normal(0, 0.02, 2*n_terms)]:
            if time.time() > deadline - 30.0:
                break
            try:
                res = minimize(objective_exact, x_start, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-7})
                p_candidate = coeffs_to_pulse(res.x)
                if _check_validity(p_candidate):
                    score_candidate = _compute_score(p_candidate, a, a_dag, n)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_pulse = p_candidate.copy()
            except Exception:
                pass
    
    return best_pulse
