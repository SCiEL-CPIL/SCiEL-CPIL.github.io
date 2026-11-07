import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _precompute_liouvillian_parts(a, a_dag, n):
    """Precompute parts of the Liouvillian that don't change with I, Q, detuning, scale."""
    I3 = np.eye(3, dtype=complex)
    
    # n and n(n-1) for the anharmonicity term
    n_np1 = n @ (n - I3)
    
    # Lindblad parts
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    L1_dag = L1.conj().T
    L2_dag = L2.conj().T
    
    # Precompute kron terms for Lindblad
    kron_L1L1d = np.kron(L1, L1.conj())
    kron_L1dL1_I = np.kron(L1_dag @ L1, I3)
    kron_I_L1dL1 = np.kron(I3, (L1_dag @ L1).T)
    
    kron_L2L2d = np.kron(L2, L2.conj())
    kron_L2dL2_I = np.kron(L2_dag @ L2, I3)
    kron_I_L2dL2 = np.kron(I3, (L2_dag @ L2).T)
    
    # Precompute kron terms for drive
    kron_a_plus = np.kron(a + a_dag, I3)
    kron_a_minus = np.kron(a - a_dag, I3)
    kron_I_a_plus_T = np.kron(I3, (a + a_dag).T)
    kron_I_a_minus_T = np.kron(I3, (a - a_dag).T)
    
    # Precompute kron for detuning and anharmonicity
    kron_n_I = np.kron(n, I3)
    kron_I_n_T = np.kron(I3, n.T)
    kron_nn1_I = np.kron(n_np1, I3)
    kron_I_nn1_T = np.kron(I3, n_np1.T)
    
    return {
        'I3': I3,
        'n_np1': n_np1,
        'kron_L1L1d': kron_L1L1d,
        'kron_L1dL1_I': kron_L1dL1_I,
        'kron_I_L1dL1': kron_I_L1dL1,
        'kron_L2L2d': kron_L2L2d,
        'kron_L2dL2_I': kron_L2dL2_I,
        'kron_I_L2dL2': kron_I_L2dL2,
        'kron_a_plus': kron_a_plus,
        'kron_a_minus': kron_a_minus,
        'kron_I_a_plus_T': kron_I_a_plus_T,
        'kron_I_a_minus_T': kron_I_a_minus_T,
        'kron_n_I': kron_n_I,
        'kron_I_n_T': kron_I_n_T,
        'kron_nn1_I': kron_nn1_I,
        'kron_I_nn1_T': kron_I_nn1_T,
    }

def _liouvillian(I_val, Q_val, detuning, scale, parts):
    """Build Liouvillian using precomputed parts."""
    I3 = parts['I3']
    
    # H = detuning * n + 0.5 * ANHARMONICITY * n(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    # L_ham = -1j * (kron(H, I3) - kron(I3, H.T))
    
    L_ham = -1j * (
        detuning * (parts['kron_n_I'] - parts['kron_I_n_T']) +
        0.5 * ANHARMONICITY * (parts['kron_nn1_I'] - parts['kron_I_nn1_T']) +
        0.5 * scale * I_val * (parts['kron_a_plus'] - parts['kron_I_a_plus_T']) +
        0.5 * scale * Q_val * (1j * parts['kron_a_minus'] - 1j * parts['kron_I_a_minus_T'])
    )
    
    L_diss = (
        parts['kron_L1L1d'] - 0.5 * (parts['kron_L1dL1_I'] + parts['kron_I_L1dL1']) +
        parts['kron_L2L2d'] - 0.5 * (parts['kron_L2dL2_I'] + parts['kron_I_L2dL2'])
    )
    
    return L_ham + L_diss

def _simulate_pulse(pulse, detuning, scale, parts, a, a_dag, n):
    """Simulate pulse and return mean fidelity over 6 input states."""
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    # Initial density matrices vectorized (column-major)
    initial_states = []
    for psi in inputs:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    states = np.column_stack(initial_states)  # 9x6
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _liouvillian(I_val, Q_val, detuning, scale, parts)
        U = expm(L * DT)
        states = U @ states
    
    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-10:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _compute_score(pulse, parts, a, a_dag, n):
    """Compute score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse(pulse, detuning, scale, parts, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _enforce_constraints(pulse):
    """Project pulse onto feasible set: amplitude and slew bounds."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    # Enforce slew from 0 to step 0
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.linalg.norm(pulse[0])
        pulse[0] *= scale
    
    # Enforce slew between steps
    for i in range(N_STEPS - 1):
        diff = pulse[i+1] - pulse[i]
        norm_diff = np.linalg.norm(diff) / DT
        if norm_diff > SLEW_BOUND:
            # Move pulse[i+1] closer to pulse[i]
            max_step = SLEW_BOUND * DT
            direction = diff / np.linalg.norm(diff)
            pulse[i+1] = pulse[i] + direction * max_step
    
    # Enforce slew from last step to 0
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.linalg.norm(pulse[-1])
        pulse[-1] *= scale
    
    return pulse

def _generate_drag_pulse(A, sigma, beta, offset=0.0):
    """Generate a DRAG-style pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = 20.0 + offset
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    area = np.sum(gauss) * DT
    if abs(area) > 1e-10:
        gauss = gauss / area
    
    # Derivative for Q
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = A * gauss
    Q = beta * A * dgauss
    
    pulse = np.column_stack([I, Q])
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 280.0
    
    a, a_dag, n = _build_matrices()
    parts = _precompute_liouvillian_parts(a, a_dag, n)
    
    # Generate several initial guesses
    initial_pulses = []
    
    # DRAG with various parameters
    for A in [0.3, 0.35, 0.4]:
        for sigma in [4.0, 5.0, 6.0]:
            for beta in [-0.2, 0.0, 0.2]:
                p = _generate_drag_pulse(A, sigma, beta)
                p = _enforce_constraints(p)
                initial_pulses.append(p)
    
    # Flat-top pulse
    t = np.arange(N_STEPS) * DT
    flat = np.zeros(N_STEPS)
    flat[10:70] = 1.0
    area = np.sum(flat) * DT
    flat = flat / area
    I_flat = 0.35 * flat
    Q_flat = np.zeros(N_STEPS)
    pulse_flat = np.column_stack([I_flat, Q_flat])
    pulse_flat = _enforce_constraints(pulse_flat)
    initial_pulses.append(pulse_flat)
    
    # Evaluate all initial guesses
    best_pulse = None
    best_score = float('inf')
    
    for p in initial_pulses:
        if time.time() > deadline - 60.0:
            break
        if not _check_validity(p):
            continue
        score = _compute_score(p, parts, a, a_dag, n)
        if score < best_score:
            best_score = score
            best_pulse = p.copy()
    
    # If no valid pulse found, use a simple fallback
    if best_pulse is None:
        best_pulse = _generate_drag_pulse(0.3, 5.0, 0.0)
        best_pulse = _enforce_constraints(best_pulse)
        best_score = _compute_score(best_pulse, parts, a, a_dag, n)
    
    # Define objective for L-BFGS-B
    def objective(params):
        p = params.reshape(N_STEPS, 2)
        p = _enforce_constraints(p)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, parts, a, a_dag, n)
    
    # L-BFGS-B optimization with bounds
    # Bounds: each component in [-0.4, 0.4]
    bounds = [(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)] * (2 * N_STEPS)
    
    # Try L-BFGS-B from the best initial guess
    if time.time() < deadline - 30.0:
        try:
            x0 = best_pulse.flatten()
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
            if res.fun < best_score:
                best_pulse = _enforce_constraints(res.x.reshape(N_STEPS, 2))
                best_score = res.fun
        except Exception:
            pass
    
    # Try a few more local optimizations from different starts if time permits
    remaining_starts = [p for p in initial_pulses if not np.allclose(p, best_pulse)]
    for p in remaining_starts[:3]:
        if time.time() > deadline - 20.0:
            break
        try:
            x0 = p.flatten()
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 20, 'ftol': 1e-7, 'gtol': 1e-5, 'maxls': 10})
            if res.fun < best_score:
                best_pulse = _enforce_constraints(res.x.reshape(N_STEPS, 2))
                best_score = res.fun
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse):
        # Fallback to a known valid pulse
        best_pulse = _generate_drag_pulse(0.3, 5.0, 0.0)
        best_pulse = _enforce_constraints(best_pulse)
    
    return best_pulse
