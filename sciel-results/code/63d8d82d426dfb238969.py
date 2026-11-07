"""Robust DRAG pulse for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_GATE = 40.0  # ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
T_PHI = 40000.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns

# Time deadline guard
DEADLINE = 320.0  # seconds, safety margin before 330s kill


def _build_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian for a given detuning, scale, and pulse amplitudes."""
    # a matrix: diag(sqrt([1,2]), 1)
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    # n = a^dag a
    n = a_dag @ a
    
    # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    n_nm1 = n @ (n - np.eye(3))
    
    H = (detuning * n + 
         0.5 * ANHARMONICITY * n_nm1 + 
         0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
    return H


def _build_liouvillian(H, rho_shape=(3, 3)):
    """Build the Liouvillian superoperator (vectorized)."""
    d = 3
    # Lindblad operators
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / T_PHI)
    
    # Vectorize rho (column-major)
    # d(rho)/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # In vectorized form: d(rho_vec)/dt = L_vec @ rho_vec
    # where L_vec = -i (H_kron_I - I_kron_H^T) + sum_k (L_k_kron_L_k^* - 0.5 (L_k^dag L_k)_kron_I - 0.5 (I_kron_(L_k^dag L_k)^T))
    
    H_kron_I = np.kron(H, np.eye(d))
    I_kron_H_T = np.kron(np.eye(d), H.conj().T)
    
    L_vec = -1j * (H_kron_I - I_kron_H_T)
    
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_L_dag = L @ L_dag
        L_vec += np.kron(L, L.conj()) - 0.5 * np.kron(L_L_dag, np.eye(d)) - 0.5 * np.kron(np.eye(d), L_L_dag.T)
    
    return L_vec


def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    d = 3
    # Initial state: rho = |0><0|
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    
    # We'll integrate the Liouvillian step by step
    # For efficiency, we can use a simple exponential map per step
    # But let's use solve_ivp for the full trajectory for accuracy
    
    def liouvillian(t, y):
        rho_mat = y.reshape((d, d))
        # We need I and Q at time t
        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1
        I_val = I_arr[idx]
        Q_val = Q_arr[idx]
        H = _build_hamiltonian(detuning, scale, I_val, Q_val)
        L_vec = _build_liouvillian(H)
        return L_vec @ y
    
    y0 = rho.flatten()
    t_eval = np.linspace(0, T_GATE, N_STEPS + 1)
    
    sol = solve_ivp(liouvillian, [0, T_GATE], y0, method='DOP853', 
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return None
    
    rho_final = sol.y[:, -1].reshape((d, d))
    return rho_final


def _fidelity(rho, psi):
    """Compute fidelity between density matrix rho and state psi."""
    psi_mat = np.outer(psi, psi.conj())
    # Fidelity = Tr(rho * psi_mat) for pure state
    fid = np.trace(rho @ psi_mat).real
    return max(0.0, min(1.0, fid))


def _compute_score(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j * psi1) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    fidelities = []
    for det in detunings:
        for scale in scales:
            rho_final = _simulate_pulse(I_arr, Q_arr, det, scale)
            if rho_final is None:
                return np.inf
            for psi in target_states:
                fid = _fidelity(rho_final, psi)
                fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity


def _check_constraints(I_arr, Q_arr):
    """Check if the pulse satisfies amplitude and slew constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        if np.sqrt(I_arr[i]**2 + Q_arr[i]**2) > AMPLITUDE_BOUND + 1e-10:
            return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # From zero to step 1
    if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / DT > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        dI = I_arr[i+1] - I_arr[i]
        dQ = Q_arr[i+1] - Q_arr[i]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True


def _gaussian_pulse(sigma, beta):
    """Generate a DRAG Gaussian pulse with area pi."""
    t = np.arange(N_STEPS) * DT
    # Center the pulse
    t_center = T_GATE / 2.0
    t_rel = t - t_center
    
    # Gaussian envelope
    envelope = np.exp(-t_rel**2 / (2 * sigma**2))
    
    # Normalize to have area pi
    area = np.sum(envelope) * DT
    if area > 0:
        envelope = envelope * (np.pi / area)
    
    # DRAG correction: Q = beta * dI/dt
    I_arr = envelope.copy()
    # Compute derivative using central differences
    Q_arr = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        if i == 0:
            dIdt = (I_arr[1] - I_arr[0]) / DT
        elif i == N_STEPS - 1:
            dIdt = (I_arr[-1] - I_arr[-2]) / DT
        else:
            dIdt = (I_arr[i+1] - I_arr[i-1]) / (2 * DT)
        Q_arr[i] = beta * dIdt
    
    return I_arr, Q_arr


def _objective(params):
    """Objective function for optimization: minimize infidelity."""
    sigma, beta, amp_scale = params
    
    # Constrain sigma to reasonable range
    sigma = np.clip(sigma, 3.0, 15.0)
    amp_scale = np.clip(amp_scale, 0.5, 1.0)
    
    I_arr, Q_arr = _gaussian_pulse(sigma, beta)
    I_arr = amp_scale * I_arr
    Q_arr = amp_scale * Q_arr
    
    if not _check_constraints(I_arr, Q_arr):
        return 1.0  # Penalty for invalid
    
    return _compute_score(I_arr, Q_arr)


def pulse(target_gate, hardware_model):
    """Generate a robust DRAG pulse for the X-pi gate."""
    start_time = time.time()
    
    # Initial guess
    sigma0 = 40.0 / 6.0  # ~6.67 ns
    beta0 = 0.5  # Initial beta guess
    amp_scale0 = 1.0
    
    # Run Nelder-Mead optimization
    best_result = None
    best_score = np.inf
    
    # Multi-start with a few different initial guesses
    initial_guesses = [
        [sigma0, beta0, amp_scale0],
        [sigma0 * 0.8, beta0 * 0.8, amp_scale0],
        [sigma0 * 1.2, beta0 * 1.2, amp_scale0],
        [sigma0, beta0 * 0.5, amp_scale0],
        [sigma0, beta0 * 1.5, amp_scale0],
    ]
    
    for x0 in initial_guesses:
        if time.time() - start_time > DEADLINE:
            break
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to default Gaussian
        I_arr, Q_arr = _gaussian_pulse(sigma0, beta0)
    else:
        sigma, beta, amp_scale = best_result
        sigma = np.clip(sigma, 3.0, 15.0)
        amp_scale = np.clip(amp_scale, 0.5, 1.0)
        I_arr, Q_arr = _gaussian_pulse(sigma, beta)
        I_arr = amp_scale * I_arr
        Q_arr = amp_scale * Q_arr
    
    # Final constraint check and projection if needed
    if not _check_constraints(I_arr, Q_arr):
        # Scale down to satisfy constraints
        max_amp = np.max(np.sqrt(I_arr**2 + Q_arr**2))
        if max_amp > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / max_amp * 0.99
            I_arr = I_arr * scale_factor
            Q_arr = Q_arr * scale_factor
        
        # Check slew
        for i in range(N_STEPS - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                # Smooth the pulse
                from scipy.ndimage import gaussian_filter1d
                I_arr = gaussian_filter1d(I_arr, sigma=1.0)
                Q_arr = gaussian_filter1d(Q_arr, sigma=1.0)
                break
    
    # Ensure the pulse has the correct area (approximately pi)
    # The Gaussian normalization should handle this, but verify
    
    # Return as (80, 2) array
    return np.column_stack([I_arr, Q_arr])

# EVOLVE-BLOCK-END
