"""Spectral-nulling DRAG pulse for three-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # in rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Time grid
t_grid = np.arange(N_STEPS) * DT  # 0 to 39.5 ns

# Precompute basis operators
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)


def build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for a given step."""
    # Hamiltonian
    H = (detuning * n_op +
         0.5 * ANHARMONICITY * n_op @ (n_op - 3 * np.eye(3)) +
         0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag)))
    
    # Superoperator: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We'll build it as a 9x9 matrix acting on vectorized rho (row-major)
    # Use the formula: vec(L(rho)) = (I ⊗ L - L^T ⊗ I + ...) vec(rho)
    # Actually, let's just compute it directly for 3x3 matrices
    
    d = 3
    L_super = np.zeros((d*d, d*d), dtype=complex)
    
    # Coherent part: -i[H, rho]
    # vec(-i[H, rho]) = -i (I ⊗ H - H^T ⊗ I) vec(rho)
    coherent = -1j * (np.kron(np.eye(d), H) - np.kron(H.T, np.eye(d)))
    
    # Dissipative part
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        # Lk rho Lk^dag
        term1 = np.kron(Lk, Lk_dag.T)  # vec(Lk rho Lk^dag) = (Lk^dag^T ⊗ Lk) vec(rho)
        # 0.5 {Lk^dag Lk, rho}
        Lk_dag_Lk = Lk_dag @ Lk
        term2 = 0.5 * (np.kron(np.eye(d), Lk_dag_Lk) + np.kron(Lk_dag_Lk.T, np.eye(d)))
        L_super += term1 - term2
    
    L_super += coherent
    return L_super


def simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix for |0> input."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        L_super = build_liouvillian(detuning, scale, I_vals[i], Q_vals[i])
        # Exponentiate: rho_next = exp(L_super * dt) vec(rho)
        # Use matrix exponential via eigen decomposition or scipy
        from scipy.linalg import expm
        U = expm(L_super * DT)
        rho_vec = U @ rho.reshape(9)
        rho = rho_vec.reshape(3, 3)
    
    return rho


def compute_fidelity(rho, target_state):
    """Compute fidelity |<target|psi>|^2 where rho is the final state."""
    # For a pure target state |t>, fidelity = <t|rho|t>
    return np.real(np.vdot(target_state, rho @ target_state))


def compute_score(pulse):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j * psi1) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    # X gate maps: |0> -> |1>, |1> -> |0>, etc.
    # Actually, X-pi gate: X = [[0,1],[1,0]] in the two-level subspace
    # So |0> -> |1>, |1> -> |0>, (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same,
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    # Target output states after X-pi:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = (|0>+|1>)/sqrt(2)
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -( |0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    target_outputs = [psi1, psi0, psi_plus, -psi_minus, 1j * psi_i_minus, -1j * psi_i_plus]
    
    fidelities = []
    for det in detunings:
        for scale in scales:
            rho = simulate_pulse(pulse, det, scale)
            for i, target in enumerate(target_outputs):
                # We need to simulate for each input state
                pass
    
    # Actually, we need to simulate for each input state separately
    # Let's redo this properly
    
    all_fidelities = []
    for det in detunings:
        for scale in scales:
            for input_state, output_state in zip(target_states, target_outputs):
                rho = simulate_pulse_for_input(pulse, det, scale, input_state)
                fid = np.real(np.vdot(output_state, rho @ output_state))
                all_fidelities.append(fid)
    
    infidelities = [1.0 - f for f in all_fidelities]
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score


def simulate_pulse_for_input(pulse, detuning, scale, input_state):
    """Simulate the pulse for a specific input state."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    rho = np.outer(input_state, input_state.conj())
    
    from scipy.linalg import expm
    for i in range(N_STEPS):
        L_super = build_liouvillian(detuning, scale, I_vals[i], Q_vals[i])
        U = expm(L_super * DT)
        rho_vec = U @ rho.reshape(9)
        rho = rho_vec.reshape(3, 3)
    
    return rho


def check_validity(pulse):
    """Check if the pulse satisfies the amplitude and slew constraints."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Amplitude bound
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > AMPLITUDE_BOUND + 1e-9:
        return False
    
    # Slew bound
    # From zero to step 1
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def drag_pulse(params):
    """Generate a DRAG pulse from parameters: [amp, drag_coeff, det_comp, q_offset, sigma, shift]"""
    amp, drag_coeff, det_comp, q_offset, sigma, shift = params
    
    # Main Gaussian envelope
    t = t_grid
    gaussian_main = np.exp(-0.5 * ((t - DURATION/2 + shift) / sigma) ** 2)
    
    # Normalize to have area pi
    area = np.sum(gaussian_main) * DT
    gaussian_main = gaussian_main * (np.pi / area)
    
    # Derivative-coupled DRAG component
    # d/dt of Gaussian
    gaussian_deriv = np.zeros_like(t)
    gaussian_deriv[1:] = (gaussian_main[1:] - gaussian_main[:-1]) / DT
    gaussian_deriv[:-1] = (gaussian_main[1:] - gaussian_main[:-1]) / DT
    
    # DRAG component is proportional to the derivative
    drag_envelope = drag_coeff * gaussian_deriv
    
    # I component
    I = amp * gaussian_main + det_comp * gaussian_main
    Q = q_offset * gaussian_main + drag_envelope
    
    # Apply offset to ensure zero at edges
    I_offset = np.mean(I)
    Q_offset = np.mean(Q)
    I = I - I_offset
    Q = Q - Q_offset
    
    # Ensure area is pi
    I_area = np.sum(I) * DT
    if abs(I_area) > 1e-10:
        I = I * (np.pi / I_area)
    
    pulse = np.column_stack([I, Q])
    
    # Clip to amplitude bound
    amps = np.sqrt(I**2 + Q**2)
    mask = amps > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amps[mask]
        I[mask] *= scale_factor
        Q[mask] *= scale_factor
    
    return pulse


def objective(params):
    """Objective function for optimization."""
    pulse = drag_pulse(params)
    if not check_validity(pulse):
        return 1e6
    try:
        return compute_score(pulse)
    except:
        return 1e6


# Initial parameters: [amp, drag_coeff, det_comp, q_offset, sigma, shift]
# Start with reasonable DRAG values
x0 = [0.35, 0.05, 0.0, 0.0, 6.0, 0.0]

# Optimization
start_time = time.time()
DEADLINE = 330.0
SAFETY_MARGIN = 10.0

best_params = x0.copy()
best_score = objective(best_params)

# Nelder-Mead optimization
try:
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    if result.fun < best_score:
        best_params = result.x
        best_score = result.fun
except:
    pass

# Check time
elapsed = time.time() - start_time
if elapsed > DEADLINE - SAFETY_MARGIN:
    # Return best found so far
    pulse = drag_pulse(best_params)
    if not check_validity(pulse):
        # Fallback to seed-like pulse
        t = t_grid
        sigma = 40.0 / 6.0
        gaussian = np.exp(-0.5 * ((t - DURATION/2) / sigma) ** 2)
        area = np.sum(gaussian) * DT
        gaussian = gaussian * (np.pi / area)
        I = 0.35 * gaussian
        Q = 0.05 * np.zeros_like(t)
        pulse = np.column_stack([I, Q])
    return pulse

# Additional optimization if time permits
if elapsed < DEADLINE - 30.0:
    # Try a few different starting points
    for x0_try in [[0.3, 0.1, 0.0, 0.0, 5.0, 0.0],
                   [0.4, 0.03, 0.0, 0.0, 7.0, 0.0],
                   [0.35, 0.08, 0.001, 0.0, 6.0, 0.5]]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        try:
            result = minimize(objective, x0_try, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except:
            pass

pulse = drag_pulse(best_params)
if not check_validity(pulse):
    # Fallback
    t = t_grid
    sigma = 40.0 / 6.0
    gaussian = np.exp(-0.5 * ((t - DURATION/2) / sigma) ** 2)
    area = np.sum(gaussian) * DT
    gaussian = gaussian * (np.pi / area)
    I = 0.35 * gaussian
    Q = 0.05 * np.zeros_like(t)
    pulse = np.column_stack([I, Q])

return pulse


def pulse(target_gate, hardware_model):
    """Entry point: return an (80, 2) array of I and Q amplitudes."""
    return _generate_pulse()


def _generate_pulse():
    """Generate the optimized pulse."""
    # Use precomputed best pulse or compute on the fly
    # For simplicity, use the optimization result from module load time
    # But we need to handle the case where optimization didn't run
    
    global _cached_pulse
    if '_cached_pulse' not in globals():
        # Run a quick optimization
        start_time = time.time()
        x0 = [0.35, 0.05, 0.0, 0.0, 6.0, 0.0]
        best_params = x0.copy()
        best_score = objective(best_params)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except:
            pass
        
        pulse = drag_pulse(best_params)
        if not check_validity(pulse):
            t = t_grid
            sigma = 40.0 / 6.0
            gaussian = np.exp(-0.5 * ((t - DURATION/2) / sigma) ** 2)
            area = np.sum(gaussian) * DT
            gaussian = gaussian * (np.pi / area)
            I = 0.35 * gaussian
            Q = 0.05 * np.zeros_like(t)
            pulse = np.column_stack([I, Q])
        
        _cached_pulse = pulse
    
    return _cached_pulse
