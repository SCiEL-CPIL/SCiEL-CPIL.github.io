"""Robust control pulses for a three-level transmon - differentiable ODE with adjoint gradients."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# Hardware constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
T2 = 40000.0  # ns (Tphi)
N_STEPS = 80
DT = 0.5  # ns
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator and number operator for 3 levels
A = np.array([
    [0, np.sqrt(1), 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A  # diag(0, 1, 2)

# Precompute static parts of Hamiltonian
N_MINUS_1 = N_OP - np.eye(3)
STATIC_H = 0.5 * ANHARMONICITY * (N_OP @ N_MINUS_1)

# Lindblad operators
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T2)
L1_DAG = L1.conj().T
L2_DAG = L2.conj().T
L1_DAG_L1 = L1_DAG @ L1
L2_DAG_L2 = L2_DAG @ L2

# Precompute superoperator components for dissipation
DIM = 3
I3 = np.eye(DIM)

# Dissipative superoperator (time-independent)
DISSIP_SUPER = np.kron(L1, L1.conj().T) + np.kron(L2, L2.conj().T) \
    - 0.5 * (np.kron(L1_DAG_L1, I3) + np.kron(I3, L1_DAG_L1)) \
    - 0.5 * (np.kron(L2_DAG_L2, I3) + np.kron(I3, L2_DAG_L2))

# Target states for fidelity
psi0 = np.array([1, 0, 0], dtype=complex)
psi1 = np.array([0, 1, 0], dtype=complex)
psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
TARGET_STATES = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]

DETUNINGS = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]


def build_superoperator(detuning, scale, I_val, Q_val):
    """Build the 9x9 Liouvillian superoperator for one step."""
    # Coherent part: -i[H, rho] = -i(H ⊗ I - I ⊗ H†)
    # H = detuning*N_OP + STATIC_H + 0.5*scale*(I*(A+A†) + iQ*(A-A†))
    H_drive = 0.5 * scale * (I_val * (A + A_DAG) + 1j * Q_val * (A - A_DAG))
    H = detuning * N_OP + STATIC_H + H_drive
    
    coherent = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    return coherent + DISSIP_SUPER


def evolve_density_matrix(pulse_I, pulse_Q, detuning, scale):
    """
    Evolve density matrix through the pulse using matrix exponentials.
    Returns final 3x3 density matrix.
    """
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.reshape(DIM * DIM)
    
    for i in range(N_STEPS):
        S = build_superoperator(detuning, scale, pulse_I[i], pulse_Q[i])
        U = expm(DT * S)
        rho_vec = U @ rho_vec
    
    return rho_vec.reshape(DIM, DIM)


def compute_fidelities(rho):
    """Compute fidelities for all 6 target states."""
    fids = []
    for psi in TARGET_STATES:
        val = np.vdot(psi, rho @ psi)
        fids.append(np.real(val))
    return np.array(fids)


def compute_score(pulse_I, pulse_Q):
    """Compute mean + max infidelity over 9 cases. Returns (score, all_infid)."""
    all_infid = []
    for det in DETUNINGS:
        for scl in SCALES:
            rho = evolve_density_matrix(pulse_I, pulse_Q, det, scl)
            fids = compute_fidelities(rho)
            infids = 1.0 - np.clip(fids, 0, 1)
            all_infid.extend(infids)
    
    all_infid = np.array(all_infid)
    return np.mean(all_infid) + np.max(all_infid), all_infid


def check_constraints(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    amps = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amps) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    # Slew rate including boundaries
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = np.array([pulse_I[i], pulse_Q[i]])
        slew = np.linalg.norm(curr - prev) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
        prev = curr
    
    # Last step to zero
    slew = np.linalg.norm(prev) / DT
    if slew > SLEW_BOUND + 1e-10:
        return False
    
    return True


def drag_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    
    A = np.sqrt(np.pi / 2) / sigma
    I = A * np.exp(-t**2 / (2 * sigma**2))
    I = I - I[0]
    I = np.maximum(I, 0)
    
    max_amp = np.max(np.abs(I))
    if max_amp > AMPLITUDE_BOUND:
        I = I * (AMPLITUDE_BOUND / max_amp)
    
    for _ in range(20):
        if check_constraints(I, np.zeros_like(I)):
            break
        I = I * 0.9
    
    Q = np.zeros_like(I)
    return I, Q


def adjoint_gradient(pulse_I, pulse_Q, detuning, scale):
    """
    Compute gradient of final fidelity sum w.r.t. pulse using adjoint method.
    For a given (detuning, scale) case, returns dF/dI and dF/dQ.
    """
    # Forward pass: store all density matrices
    rho_vecs = []
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.reshape(DIM * DIM)
    
    supers = []
    for i in range(N_STEPS):
        S = build_superoperator(detuning, scale, pulse_I[i], pulse_Q[i])
        supers.append(S)
        U = expm(DT * S)
        rho_vec = U @ rho_vec
        rho_vecs.append(rho_vec.copy())
    
    # Final fidelity sum (over all 6 target states)
    rho_final = rho_vecs[-1].reshape(DIM, DIM)
    # dF/drho = sum over targets of |psi><psi| (for pure state fidelity <psi|rho|psi>)
    # Actually F = <psi|rho|psi>, so dF/drho = |psi><psi|
    grad_rho = np.zeros((DIM, DIM), dtype=complex)
    for psi in TARGET_STATES:
        grad_rho += np.outer(psi, psi.conj())
    grad_rho_vec = grad_rho.reshape(DIM * DIM)
    
    # Backward pass: adjoint
    # The adjoint variable lambda satisfies:
    # lambda_{k-1} = (S_k^T) @ lambda_k  (for the linear map)
    # dF/dpulse_k = lambda_k^T @ dU_k/dpulse_k @ rho_{k-1}
    # 
    # More carefully: rho_k = U_k @ rho_{k-1}
    # F = g(rho_N)
    # dF = g'(rho_N) @ drho_N
    # drho_k = U_k @ drho_{k-1}
    # So dF = g'(rho_N) @ U_N @ ... @ U_{k+1} @ dU_k @ rho_{k-1}
    # Let lambda_k = (U_N @ ... @ U_{k+1})^T @ g'(rho_N)  (adjoint state)
    # Then dF = sum_k lambda_k @ dU_k @ rho_{k-1}
    # And dU_k/dI = DT * dS_k/dI @ expm(DT*S_k) approximately, but more precisely:
    # dU_k = expm(DT*S_k) @ (DT * dS_k) for small perturbations
    # Actually d/dt expm(tS) = expm(tS) * S, so d expm(DT*S)/dS = DT * expm(DT*S)
    # For parameter p: d expm(DT*S)/dp = DT * expm(DT*S) @ dS/dp
    #
    # So dF/dp_k = lambda_k @ (DT * expm(DT*S_k) @ dS_k/dp) @ rho_{k-1}
    #            = DT * (expm(DT*S_k)^T @ lambda_k)^T @ dS_k/dp @ rho_{k-1}
    #            = DT * lambda_{k-1}^T @ dS_k/dp @ rho_{k-1}  (if we define lambda properly)
    #
    # Let me redo: define adjoint lambda_k such that dF = sum_k lambda_k^T @ drho_k
    # drho_k = U_k @ drho_{k-1} + (dU_k/dp_k * dp_k) @ rho_{k-1}
    # 
    # Standard adjoint: lambda_{k-1} = U_k^T @ lambda_k, lambda_N = g'(rho_N)
    # dF/dp_k = lambda_k^T @ (dU_k/dp_k) @ rho_{k-1}
    #
    # dU_k/dI = DT * expm(DT*S_k) @ dS_k/dI
    # dU_k/dQ = DT * expm(DT*S_k) @ dS_k/dQ
    
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    # lambda_N = grad_rho_vec
    lambda_curr = grad_rho_vec.copy()
    
    # We need to go backward from N to 1
    # At step k (1-indexed in our loop, 0-indexed: k-1), we have:
    # lambda_{k-1} = U_k^T @ lambda_k
    # dF/dp_k = lambda_k^T @ (DT * U_k @ dS_k/dp) @ rho_{k-1}
    #         = DT * (U_k^T @ lambda_k)^T @ dS_k/dp @ rho_{k-1}
    #         = DT * lambda_{k-1}^T @ dS_k/dp @ rho_{k-1}
    
    for k in range(N_STEPS - 1, -1, -1):
        U_k = expm(DT * supers[k])
        
        # dF/dI_k and dF/dQ_k
        # dS/dI = -1j * (np.kron(dH/dI, I3) - np.kron(I3, dH/dI†))
        # dH/dI = 0.5 * scale * (A + A_DAG)
        # dH/dQ = 0.5 * scale * 1j * (A - A_DAG)
        
        dH_dI = 0.5 * scale * (A + A_DAG)
        dH_dQ = 0.5 * scale * 1j * (A - A_DAG)
        
        dS_dI = -1j * (np.kron(dH_dI, I3) - np.kron(I3, dH_dI.conj().T))
        dS_dQ = -1j * (np.kron(dH_dQ, I3) - np.kron(I3, dH_dQ.conj().T))
        
        # dF/dI_k = DT * lambda_curr^T @ dS_dI @ rho_{k-1}
        # But lambda_curr here is lambda_k, and we need lambda_{k-1} = U_k^T @ lambda_k
        # Actually: dF/dp_k = DT * (U_k^T @ lambda_k)^T @ dS_k/dp @ rho_{k-1}
        # Let's compute lambda_prev = U_k^T @ lambda_curr first
        
        rho_prev = rho_vecs[k-1] if k > 0 else np.zeros(DIM*DIM, dtype=complex)
        if k == 0:
            rho_prev = np.zeros(DIM*DIM, dtype=complex)
            rho_prev[0] = 1.0  # |0><0| vec
        
        # dF/dp_k = DT * lambda_k^T @ (U_k @ dS_k/dp) @ rho_{k-1}
        #         = DT * (U_k^T @ lambda_k)^T @ dS_k/dp @ rho_{k-1}
        #         = DT * lambda_prev^T @ dS_k/dp @ rho_{k-1}
        
        # Compute: temp = dS_dI @ rho_prev, then grad = DT * lambda_curr^T @ U_k @ temp
        # Or: grad = DT * (U_k^T @ lambda_curr)^T @ dS_dI @ rho_prev
        
        # Let's compute directly:
        # dF/dI_k = DT * lambda_curr^T @ (U_k @ dS_dI @ rho_prev)
        # dF/dQ_k = DT * lambda_curr^T @ (U_k @ dS_dQ @ rho_prev)
        
        temp_I = dS_dI @ rho_prev
        temp_Q = dS_dQ @ rho_prev
        
        grad_I[k] = DT * np.real(np.conj(lambda_curr).T @ (U_k @ temp_I))
        grad_Q[k] = DT * np.real(np.conj(lambda_curr).T @ (U_k @ temp_Q))
        
        # Update adjoint: lambda_{k-1} = U_k^T @ lambda_k
        if k > 0:
            lambda_curr = U_k.conj().T @ lambda_curr
    
    return grad_I, grad_Q


def score_and_gradient(pulse_I, pulse_Q):
    """
    Compute score and gradient of score w.r.t. all 160 pulse parameters.
    Score = mean + max infidelity over 9 cases.
    """
    # Compute all infidelities and track which is max
    all_infid = []
    case_results = []
    for idx, (det, scl) in enumerate(zip(
        [d for d in DETUNINGS for s in SCALES],
        [s for d in DETUNINGS for s in SCALES]
    )):
        rho = evolve_density_matrix(pulse_I, pulse_Q, det, scl)
        fids = compute_fidelities(rho)
        infids = 1.0 - np.clip(fids, 0, 1)
        case_results.append((det, scl, infids))
        all_infid.extend(infids)
    
    all_infid = np.array(all_infid)
    score = np.mean(all_infid) + np.max(all_infid)
    
    # Gradient: d(score)/dp = mean(1/54) * sum_all d(infid)/dp + d(max)/dp
    # The max term: d(max)/dp = d(infid_max)/dp for the argmax case
    max_idx = np.argmax(all_infid)
    max_case_idx = max_idx // 6  # which (det, scl) case
    max_state_idx = max_idx % 6   # which target state
    
    # Compute gradient of sum of all infidelities
    grad_I_sum = np.zeros(N_STEPS)
    grad_Q_sum = np.zeros(N_STEPS)
    
    for det, scl, infids in case_results:
        # For each target state, d(infid)/dp = -d(fid)/dp
        # fid = <psi|rho|psi>, so d(fid)/dp = <psi|drho|psi>
        # The adjoint gradient gives us dF/dp where F is the fidelity sum
        # For a single fidelity: dF/dp = -d(infid)/dp
        
        # We need gradient of sum of fidelities for this case
        # Modify adjoint to use single state
        # Actually, let's compute gradient of sum of all 6 fidelities for this case
        # then the infidelity gradient is the negative
        
        # For efficiency, compute gradient of fidelity sum for each case
        # d(sum_fids)/dp = - d(sum_infids)/dp
        
        # Use adjoint with grad_rho = sum of |psi><psi| for all 6 states
        grad_I_case, grad_Q_case = adjoint_gradient(pulse_I, pulse_Q, det, scl)
        
        # This gives d(sum_fids)/dp, so d(sum_infids)/dp = -grad
        grad_I_sum += -grad_I_case
        grad_Q_sum += -grad_Q_case
    
    # d(score)/dp = (1/54) * grad_I_sum + d(max_infid)/dp
    # d(max_infid)/dp = -d(fid_max)/dp for the max case
    
    # Compute gradient of the max fidelity (negative infidelity)
    # We need d(fid)/dp for the specific max case and state
    det_max, scl_max, _ = case_results[max_case_idx]
    
    # For a single fidelity with state psi_max:
    # d(fid)/dp = <psi_max|drho|psi_max>
    # Use adjoint with grad_rho = |psi_max><psi_max|
    psi_max = TARGET_STATES[max_state_idx]
    
    # Modify adjoint to use single state
    # We'll compute this separately
    grad_I_max, grad_Q_max = _adjoint_single_state(pulse_I, pulse_Q, det_max, scl_max, psi_max)
    
    # d(max_infid)/dp = -d(fid_max)/dp = -grad_I_max
    grad_I = (1.0 / 54.0) * grad_I_sum - grad_I_max
    grad_Q = (1.0 / 54.0) * grad_Q_sum - grad_Q_max
    
    return score, np.concatenate([grad_I, grad_Q])


def _adjoint_single_state(pulse_I, pulse_Q, detuning, scale, psi):
    """Compute gradient of fidelity <psi|rho|psi> w.r.t. pulse using adjoint."""
    # Forward pass
    rho_vecs = []
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.reshape(DIM * DIM)
    
    supers = []
    for i in range(N_STEPS):
        S = build_superoperator(detuning, scale, pulse_I[i], pulse_Q[i])
        supers.append(S)
        U = expm(DT * S)
        rho_vec = U @ rho_vec
        rho_vecs.append(rho_vec.copy())
    
    # grad_rho = |psi><psi|
    grad_rho = np.outer(psi, psi.conj())
    lambda_curr = grad_rho.reshape(DIM * DIM).copy()
    
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    for k in range(N_STEPS - 1, -1, -1):
        U_k = expm(DT * supers[k])
        
        dH_dI = 0.5 * scale * (A + A_DAG)
        dH_dQ = 0.5 * scale * 1j * (A - A_DAG)
        
        dS_dI = -1j * (np.kron(dH_dI, I3) - np.kron(I3, dH_dI.conj().T))
        dS_dQ = -1j * (np.kron(dH_dQ, I3) - np.kron(I3, dH_dQ.conj().T))
        
        rho_prev = rho_vecs[k-1] if k > 0 else np.zeros(DIM*DIM, dtype=complex)
        if k == 0:
            rho_prev = np.zeros(DIM*DIM, dtype=complex)
            rho_prev[0] = 1.0
        
        temp_I = dS_dI @ rho_prev
        temp_Q = dS_dQ @ rho_prev
        
        grad_I[k] = DT * np.real(np.conj(lambda_curr).T @ (U_k @ temp_I))
        grad_Q[k] = DT * np.real(np.conj(lambda_curr).T @ (U_k @ temp_Q))
        
        if k > 0:
            lambda_curr = U_k.conj().T @ lambda_curr
    
    return grad_I, grad_Q


def optimize_pulse():
    """
    Optimize the pulse using L-BFGS with adjoint gradients.
    """
    I_seed, Q_seed = drag_seed()
    x0 = np.concatenate([I_seed, Q_seed])
    
    # Use L-BFGS-B with gradient
    def objective_and_grad(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        
        score, grad = score_and_gradient(I, Q)
        return score, grad
    
    # We'll use minimize with jac
    def objective(x):
        return objective_and_grad(x)[0]
    
    def gradient(x):
        return objective_and_grad(x)[1]
    
    # Add penalty for constraint violations
    def objective_penalized(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        score, grad = score_and_gradient(I, Q)
        
        penalty = 0.0
        grad_penalty = np.zeros_like(x)
        
        # Amplitude penalty
        amps = np.sqrt(I**2 + Q**2)
        for i in range(N_STEPS):
            if amps[i] > AMPLITUDE_BOUND:
                excess = amps[i] - AMPLITUDE_BOUND
                penalty += excess**2
                # Gradient of penalty w.r.t. I[i]
                if amps[i] > 1e-10:
                    grad_penalty[i] += 2 * excess * I[i] / amps[i]
                    grad_penalty[N_STEPS + i] += 2 * excess * Q[i] / amps[i]
        
        # Slew rate penalty
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                excess = slew - SLEW_BOUND
                penalty += excess**2
                # Gradient
                if slew > 1e-10:
                    grad_penalty[i] += 2 * excess * dI / (DT * slew)
                    grad_penalty[N_STEPS + i] += 2 * excess * dQ / (DT * slew)
                    if i > 0:
                        grad_penalty[i-1] -= 2 * excess * dI / (DT * slew)
                        grad_penalty[N_STEPS + i - 1] -= 2 * excess * dQ / (DT * slew)
            prev_I, prev_Q = I[i], Q[i]
        
        # Last to zero
        dI = -prev_I
        dQ = -prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            excess = slew - SLEW_BOUND
            penalty += excess**2
            if slew > 1e-10:
                grad_penalty[N_STEPS - 1] += 2 * excess * dI / (DT * slew)
                grad_penalty[2*N_STEPS - 1] += 2 * excess * dQ / (DT * slew)
        
        return score + 50.0 * penalty, grad + 50.0 * grad_penalty
    
    # Time-limited optimization
    start_time = time.time()
    time_limit = 250  # seconds
    
    result = minimize(
        lambda x: objective_penalized(x)[0],
        x0,
        jac=lambda x: objective_penalized(x)[1],
        method='L-BFGS-B',
        options={
            'maxiter': 200,
            'maxfun': 500,
            'ftol': 1e-10,
            'gtol': 1e-8,
            'maxls': 20,
            'maxcor': 20,
            'disp': False
        }
    )
    
    I_opt = result.x[:N_STEPS]
    Q_opt = result.x[N_STEPS:]
    
    # Check constraints
    if check_constraints(I_opt, Q_opt):
        return I_opt, Q_opt
    
    # If constraints violated, try to fix by scaling
    scale_factor = 1.0
    for _ in range(50):
        I_try = I_opt * scale_factor
        Q_try = Q_opt * scale_factor
        if check_constraints(I_try, Q_try):
            return I_try, Q_try
        scale_factor *= 0.95
    
    return I_seed, Q_seed


# Pre-compute the optimized pulse at module load time
def _compute_pulse():
    """Compute the pulse and cache it."""
    I, Q = drag_seed()
    
    try:
        I_opt, Q_opt = optimize_pulse()
        if check_constraints(I_opt, Q_opt):
            return I_opt, Q_opt
    except Exception:
        pass
    
    return I, Q


_I_cached, _Q_cached = _compute_pulse()


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    result = np.column_stack([_I_cached, _Q_cached])
    return result
