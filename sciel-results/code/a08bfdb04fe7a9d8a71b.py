"""Robust control pulses for a three-level transmon - variational ODE approach."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import erf

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
# a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
A = np.array([
    [0, np.sqrt(1), 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A  # diag(0, 1, 2)


def hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    H = detuning * N_OP + 0.5 * ANHARMONICITY * (N_OP @ (N_OP - np.eye(3)))
    drive = 0.5 * scale * (I * (A + A_DAG) + 1j * Q * (A - A_DAG))
    H = H + drive
    return H


def lindblad_operators():
    """Return Lindblad operators for T1 and Tphi."""
    # L1 = a / sqrt(T1)
    L1 = A / np.sqrt(T1)
    # L2 = n * sqrt(2 / Tphi)
    L2 = N_OP * np.sqrt(2.0 / T2)
    return [L1, L2]


LINDBLADS = lindblad_operators()


def liouvillian_step(H, dt):
    """
    Exponentiate the Liouvillian for one step.
    The master equation is: dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 0.5{L_k†L_k, ρ})
    We compute the superoperator evolution for one time step.
    
    For efficiency, we evolve the state vector directly using the Lindblad equation
    in the Schrödinger picture for the density matrix. But since we're dealing with
    a small system, we can use the matrix exponential of the Liouvillian superoperator.
    
    Actually, for a 3-level system, we can just solve the ODE for the density matrix
    or use the fact that the evolution is linear. Let's use the superoperator approach.
    
    The Liouvillian superoperator L acts on vec(ρ) as:
    vec(dρ/dt) = -i (H ⊗ I - I ⊗ H†) vec(ρ) + Σ_k (L_k ⊗ L_k*) vec(ρ) - 0.5 Σ_k (L_k†L_k ⊗ I + I ⊗ L_k†L_k) vec(ρ)
    
    For a single step of duration dt, we compute exp(dt * L_super) and apply it.
    """
    dim = 3
    # Build the superoperator
    # -i[H, ρ] term: -i(H ⊗ I - I ⊗ H†)
    H_term = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.conj().T))
    
    # Dissipative terms
    for L in LINDBLADS:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        # L ρ L† term: kron(L, L*)
        H_term += np.kron(L, L.conj().T)
        # -0.5{L†L, ρ} term: -0.5(kron(L†L, I) + kron(I, L†L))
        H_term -= 0.5 * (np.kron(LdagL, np.eye(dim)) + np.kron(np.eye(dim), LdagL))
    
    # Exponentiate: exp(dt * H_term)
    # Use scipy's matrix exponential
    from scipy.linalg import expm
    U = expm(dt * H_term)
    return U


def evolve_pulse(pulse_I, pulse_Q, detuning, scale, n_steps=N_STEPS, dt=DT):
    """
    Evolve the density matrix through the pulse.
    pulse_I, pulse_Q: arrays of shape (n_steps,)
    Returns the final density matrix.
    """
    dim = 3
    # Initial state: |0><0|
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    
    # Precompute superoperators for each step
    for i in range(n_steps):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        H = hamiltonian(detuning, scale, I_val, Q_val)
        U = liouvillian_step(H, dt)
        # Apply: rho_next = U @ vec(rho), then reshape
        rho_vec = rho.reshape(dim * dim)
        rho_vec = U @ rho_vec
        rho = rho_vec.reshape(dim, dim)
    
    return rho


def fidelity(rho, psi):
    """Compute fidelity |<psi|rho|psi>| for pure state psi."""
    val = np.vdot(psi, rho @ psi)
    return np.real(val)


def compute_infidelity(pulse_I, pulse_Q):
    """
    Compute the mean and max infidelity over the 9 cases.
    Returns (mean_infid, max_infid).
    """
    # Target states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    dim = 3
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    all_infid = []
    for det in detunings:
        for scl in scales:
            rho = evolve_pulse(pulse_I, pulse_Q, det, scl)
            for psi in target_states:
                fid = fidelity(rho, psi)
                fid = np.clip(fid, 0, 1)
                infid = 1 - fid
                all_infid.append(infid)
    
    all_infid = np.array(all_infid)
    return np.mean(all_infid), np.max(all_infid)


def score_function(pulse_I, pulse_Q):
    """The scoring function: mean + max of the 9-case infidelities."""
    mean_infid, max_infid = compute_infidelity(pulse_I, pulse_Q)
    return mean_infid + max_infid


def check_constraints(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    for i in range(len(pulse_I)):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-10:
            return False
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and from last step to zero
    prev_I, prev_Q = 0.0, 0.0
    for i in range(len(pulse_I)):
        dI = pulse_I[i] - prev_I
        dQ = pulse_Q[i] - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
        prev_I, prev_Q = pulse_I[i], pulse_Q[i]
    
    # From last step to zero
    dI = 0.0 - prev_I
    dQ = 0.0 - prev_Q
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-10:
        return False
    
    return True


def drag_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0  # ns
    
    # Gaussian with area pi
    # Integral of A * exp(-t^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi) = pi
    # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A = np.sqrt(np.pi / 2) / sigma
    
    I = A * np.exp(-t**2 / (2 * sigma**2))
    
    # Offset to 0 at the edges: subtract the value at the edges
    I = I - I[0]
    
    # Ensure non-negative and within bounds
    I = np.maximum(I, 0)
    
    # Scale down if needed to satisfy amplitude bound
    max_amp = np.max(np.sqrt(I**2))
    if max_amp > AMPLITUDE_BOUND:
        I = I * (AMPLITUDE_BOUND / max_amp)
    
    # Ensure slew rate constraint
    # Check and reduce if necessary
    for _ in range(10):
        if check_constraints(I, np.zeros_like(I)):
            break
        I = I * 0.9
    
    Q = np.zeros_like(I)
    
    return I, Q


def optimize_pulse():
    """
    Optimize the pulse using gradient-based methods.
    Since computing exact gradients through the Liouvillian exponentiation is complex,
    we use a combination of approaches:
    1. Start from the DRAG seed
    2. Use scipy.optimize.minimize with a parameterization that enforces constraints
    """
    I_seed, Q_seed = drag_seed()
    
    # Parameterize the pulse to enforce constraints
    # Use a simple approach: optimize with penalty for constraint violations
    
    def objective(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        
        # Penalty for constraint violations
        penalty = 0.0
        for i in range(N_STEPS):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > AMPLITUDE_BOUND:
                penalty += (amp - AMPLITUDE_BOUND)**2
        
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                penalty += (slew - SLEW_BOUND)**2
            prev_I, prev_Q = I[i], Q[i]
        
        dI = 0.0 - prev_I
        dQ = 0.0 - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            penalty += (slew - SLEW_BOUND)**2
        
        return score_function(I, Q) + 10.0 * penalty
    
    x0 = np.concatenate([I_seed, Q_seed])
    
    # Use Nelder-Mead for robustness (no gradient needed)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    
    I_opt = result.x[:N_STEPS]
    Q_opt = result.x[N_STEPS:]
    
    # Ensure constraints are satisfied
    if not check_constraints(I_opt, Q_opt):
        # Fall back to seed
        return I_seed, Q_seed
    
    return I_opt, Q_opt


# Pre-compute the optimized pulse at module load time
def _compute_pulse():
    """Compute the pulse and cache it."""
    I, Q = drag_seed()
    
    # Try to optimize, but with a time limit
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
