"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0  # Tphi

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Build ladder operators for 3-level system
def build_operators():
    """Build a, a_dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

def hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    drive_term = 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    H += drive_term
    return H

def liouvillian_step(rho, H, dt):
    """Apply one step of the Liouvillian evolution using matrix exponential."""
    # Coherent evolution
    U = _matrix_exponential(-1j * H * dt)
    rho = U @ rho @ U.conj().T
    
    # Incoherent evolution (T1 and Tphi)
    # T1: decay from |1> to |0> and |2> to |1>
    gamma1 = 1.0 / T1
    gamma_phi = 1.0 / T2
    
    # Lindblad operators
    # L1 = a / sqrt(T1)
    L1 = a_op / np.sqrt(T1)
    # L2 = n * sqrt(2/Tphi)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    # Apply Lindblad terms approximately for small dt
    # rho = rho - 0.5*dt*(L†L rho + rho L†L) + dt*L rho L†
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        rho = rho - 0.5 * dt * (LdagL @ rho + rho @ LdagL)
        rho = rho + dt * L @ rho @ L.conj().T
    
    return rho

def _matrix_exponential(A):
    """Compute matrix exponential using scipy."""
    from scipy.linalg import expm
    return expm(A)

def fidelity_with_X(rho):
    """Compute mean fidelity to X gate over the 5 test states."""
    # Target: X gate swaps |0> and |1>
    # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi in states:
        # Target state after X
        psi_target = np.zeros(3, dtype=complex)
        psi_target[0] = psi[1]
        psi_target[1] = psi[0]
        psi_target[2] = psi[2]
        
        # Fidelity = |<psi_target|psi_final>|^2
        psi_final = _apply_to_state(rho, psi)
        overlap = np.vdot(psi_target, psi_final)
        fid = np.abs(overlap) ** 2
        fidelities.append(fid)
    
    return np.mean(fidelities)

def _apply_to_state(rho, psi):
    """Apply density matrix to state vector."""
    return rho @ psi

def evolve_pulse(pulse, detuning, scale):
    """Evolve the system with the given pulse and return final density matrix."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        H = hamiltonian(detuning, scale, I[i], Q[i])
        rho = liouvillian_step(rho, H, DT)
    
    return rho

def compute_score(pulse):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            rho = evolve_pulse(pulse, det, sc)
            fid = fidelity_with_X(rho)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def check_validity(pulse):
    """Check if the pulse satisfies amplitude and slew constraints."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Amplitude bound
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    # Slew bound
    # From zero to step 1
    slew0 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_last > SLEW_BOUND + 1e-10:
        return False
    
    return True

def generate_drag_pulse(alpha, beta, gamma):
    """Generate a DRAG-style pulse with Gaussian envelope.
    
    Parameters:
    - alpha: overall amplitude scale
    - beta: Gaussian width parameter (sigma)
    - gamma: DRAG quadrature component strength
    """
    t = np.arange(N_STEPS) * DT
    t_center = TOTAL_TIME / 2.0
    
    # Gaussian envelope
    sigma = beta
    env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to ensure area = pi
    # Area of Gaussian: integral of env dt = sigma * sqrt(2*pi) * (1/DT) approximately
    # Actually, discrete sum: sum(env) * DT
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    
    # I component: alpha * env
    I = alpha * env
    
    # Q component: DRAG term, proportional to derivative of envelope
    denv_dt = np.gradient(env, DT)
    Q = gamma * denv_dt
    
    # Clip to bounds
    pulse = np.column_stack([I, Q])
    
    # Ensure amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    return pulse

def optimize_pulse():
    """Optimize the pulse parameters."""
    def objective(params):
        alpha, beta, gamma = params
        pulse = generate_drag_pulse(alpha, beta, gamma)
        if not check_validity(pulse):
            return 1e6
        return compute_score(pulse)
    
    # Initial guess
    x0 = [0.3, 5.0, 0.05]
    
    # Bounds
    bounds = [(0.01, AMPLITUDE_BOUND), (2.0, 10.0), (-0.2, 0.2)]
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      bounds=bounds, 
                      options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
    
    return generate_drag_pulse(*result.x)

# Generate the optimized pulse at module load time
_optimized_pulse = optimize_pulse()

def pulse(target_gate, hardware_model):
    """Return the optimized pulse for the given target gate and hardware model."""
    global _optimized_pulse
    return _optimized_pulse.copy()
# EVOLVE-BLOCK-END
