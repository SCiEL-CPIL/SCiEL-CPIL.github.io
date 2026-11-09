"""
Robust DRAG pulse for 3-level transmon X-pi gate.
Uses low-dimensional parameterization (5 params) with Differential Evolution
to find optimal pulse shape that minimizes infidelity across 9 detuning/scale cases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Constants from task contract
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2.0 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2.0 * np.pi * 0.002
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_a_dag = _a.conj().T
_n = _a_dag @ _a
_n2 = _n @ _n
# n(n-1) = n^2 - n
_nn1 = _n2 - _n

# Detuning and scale values for the 9 cases
detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
scales = [0.97, 1.0, 1.03]

# Initial states for fidelity calculation
# |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
# We need 5 states (the task says |0>, |1>, (|0>+-|1>)/sqrt(2), (|0>+-i|1>)/sqrt(2))
# That's 1 + 1 + 2 + 2 = 6 states? Let me re-read: "over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
# So: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2) = 6 states
_states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0]
], dtype=complex)

# Target: X gate on |0> and |1> subspace
# X|0> = |1>, X|1> = |0>
# For the 2-level subspace, X = [[0,1],[1,0]]
# In 3-level, we consider the evolution projected to |0>,|1> subspace
# Fidelity = |<psi_target|psi_final>|^2 where target = X|psi_initial>

def _build_hamiltonian(I, Q, detuning, scale):
    """Build 3x3 Hamiltonian."""
    H = (detuning * _n 
         + 0.5 * ANHARMONICITY * _nn1
         + 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag)))
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build Liouvillian superoperator for density matrix evolution.
    
    d rho/dt = -i[H, rho] - 0.5 {L, L^dagger rho - L rho L^dagger} for each L
    
    Lindblad operators: a/sqrt(T1) and n*sqrt(2/Tphi)
    """
    n_dim = 3
    L1 = _a / np.sqrt(T1)
    L2 = _n * np.sqrt(2.0 / Tphi)
    
    # Build superoperator in vec form
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    # d rho/dt = -i H rho + i rho H - 0.5 (L rho L^dagger + L^dagger L rho - 2 L^dagger rho L)
    # Actually: d rho/dt = -i[H,rho] - sum_k 0.5(L_k rho L_k^dagger + L_k^dagger L_k rho - L_k^dagger rho L_k - rho L_k^dagger L_k)
    # = -i[H,rho] - sum_k 0.5({L_k^dagger L_k, rho} - 2 L_k^dagger rho L_k)
    
    L1_dag = L1.conj().T
    L2_dag = L2.conj().T
    
    # Superoperator: L = -i(H ⊗ I - I ⊗ H^T) - sum 0.5(L_k^T ⊗ L_k^dagger + L_k^dagger^T ⊗ L_k) + ... 
    # Let me use the standard form:
    # vec(d rho/dt) = -i (H ⊗ I - I ⊗ H^T) vec(rho) - sum_k 0.5 [ (L_k^T ⊗ L_k^dagger) vec(L_k^dagger rho L_k) ... ]
    
    # Simpler: use the commutator and anticommutator forms
    # -i[H,rho]: superoperator is -i(H⊗I - I⊗H^T)
    # For Lindblad term -0.5{L^dagger L, rho} + L^dagger rho L:
    #   -0.5(L^dagger L ⊗ I + I ⊗ (L^dagger L)^T) vec(rho) + (L^T ⊗ L^dagger) vec(rho)
    
    H_T = H.T
    L1L1dag = L1_dag @ L1
    L2L2dag = L2_dag @ L2
    
    S = -1j * (np.kron(H, np.eye(n_dim)) - np.kron(np.eye(n_dim), H_T))
    S -= 0.5 * (np.kron(L1L1dag, np.eye(n_dim)) + np.kron(np.eye(n_dim), L1L1dag.T))
    S += np.kron(L1.T, L1_dag)
    S -= 0.5 * (np.kron(L2L2dag, np.eye(n_dim)) + np.kron(np.eye(n_dim), L2L2dag.T))
    S += np.kron(L2.T, L2_dag)
    
    return S

def _evolve_pulse_params(params, detuning, scale):
    """Evolve the system with a pulse defined by parameters.
    
    params: [sigma, beta, amp_scale, phase_shift, drag_shift]
    Returns the final density matrix (as a 9-element vector) or the unitary-like evolution.
    """
    sigma, beta, amp_scale, phase_shift, drag_shift = params
    
    # Generate pulse
    t = np.linspace(0, TOTAL_TIME, N_STEPS + 1)  # 81 points including start
    t_mid = (t[:-1] + t[1:]) / 2.0  # 80 midpoints
    
    # Gaussian envelope
    # Center at T/2
    center = TOTAL_TIME / 2.0
    # sigma in ns, area = pi * amp_scale
    # Gaussian: A * exp(-(t-center)^2 / (2*sigma^2))
    # Area = A * sigma * sqrt(2*pi) = pi * amp_scale
    # So A = pi * amp_scale / (sigma * sqrt(2*pi))
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    
    envelope = A * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    
    # DRAG correction: proportional to derivative of envelope
    # d/dt [exp(-(t-center)^2/(2 sigma^2))] = -(t-center)/sigma^2 * exp(...)
    drag_envelope = -beta * (t_mid - center) / sigma**2 * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    
    # Phase shift and drag shift
    phase = phase_shift * np.ones_like(t_mid)
    q_phase = drag_shift * np.ones_like(t_mid)
    
    I_pulse = envelope * np.cos(phase)
    Q_pulse = envelope * np.sin(phase) + drag_envelope * np.cos(q_phase)
    
    # Apply amplitude bound
    amp = np.sqrt(I_pulse**2 + Q_pulse**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I_pulse[mask] *= scale_factor
        Q_pulse[mask] *= scale_factor
    
    # Check slew rate and clip if needed
    # ||p[i+1]-p[i]||/dt <= 0.12
    # We'll handle this by reducing amplitude if slew is violated
    for i in range(N_STEPS - 1):
        dI = I_pulse[i+1] - I_pulse[i]
        dQ = Q_pulse[i+1] - Q_pulse[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            # Reduce both steps
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I_pulse[i+1] = I_pulse[i] + dI * factor
            Q_pulse[i+1] = Q_pulse[i] + dQ * factor
    
    # Check from zero to first step
    slew0 = np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / DT
    if slew0 > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2)
        I_pulse[0] *= factor
        Q_pulse[0] *= factor
    
    # Check from last step to zero
    slewn = np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / DT
    if slewn > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2)
        I_pulse[-1] *= factor
        Q_pulse[-1] *= factor
    
    # Evolve using piecewise constant Hamiltonian
    # Use density matrix approach for proper decoherence
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    # For efficiency, we'll evolve each initial state separately using state vector
    # since the dynamics are linear
    # Actually, let's use the Liouvillian approach but it's expensive.
    # Alternative: evolve state vectors directly (unitary part) and approximate decoherence
    
    # For speed, let's use state vector evolution with the Hamiltonian only
    # and add decoherence effects approximately
    # Actually, T1 and Tphi are large (25us, 40us) compared to 40ns pulse
    # So decoherence is small. Let's first try unitary evolution.
    
    # Evolve each of the 6 initial states
    fidelities = []
    for state in _states:
        psi = state.copy()
        for i in range(N_STEPS):
            H = _build_hamiltonian(I_pulse[i], Q_pulse[i], detuning, scale)
            # Evolve for dt
            # Use expm for small dt
            U = expm(-1j * H * DT)
            psi = U @ psi
        # Target: X gate
        # X|0> = |1>, X|1> = |0>
        # For states in span{|0>,|1>}:
        # |0> -> |1>
        # |1> -> |0>
        # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
        # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
        # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2) ... 
        # Let me compute targets properly
        # X matrix in 2D: [[0,1],[1,0]]
        # target = X @ initial (in 2D subspace)
        # In 3D, X acts as: |0><0| + |1><1| -> |1><0| + |0><1|
        # So target state for initial |psi> is: (psi[1], psi[0], 0)
        target = np.array([state[1], state[0], 0], dtype=complex)
        # Fidelity = |<target|psi>|^2
        fid = np.abs(np.vdot(target, psi))**2
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def _score_params(params):
    """Score a set of parameters across all 9 cases."""
    sigma, beta, amp_scale, phase_shift, drag_shift = params
    
    # Quick validity check
    if sigma <= 0 or sigma > 20:
        return 1.0
    if abs(amp_scale) > 1.5:
        return 1.0
    if abs(beta) > 1.0:
        return 1.0
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _evolve_pulse_params(params, det, sc)
            infidelities.append(inf)
    
    inf_arr = np.array(infidelities)
    score = np.mean(inf_arr) + np.max(inf_arr)
    return score

def _generate_pulse(params):
    """Generate the final (80, 2) pulse array from parameters."""
    sigma, beta, amp_scale, phase_shift, drag_shift = params
    
    t_mid = np.linspace(DT/2, TOTAL_TIME - DT/2, N_STEPS)
    center = TOTAL_TIME / 2.0
    
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    envelope = A * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    drag_envelope = -beta * (t_mid - center) / sigma**2 * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    
    I_pulse = envelope * np.cos(phase_shift)
    Q_pulse = envelope * np.sin(phase_shift) + drag_envelope * np.cos(drag_shift)
    
    # Apply amplitude bound
    amp = np.sqrt(I_pulse**2 + Q_pulse**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I_pulse[mask] *= scale_factor
        Q_pulse[mask] *= scale_factor
    
    # Fix slew rate violations
    for i in range(N_STEPS - 1):
        dI = I_pulse[i+1] - I_pulse[i]
        dQ = Q_pulse[i+1] - Q_pulse[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I_pulse[i+1] = I_pulse[i] + dI * factor
            Q_pulse[i+1] = Q_pulse[i] + dQ * factor
    
    # From zero to first
    slew0 = np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / DT
    if slew0 > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2)
        I_pulse[0] *= factor
        Q_pulse[0] *= factor
    
    # From last to zero
    slewn = np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / DT
    if slewn > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2)
        I_pulse[-1] *= factor
        Q_pulse[-1] *= factor
    
    return np.column_stack([I_pulse, Q_pulse])

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I,Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, keep safety margin
    
    # Define parameter bounds
    # sigma: Gaussian width in ns (40/6 ≈ 6.67 ns for seed)
    # beta: DRAG strength
    # amp_scale: amplitude scaling (area = pi * amp_scale)
    # phase_shift: overall phase
    # drag_shift: phase for DRAG component
    bounds = [
        (3.0, 15.0),      # sigma
        (-0.5, 0.5),      # beta
        (0.5, 1.5),       # amp_scale
        (-0.5, 0.5),      # phase_shift
        (-0.5, 0.5),      # drag_shift
    ]
    
    # Initial guess based on seed: sigma = 40/6, area = pi so amp_scale = 1
    x0 = np.array([40.0/6.0, 0.1, 1.0, 0.0, 0.0])
    
    # Try Differential Evolution with limited budget
    best_params = x0.copy()
    best_score = _score_params(x0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0:
        # Run DE with limited iterations
        try:
            result = differential_evolution(
                _score_params,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='latinhypercube',
                update_interval=1,
            )
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Refine with local optimization if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        try:
            from scipy.optimize import minimize
            res = minimize(
                _score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6},
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Generate final pulse
    pulse_array = _generate_pulse(best_params)
    
    # Final validity check
    amp = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-9):
        # Clip
        mask = amp > AMPLITUDE_BOUND
        factor = AMPLITUDE_BOUND / amp[mask]
        pulse_array[mask, 0] *= factor
        pulse_array[mask, 1] *= factor
    
    # Slew rate check
    for i in range(len(pulse_array) - 1):
        dI = pulse_array[i+1, 0] - pulse_array[i, 0]
        dQ = pulse_array[i+1, 1] - pulse_array[i, 1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            pulse_array[i+1, 0] = pulse_array[i, 0] + dI * factor
            pulse_array[i+1, 1] = pulse_array[i, 1] + dQ * factor
    
    # From zero to first
    slew0 = np.sqrt(pulse_array[0, 0]**2 + pulse_array[0, 1]**2) / DT
    if slew0 > SLEW_BOUND + 1e-9:
        factor = SLEW_BOUND * DT / np.sqrt(pulse_array[0, 0]**2 + pulse_array[0, 1]**2)
        pulse_array[0, 0] *= factor
        pulse_array[0, 1] *= factor
    
    # From last to zero
    slewn = np.sqrt(pulse_array[-1, 0]**2 + pulse_array[-1, 1]**2) / DT
    if slewn > SLEW_BOUND + 1e-9:
        factor = SLEW_BOUND * DT / np.sqrt(pulse_array[-1, 0]**2 + pulse_array[-1, 1]**2)
        pulse_array[-1, 0] *= factor
        pulse_array[-1, 1] *= factor
    
    return pulse_array
# EVOLVE-BLOCK-END
