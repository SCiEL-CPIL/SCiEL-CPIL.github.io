"""Robust DRAG pulse for 3-level transmon X-pi gate with Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Constants
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns
T1 = 25000.0  # ns
T2 = 40000.0  # ns  # Tphi
ANHARMONICITY = -2 * np.pi * 0.25
MAX_AMP = 0.4
MAX_SLEW = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)

# Precompute ladder operators for 3 levels (|0>, |1>, |2>)
A = np.array([[0, 1, 0],
              [0, 0, np.sqrt(2)],
              [0, 0, 0]], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A

def _build_hamiltonian(I, Q, detuning, scale):
    """Build 3x3 Hamiltonian for given pulse step."""
    # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    n_nm1 = N_OP @ (N_OP - np.eye(3))
    H = (detuning * N_OP
         + 0.5 * ANHARMONICITY * n_nm1
         + 0.5 * scale * (I * (A + A_DAG) + 1j * Q * (A - A_DAG)))
    return H

def _build_liouvillian(H, T1, T2):
    """Build Liouvillian superoperator for 3x3 system."""
    # Lindblad operators: L1 = a/sqrt(T1), L2 = n*sqrt(2/T2)
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / T2)
    
    dim = 3
    # Liouvillian: dρ/dt = -i[H,ρ] + Σ (LρL† - 1/2{L†L,ρ})
    # In vectorized form: L = -i(H⊗I - I⊗H†) + Σ (L_k⊗L_k† - 1/2(L_k†L_k⊗I + I⊗L_k†L_k))
    I3 = np.eye(3, dtype=complex)
    
    # Coherent part: -i(H⊗I - I⊗H†)
    L = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        L += np.kron(Lk, Lk_dag) - 0.5 * np.kron(Lk_dag @ Lk, I3) - 0.5 * np.kron(I3, Lk_dag @ Lk)
    
    return L

def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return the final density matrix for each initial state."""
    dim = 3
    rho = np.zeros((dim, dim), dtype=complex)
    
    # We'll track evolution of each basis state separately for fidelity calculation
    # Actually, let's just evolve the full density matrix for a general initial state
    # But we need fidelity for specific states. Let's evolve the unitary part and compute
    # fidelities from that, since decoherence is small.
    
    # For efficiency, let's compute the propagator U(t) = exp(L*DT) applied step by step
    # But L is 9x9 superoperator. Let's use the density matrix approach.
    
    # Actually, for fidelity calculation, we need to know the final state for each initial state.
    # Let's evolve the density matrix for each initial state separately, or better,
    # compute the process matrix.
    
    # Simpler approach: evolve the state vector for each of the 5 test states
    # But the system is open, so we need density matrices.
    
    # Let's evolve the density matrix for each initial state
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2)
    
    # For each step, compute expm(L * DT) and apply to rho
    # This is expensive but correct.
    
    # Precompute expm for each step
    propagators = []
    for i in range(N_STEPS):
        H = _build_hamiltonian(I_arr[i], Q_arr[i], detuning, scale)
        L = _build_liouvillian(H, T1, T2)
        # Vectorized: rho_vec = rho.flatten() (column-major)
        # rho_vec_next = expm(L * DT) @ rho_vec
        # But we need to be careful about the vectorization convention
        # Using column-major: vec(rho) = rho.flatten(order='F')
        # The superoperator L acts as: vec(dρ/dt) = L @ vec(ρ)
        # So vec(ρ(t+dt)) = expm(L*dt) @ vec(ρ(t))
        
        # For efficiency, we can diagonalize L or use scipy
        P = expm(L * DT)
        propagators.append(P)
    
    # Now simulate for each initial state
    # |0> = [1,0,0,0,0,0,0,0,0]
    # |1> = [0,0,0,1,0,0,0,0,0]
    # (|0>+|1>)/sqrt(2) = (1/2)[[1,1,0],[1,1,0],[0,0,0]].flatten(order='F')
    # etc.
    
    states = {
        '0': np.zeros(9, dtype=complex),
        '1': np.zeros(9, dtype=complex),
        '+': np.zeros(9, dtype=complex),
        '-': np.zeros(9, dtype=complex),
        '+i': np.zeros(9, dtype=complex),
        '-i': np.zeros(9, dtype=complex),
    }
    states['0'][0] = 1
    states['1'][3] = 1
    # |+> = (|0>+|1>)/sqrt(2)
    states['+'][0] = 0.5
    states['+'][1] = 0.5
    states['+'][3] = 0.5
    states['+'][4] = 0.5
    # |-> = (|0>-|1>)/sqrt(2)
    states['-'][0] = 0.5
    states['-'][1] = -0.5
    states['-'][3] = -0.5
    states['-'][4] = 0.5
    # |+i> = (|0>+i|1>)/sqrt(2)
    states['+i'][0] = 0.5
    states['+i'][1] = 0.5j
    states['+i'][3] = -0.5j
    states['+i'][4] = 0.5
    # |-i> = (|0>-i|1>)/sqrt(2)
    states['-i'][0] = 0.5
    states['-i'][1] = -0.5j
    states['-i'][3] = 0.5j
    states['-i'][4] = 0.5
    
    # Target states for X gate:
    # X|0> = |1>, X|1> = |0>, X|+> = |+>, X|-> = |->, X|+i> = -i|+i> (up to phase)
    # Actually, X|+i> = (|1> - i|0>)/sqrt(2) = -i(|0> + i|1>)/sqrt(2) = -i|+i>
    # So the target is up to a global phase.
    
    # For fidelity, we compute F = |<psi_target|psi_final>|^2 for pure states
    # But with decoherence, we have mixed states. Use F = Tr(rho_target * rho_final)
    
    # Target density matrices
    targets = {
        '0': np.zeros(9, dtype=complex),  # X|0> = |1>
        '1': np.zeros(9, dtype=complex),  # X|1> = |0>
        '+': np.zeros(9, dtype=complex),  # X|+> = |+>
        '-': np.zeros(9, dtype=complex),  # X|-> = |->
        '+i': np.zeros(9, dtype=complex), # X|+i> = -i|+i>
        '-i': np.zeros(9, dtype=complex), # X|-i> = i|-i>
    }
    # X|0> = |1>
    targets['0'][3] = 1
    # X|1> = |0>
    targets['1'][0] = 1
    # X|+> = |+>
    targets['+'][0] = 0.5
    targets['+'][1] = 0.5
    targets['+'][3] = 0.5
    targets['+'][4] = 0.5
    # X|-> = |->
    targets['-'][0] = 0.5
    targets['-'][1] = -0.5
    targets['-'][3] = -0.5
    targets['-'][4] = 0.5
    # X|+i> = -i|+i>
    targets['+i'][0] = 0.5
    targets['+i'][1] = -0.5j
    targets['+i'][3] = 0.5j
    targets['+i'][4] = 0.5
    # X|-i> = i|-i>
    targets['-i'][0] = 0.5
    targets['-i'][1] = 0.5j
    targets['-i'][3] = -0.5j
    targets['-i'][4] = 0.5
    
    # Simulate each state
    fidelities = []
    for key in ['0', '1', '+', '-', '+i', '-i']:
        rho_vec = states[key].copy()
        for i in range(N_STEPS):
            rho_vec = propagators[i] @ rho_vec
        # Convert back to matrix
        rho_final = rho_vec.reshape((3, 3), order='F')
        rho_target = targets[key].reshape((3, 3), order='F')
        # Fidelity for pure target and possibly mixed final:
        # F = Tr(rho_target * rho_final)
        # But this can be complex. Use F = |Tr(sqrt(rho_target) rho_final sqrt(rho_target))|^2
        # For pure target, F = <psi_target|rho_final|psi_target>
        # Let's extract the target state vector
        if key == '0':
            psi_t = np.array([0, 1, 0], dtype=complex)
        elif key == '1':
            psi_t = np.array([1, 0, 0], dtype=complex)
        elif key == '+':
            psi_t = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        elif key == '-':
            psi_t = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        elif key == '+i':
            psi_t = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
        else:  # '-i'
            psi_t = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        
        F = np.abs(np.vdot(psi_t, rho_final @ psi_t)) ** 2
        # Clamp to [0,1]
        F = np.clip(F, 0, 1)
        fidelities.append(F)
    
    return fidelities

def _check_validity(I_arr, Q_arr):
    """Check amplitude and slew constraints."""
    # Amplitude constraint
    amp = np.sqrt(I_arr**2 + Q_arr**2)
    if np.any(amp > MAX_AMP + 1e-10):
        return False
    
    # Slew constraint
    # From zero into step 1
    slew0 = np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / DT
    if slew0 > MAX_SLEW + 1e-10:
        return False
    
    # Between steps
    dI = np.diff(I_arr)
    dQ = np.diff(Q_arr)
    slews = np.sqrt(dI**2 + dQ**2) / DT
    if np.any(slews > MAX_SLEW + 1e-10):
        return False
    
    # From last step to zero
    slew_end = np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / DT
    if slew_end > MAX_SLEW + 1e-10:
        return False
    
    return True

def _score_pulse(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    if not _check_validity(I_arr, Q_arr):
        return 1e10
    
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    all_fidelities = []
    for det in detunings:
        for sc in scales:
            fids = _simulate_pulse(I_arr, Q_arr, det, sc)
            mean_f = np.mean(fids)
            all_fidelities.append(mean_f)
    
    mean_score = np.mean(all_fidelities)
    max_score = np.max(all_fidelities)
    # Infidelity = 1 - fidelity
    score = (1 - mean_score) + (1 - max_score)
    return score

def _generate_drag_pulse(params):
    """Generate DRAG pulse from parameters.
    params = [sigma, beta, alpha, phase]
    sigma: Gaussian width
    beta: DRAG beta parameter
    alpha: amplitude scaling
    phase: phase offset
    """
    sigma = params[0]
    beta = params[1]
    alpha = params[2]
    phase = params[3]
    
    t = np.arange(N_STEPS) * DT
    t_center = DURATION / 2
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # DRAG: add derivative term
    # d/dt gauss = -((t - t_center) / sigma^2) * gauss
    drag_term = beta * ((t - t_center) / (sigma ** 2)) * gauss
    
    # Amplitude
    I = alpha * (gauss + drag_term) * np.cos(phase)
    Q = alpha * (gauss + drag_term) * np.sin(phase)
    
    # Ensure area is approximately pi
    # Area = integral of amplitude * dt
    area = np.sum(np.sqrt(I**2 + Q**2)) * DT
    if area > 0:
        scale_factor = np.pi / area
        I *= scale_factor
        Q *= scale_factor
    
    # Clip to valid range
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > MAX_AMP
    if np.any(mask):
        I[mask] *= MAX_AMP / amp[mask]
        Q[mask] *= MAX_AMP / amp[mask]
    
    return I, Q

def pulse(target_gate, hardware_model):
    """Generate a robust X-pi gate pulse."""
    start_time = time.time()
    deadline = start_time + 320.0  # Leave 10s margin
    
    # Initial DRAG pulse
    I0, Q0 = _generate_drag_pulse([40.0 / 6.0, 0.0, 1.0, 0.0])
    
    # Check initial validity
    if not _check_validity(I0, Q0):
        # Scale down
        amp = np.sqrt(I0**2 + Q0**2)
        max_amp = np.max(amp) if np.max(amp) > 0 else 1
        if max_amp > MAX_AMP:
            I0 *= MAX_AMP / max_amp
            Q0 *= MAX_AMP / max_amp
    
    best_I = I0.copy()
    best_Q = Q0.copy()
    best_score = _score_pulse(best_I, best_Q)
    
    # Nelder-Mead optimization
    x0 = np.array([40.0 / 6.0, 0.0, 1.0, 0.0])
    
    def objective(params):
        I, Q = _generate_drag_pulse(params)
        if not _check_validity(I, Q):
            return 1e10
        return _score_pulse(I, Q)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            I_opt, Q_opt = _generate_drag_pulse(result.x)
            if _check_validity(I_opt, Q_opt):
                score_opt = _score_pulse(I_opt, Q_opt)
                if score_opt < best_score:
                    best_I = I_opt.copy()
                    best_Q = Q_opt.copy()
                    best_score = score_opt
    except Exception:
        pass
    
    # Return the best pulse found
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
