"""
Robust DRAG pulse for 3-level transmon X-pi gate.
Uses low-dimensional parameterization (6 params) with Differential Evolution
to find optimal pulse shape that minimizes infidelity across 9 detuning/scale cases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.linalg import expm

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
_states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0]
], dtype=complex)

def _build_hamiltonian(I, Q, detuning, scale):
    """Build 3x3 Hamiltonian."""
    H = (detuning * _n 
         + 0.5 * ANHARMONICITY * _nn1
         + 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag)))
    return H

def _make_pulse(params):
    """Generate pulse from parameters: [sigma, beta, amp_scale, phase_shift, drag_shift, center_shift]"""
    sigma, beta, amp_scale, phase_shift, drag_shift, center_shift = params
    t_mid = np.linspace(DT/2, TOTAL_TIME - DT/2, N_STEPS)
    center = TOTAL_TIME / 2.0 + center_shift
    
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
    
    return I_pulse, Q_pulse

def _evolve_pulse_params(params, detuning, scale):
    """Evolve the system with a pulse defined by parameters."""
    I_pulse, Q_pulse = _make_pulse(params)
    
    # Evolve each of the 6 initial states using state vector evolution
    fidelities = []
    for state in _states:
        psi = state.copy()
        for i in range(N_STEPS):
            H = _build_hamiltonian(I_pulse[i], Q_pulse[i], detuning, scale)
            U = expm(-1j * H * DT)
            psi = U @ psi
        # Target: X gate on |0>,|1> subspace
        target = np.array([state[1], state[0], 0], dtype=complex)
        fid = np.abs(np.vdot(target, psi))**2
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def _score_params(params):
    """Score a set of parameters across all 9 cases."""
    sigma, beta, amp_scale, phase_shift, drag_shift, center_shift = params
    
    # Quick validity check
    if sigma <= 0 or sigma > 20:
        return 1.0
    if abs(amp_scale) > 1.5:
        return 1.0
    if abs(beta) > 1.0:
        return 1.0
    if abs(center_shift) > 10:
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
    I_pulse, Q_pulse = _make_pulse(params)
    return np.column_stack([I_pulse, Q_pulse])

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I,Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep safety margin
    
    # Define parameter bounds
    # sigma: Gaussian width in ns
    # beta: DRAG strength
    # amp_scale: amplitude scaling (area = pi * amp_scale)
    # phase_shift: overall phase
    # drag_shift: phase for DRAG component
    # center_shift: shift pulse center from T/2
    bounds = [
        (3.0, 15.0),      # sigma
        (-0.5, 0.5),      # beta
        (0.5, 1.5),       # amp_scale
        (-0.5, 0.5),      # phase_shift
        (-0.5, 0.5),      # drag_shift
        (-5.0, 5.0),      # center_shift
    ]
    
    # Initial guess based on seed
    x0 = np.array([40.0/6.0, 0.1, 1.0, 0.0, 0.0, 0.0])
    
    # Try Differential Evolution with limited budget
    best_params = x0.copy()
    best_score = _score_params(x0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 80.0:
        # Run DE with limited iterations
        try:
            result = differential_evolution(
                _score_params,
                bounds,
                seed=42,
                maxiter=40,
                popsize=18,
                tol=1e-5,
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
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7},
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
