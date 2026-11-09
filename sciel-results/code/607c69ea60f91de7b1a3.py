"""
Robust DRAG pulse for 3-level transmon X-pi gate.
Uses low-dimensional parameterization (8 params) with Differential Evolution
and Nelder-Mead refinement to find optimal pulse shape that minimizes infidelity
across 9 detuning/scale cases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
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
_nn1 = _n2 - _n  # n(n-1)
_a_plus_adag = _a + _a_dag
_a_minus_adag = _a - _a_dag

# Time samples
t_samples = (np.arange(N_STEPS) + 0.5) * DT
t_center = TOTAL_TIME / 2.0

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

# Target X gate maps |0><->|1>, |2>->|2>
_X_target = np.diag([0, 1, 1]).astype(complex)
# Actually X swaps 0 and 1, leaves 2
_X_target = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)

def _build_hamiltonian(I, Q, detuning, scale):
    """Build 3x3 Hamiltonian."""
    H = (detuning * _n 
         + 0.5 * ANHARMONICITY * _nn1
         + 0.5 * scale * (I * _a_plus_adag + 1j * Q * _a_minus_adag))
    return H

def _make_pulse(params):
    """Generate pulse from 8 parameters:
    [sigma, beta, amp_scale, phase_shift, drag_phase, center_shift, shape_param, tail_width]
    """
    sigma, beta, amp_scale, phase_shift, drag_phase, center_shift, shape_param, tail_width = params
    
    # Center of pulse
    center = t_center + center_shift
    
    # Gaussian envelope
    x = (t_samples - center) / sigma
    gaussian = np.exp(-x**2 / 2.0)
    
    # Shape modification: add a slight asymmetry or flat-top
    # shape_param in [-1, 1]: positive = wider center, negative = narrower
    if abs(shape_param) > 1e-6:
        # Use a power-law modification
        shape_mod = 1.0 + shape_param * np.exp(-x**2 / 4.0)
        gaussian = gaussian * shape_mod
    
    # Normalize area to pi * amp_scale
    # Area of gaussian with sigma: sigma * sqrt(2*pi)
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    envelope = A * gaussian
    
    # DRAG component: proportional to derivative of envelope
    # d/dt(envelope) ~ -beta * (t - center)/sigma^2 * gaussian
    drag_envelope = -beta * (t_samples - center) / sigma**2 * gaussian
    
    # I and Q components
    I_pulse = envelope * np.cos(phase_shift)
    Q_pulse = envelope * np.sin(phase_shift) + drag_envelope * np.cos(drag_phase)
    
    # Apply amplitude bound
    amp = np.sqrt(I_pulse**2 + Q_pulse**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I_pulse[mask] *= scale_factor
        Q_pulse[mask] *= scale_factor
    
    # Fix slew rate violations (iterative)
    for _ in range(3):
        # From zero to first
        slew0 = np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / DT
        if slew0 > SLEW_BOUND:
            factor = SLEW_BOUND * DT / np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2)
            I_pulse[0] *= factor
            Q_pulse[0] *= factor
        
        # Between steps
        for i in range(N_STEPS - 1):
            dI = I_pulse[i+1] - I_pulse[i]
            dQ = Q_pulse[i+1] - Q_pulse[i]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
                I_pulse[i+1] = I_pulse[i] + dI * factor
                Q_pulse[i+1] = Q_pulse[i] + dQ * factor
        
        # From last to zero
        slewn = np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / DT
        if slewn > SLEW_BOUND:
            factor = SLEW_BOUND * DT / np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2)
            I_pulse[-1] *= factor
            Q_pulse[-1] *= factor
    
    return I_pulse, Q_pulse

def _simulate_pulse(I_pulse, Q_pulse, detuning, scale):
    """Simulate pulse evolution and return infidelity."""
    # Build all unitaries
    U_list = []
    for i in range(N_STEPS):
        H = _build_hamiltonian(I_pulse[i], Q_pulse[i], detuning, scale)
        U = expm(-1j * H * DT)
        U_list.append(U)
    
    # Evolve all 6 states simultaneously
    psi_all = _states.copy()  # (6, 3)
    for i in range(N_STEPS):
        U = U_list[i]
        psi_all = (U @ psi_all.T).T
    
    # Compute fidelity for each state
    # Target: X gate
    fid_sum = 0.0
    for k in range(6):
        psi = psi_all[k]
        target = _X_target @ _states[k]
        fid = np.abs(np.vdot(target, psi))**2
        fid_sum += fid
    
    mean_fid = fid_sum / 6.0
    infidelity = 1.0 - mean_fid
    return infidelity

def _score_params(params):
    """Score a set of parameters across all 9 cases."""
    sigma, beta, amp_scale, phase_shift, drag_phase, center_shift, shape_param, tail_width = params
    
    # Quick validity check
    if sigma <= 0.5 or sigma > 25:
        return 1.0
    if abs(amp_scale) > 2.0:
        return 1.0
    if abs(beta) > 2.0:
        return 1.0
    if abs(center_shift) > 15:
        return 1.0
    if abs(shape_param) > 1.0:
        return 1.0
    
    I_pulse, Q_pulse = _make_pulse(params)
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _simulate_pulse(I_pulse, Q_pulse, det, sc)
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
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    # Define parameter bounds
    bounds = [
        (2.0, 20.0),      # sigma (Gaussian width)
        (-1.0, 1.0),      # beta (DRAG strength)
        (0.5, 2.0),       # amp_scale (area scaling)
        (-1.0, 1.0),      # phase_shift
        (-1.0, 1.0),      # drag_phase
        (-10.0, 10.0),    # center_shift
        (-0.5, 0.5),      # shape_param
        (0.5, 5.0),       # tail_width (unused but kept for interface)
    ]
    
    # Initial guess based on seed (sigma=40/6≈6.67)
    x0 = np.array([40.0/6.0, 0.15, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    
    best_params = x0.copy()
    best_score = _score_params(x0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 100.0:
        # Run Differential Evolution
        try:
            result = differential_evolution(
                _score_params,
                bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='latinhypercube',
                update_interval=1,
            )
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Refine with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 40.0:
        try:
            res = minimize(
                _score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8},
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Second refinement with tighter tolerances if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        try:
            res = minimize(
                _score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9},
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
    for _ in range(3):
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
