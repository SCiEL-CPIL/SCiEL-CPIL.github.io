"""Robust control pulses for a three-level transmon."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

_DEADLINE = 320.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

_I3 = np.eye(3, dtype=complex)

# Dissipative part is constant
_L_DISS = (np.kron(_L1, _L1.conj()) - 0.5*(np.kron(_I3, _L1ADAG @ _L1) + np.kron(_L1ADAG @ _L1, _I3))
          + np.kron(_L2, _L2.conj()) - 0.5*(np.kron(_I3, _L2ADAG @ _L2) + np.kron(_L2ADAG @ _L2, _I3)))

# Precompute the static (detuning + anharmonicity) part for each detuning value
_STATIC_LIOUV = {}
for det in _DET_GRID:
    H_static = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - _I3))
    L_static = -1j * (np.kron(H_static, _I3) - np.kron(_I3, H_static.T))
    _STATIC_LIOUV[det] = L_static + _L_DISS

# Precompute drive-related kron products
_A_PLUS_ADAG = _A + _ADAG
_A_MINUS_ADAG = _A - _ADAG

_KRON_AP_I = np.kron(_A_PLUS_ADAG, _I3)
_KRON_I_APT = np.kron(_I3, _A_PLUS_ADAG.T)
_KRON_AM_I = np.kron(_A_MINUS_ADAG, _I3)
_KRON_I_AMT = np.kron(_I3, _A_MINUS_ADAG.T)

def _liouvillian_step(dt, I, Q, detuning, scale):
    """Compute the Liouvillian superoperator for one step."""
    L_total = _STATIC_LIOUV[detuning]
    
    if abs(I) > 1e-15 or abs(Q) > 1e-15:
        coeff_I = 0.5 * scale * I
        coeff_Q = 0.5 * scale * Q
        L_total = L_total + (-1j * coeff_I * _KRON_AP_I + 1j * coeff_I * _KRON_I_APT 
                            + coeff_Q * _KRON_AM_I - coeff_Q * _KRON_I_AMT)
    
    # Propagator: expm(L_total * dt)
    return expm(L_total * dt)

def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return final density matrices for each initial basis state."""
    N = len(I_arr)
    
    # Precompute all propagators for this detuning/scale
    props = []
    for i in range(N):
        prop = _liouvillian_step(_DT, I_arr[i], Q_arr[i], detuning, scale)
        props.append(prop)
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    results = []
    for state in states:
        rho = np.outer(state, state.conj())
        vec_rho = rho.flatten()
        
        for prop in props:
            vec_rho = prop @ vec_rho
        
        rho_final = vec_rho.reshape(3, 3)
        results.append(rho_final)
    
    return results

def _compute_fidelity(rho_final, target_state):
    """Compute fidelity between density matrix and target pure state."""
    target_rho = np.outer(target_state, target_state.conj())
    return np.real(np.trace(target_rho @ rho_final))

def _score_pulse(I_arr, Q_arr):
    """Compute the score for a given pulse (lower is better)."""
    # Target states for X gate
    target_0 = np.array([0, 1, 0], dtype=complex)  # X|0> = |1>
    target_1 = np.array([1, 0, 0], dtype=complex)  # X|1> = |0>
    
    all_fidelities = []
    
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            results = _simulate_pulse(I_arr, Q_arr, det, scale)
            
            targets = [
                target_1,  # for |0>
                target_0,  # for |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+|1>)/sqrt(2)
                (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-|1>)/sqrt(2)
                (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+i|1>)/sqrt(2)
                (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-i|1>)/sqrt(2)
            ]
            
            for i, target in enumerate(targets):
                fid = _compute_fidelity(results[i], target)
                all_fidelities.append(fid)
    
    mean_fid = np.mean(all_fidelities)
    max_fid = np.max(all_fidelities)
    infidelity = 1 - mean_fid + (1 - max_fid)
    
    return infidelity

def _make_drag_pulse(beta, sigma_factor=1.0, offset_q=0.0, area_scale=1.0):
    """Create a DRAG pulse with given beta parameter."""
    t = np.arange(_N) * _DT
    center = _N * _DT / 2.0
    sigma = (40.0/6.0) * sigma_factor
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset so edges are zero
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    env = env - edge_val
    
    # Scale to get area = pi * area_scale
    current_area = np.sum(env) * _DT
    if current_area > 1e-10:
        env = env * (np.pi * area_scale / current_area)
    
    I = env.copy()
    
    # Add DRAG term (derivative of envelope)
    denv = np.gradient(env, _DT)
    Q = beta * denv + offset_q * env
    
    # Enforce constraints
    I, Q = _enforce_constraints(I, Q, _AMP, _SLEW, _DT)
    
    return I, Q

def _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Enforce amplitude and slew rate constraints."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    
    for _ in range(100):
        # Enforce amplitude
        max_amp = 0
        for i in range(N):
            amp = np.sqrt(I_arr[i]**2 + Q_arr[i]**2)
            if amp > amp_bound:
                scale = amp_bound / amp
                I_arr[i] *= scale
                Q_arr[i] *= scale
                max_amp = amp
            elif amp > max_amp:
                max_amp = amp
        
        # Enforce slew rate - from zero to first
        amp0 = np.sqrt(I_arr[0]**2 + Q_arr[0]**2)
        if amp0 / dt > slew_bound:
            scale = slew_bound * dt / amp0
            I_arr[0] *= scale
            Q_arr[0] *= scale
        
        # Enforce slew rate - between steps
        for i in range(N - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            amp = np.sqrt(dI**2 + dQ**2)
            if amp / dt > slew_bound:
                scale = slew_bound * dt / amp
                I_arr[i+1] = I_arr[i] + dI * scale
                Q_arr[i+1] = Q_arr[i] + dQ * scale
        
        # Enforce slew rate - from last to zero
        amp_last = np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2)
        if amp_last / dt > slew_bound:
            scale = slew_bound * dt / amp_last
            I_arr[-1] *= scale
            Q_arr[-1] *= scale
        
        # Check if all constraints satisfied
        valid = True
        for i in range(N):
            if np.sqrt(I_arr[i]**2 + Q_arr[i]**2) > amp_bound + 1e-10:
                valid = False
                break
        if valid:
            if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / dt > slew_bound + 1e-10:
                valid = False
            for i in range(N - 1):
                dI = I_arr[i+1] - I_arr[i]
                dQ = Q_arr[i+1] - Q_arr[i]
                if np.sqrt(dI**2 + dQ**2) / dt > slew_bound + 1e-10:
                    valid = False
                    break
            if valid and np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / dt > slew_bound + 1e-10:
                valid = False
        if valid:
            break
    
    return I_arr, Q_arr

def _simulate_continuous(t_eval, I_func, Q_func, detuning, scale, initial_states):
    """Simulate using continuous integration with solve_ivp."""
    N = len(t_eval)
    
    def liouvillian(t, rho_vec):
        I = I_func(t)
        Q = Q_func(t)
        L_total = _STATIC_LIOUV[detuning]
        
        if abs(I) > 1e-15 or abs(Q) > 1e-15:
            coeff_I = 0.5 * scale * I
            coeff_Q = 0.5 * scale * Q
            L_total = L_total + (-1j * coeff_I * _KRON_AP_I + 1j * coeff_I * _KRON_I_APT 
                                + coeff_Q * _KRON_AM_I - coeff_Q * _KRON_I_AMT)
        
        return L_total @ rho_vec
    
    results = []
    for state in initial_states:
        rho = np.outer(state, state.conj())
        vec_rho = rho.flatten()
        
        sol = solve_ivp(liouvillian, [0, _T], vec_rho, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        rho_final = sol.y[:, -1].reshape(3, 3)
        results.append(rho_final)
    
    return results

def _score_pulse_continuous(I_func, Q_func):
    """Compute the score for a continuous pulse (lower is better)."""
    # Target states for X gate
    target_0 = np.array([0, 1, 0], dtype=complex)  # X|0> = |1>
    target_1 = np.array([1, 0, 0], dtype=complex)  # X|1> = |0>
    
    all_fidelities = []
    t_eval = np.arange(_N + 1) * _DT
    
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            initial_states = [
                np.array([1, 0, 0], dtype=complex),  # |0>
                np.array([0, 1, 0], dtype=complex),  # |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
            ]
            
            results = _simulate_continuous(t_eval, I_func, Q_func, det, scale, initial_states)
            
            targets = [
                target_1,  # for |0>
                target_0,  # for |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+|1>)/sqrt(2)
                (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-|1>)/sqrt(2)
                (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+i|1>)/sqrt(2)
                (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-i|1>)/sqrt(2)
            ]
            
            for i, target in enumerate(targets):
                fid = _compute_fidelity(results[i], target)
                all_fidelities.append(fid)
    
    mean_fid = np.mean(all_fidelities)
    max_fid = np.max(all_fidelities)
    infidelity = 1 - mean_fid + (1 - max_fid)
    
    return infidelity

def pulse(target_gate, hardware_model):
    """
    Generate a robust control pulse for a three-level transmon.
    
    Args:
        target_gate: 'X_pi' for a pi pulse on the X gate
        hardware_model: dict with hardware parameters
    
    Returns:
        (80, 2) array of I and Q amplitudes (rad/ns)
    """
    if target_gate != 'X_pi':
        raise ValueError(f"Unknown target gate: {target_gate}")
    
    # Extract hardware parameters
    anharmonicity = hardware_model.get('anharmonicity', -2 * np.pi * 0.25)
    amp_bound = hardware_model.get('amplitude_bound', 0.4)
    slew_bound = hardware_model.get('slew_bound', 0.12)
    detuning_bounds = hardware_model.get('detuning_bounds', [-2*np.pi*0.002, 2*np.pi*0.002])
    drive_scale_bounds = hardware_model.get('drive_scale_bounds', [0.97, 1.03])
    T1 = hardware_model.get('T1', 25000.0)
    Tphi = hardware_model.get('Tphi', 40000.0)
    
    # Try multiple DRAG beta values and sigma factors
    best_I, best_Q = None, None
    best_score = np.inf
    
    # Grid of beta values to try
    beta_values = np.linspace(-0.20, 0.20, 21)
    sigma_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    area_scales = [0.98, 0.99, 1.0, 1.01, 1.02]
    
    for sf in sigma_factors:
        for area_s in area_scales:
            for beta in beta_values:
                if _time_remaining() < 30:
                    break
                
                I, Q = _make_drag_pulse(beta, sf, area_scale=area_s)
                
                # Quick evaluation on nominal case first
                results = _simulate_pulse(I, Q, 0.0, 1.0)
                target_0 = np.array([0, 1, 0], dtype=complex)
                target_1 = np.array([1, 0, 0], dtype=complex)
                
                fid_0 = _compute_fidelity(results[0], target_1)
                fid_1 = _compute_fidelity(results[1], target_0)
                quick_score = 1 - (fid_0 + fid_1) / 2
                
                # If quick score is bad, skip full evaluation
                if quick_score > 0.3:
                    continue
                
                # Full evaluation
                score = _score_pulse(I, Q)
                
                if score < best_score:
                    best_score = score
                    best_I = I.copy()
                    best_Q = Q.copy()
        
        if _time_remaining() < 60:
            break
    
    # If no valid pulse found, use default
    if best_I is None:
        best_I, best_Q = _make_drag_pulse(0.05)
    
    # Return as (80, 2) array
    return np.column_stack([best_I, best_Q])
