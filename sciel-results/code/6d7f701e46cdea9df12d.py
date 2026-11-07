"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

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
# Note: The task description says a = diag(sqrt([1, 2]), 1). 
# Usually a is lower triangular. Let's stick to the standard transmon model 
# where a|n> = sqrt(n)|n-1>. 
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
# However, the current code uses np.diag([1.0, np.sqrt(2.0), 1.0]). 
# Let's re-read carefully: "a = diag(sqrt([1, 2]), 1)". 
# This is ambiguous. Standard physics: a is annihilation operator.
# If a = diag(...), it doesn't make sense for a transmon unless it's a specific basis.
# The current code's _A = np.diag([1.0, np.sqrt(2.0), 1.0]) implies a diagonal operator?
# No, the code defines _A and then uses it in H = ... I * (_A + _ADAG) ...
# If _A is diagonal, _A + _ADAG is diagonal, so no transitions? 
# Wait, the current code has:
# _A = np.diag([1.0, np.sqrt(2.0), 1.0])
# This would mean H_drive is diagonal. That can't drive transitions.
# 
# Let's look at Inspiration 2:
# a[0, 1] = 1.0
# a[1, 2] = np.sqrt(2.0)
# This is the standard annihilation operator.
# The task text says: "a = diag(sqrt([1, 2]), 1)". This is likely a typo in the prompt or my interpretation.
# "sqrt([1, 2])" usually means the vector [1, sqrt(2)].
# If a is the annihilation operator, it has entries a_{n-1, n} = sqrt(n).
# So a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]].
# The current code's _A is definitely wrong for a drive term if it's diagonal.
# However, the current code scored 0.59, which is not zero. 
# If _A was diagonal, the drive term I*(A+A^dag) would be diagonal, causing no transitions.
# The infidelity would be 1.0 (max) + 1.0 (mean) = 2.0.
# Score 0.59 is much better than 2.0.
# Therefore, the current code's _A MUST be doing something else or I am misinterpreting.
# 
# Let's re-read the current code:
# _A = np.diag([1.0, np.sqrt(2.0), 1.0])
# _ADAG = _A.conj().T
# H = ... 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
# If _A is real diagonal, _A - _ADAG = 0. _A + _ADAG = 2*_A.
# So H_drive = scale * I * _A. This is diagonal.
# 
# Why did it score 0.59?
# Maybe the "X_pi" gate target is just identity? No, X swaps 0 and 1.
# If the drive is diagonal, it only causes phase shifts.
# Fidelity to X would be low.
# 
# Perhaps the "a = diag(sqrt([1, 2]), 1)" in the prompt is a garbled way of saying 
# "The matrix a has entries...". 
# Given Inspiration 2 uses the standard a, and standard physics applies, 
# I will use the standard annihilation operator.
# 
# BUT, the current code is the "Parent". If I change the physics model, I might break the comparison.
# However, the prompt says "Physics: robust control pulses...". 
# If the current code is using a wrong physics model, it's a bug.
# Let's check the score 0.59. 
# If I use the standard a, I should get a better score.
# I will switch to the standard annihilation operator as in Inspiration 2, which is physically correct.

_A = np.zeros((3, 3), dtype=complex)
_A[0, 1] = 1.0
_A[1, 2] = np.sqrt(2.0)

_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

_I3 = np.eye(3, dtype=complex)

# Precompute static dissipative part
_L_DISS = (np.kron(_L1, _L1.conj()) - 0.5*(np.kron(_I3, _L1ADAG @ _L1) + np.kron(_L1ADAG @ _L1, _I3))
          + np.kron(_L2, _L2.conj()) - 0.5*(np.kron(_I3, _L2ADAG @ _L2) + np.kron(_L2ADAG @ _L2, _I3)))

def _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Enforce amplitude and slew rate constraints by clipping."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    
    for _ in range(100):
        # Enforce amplitude
        for i in range(N):
            amp = np.sqrt(I_arr[i]**2 + Q_arr[i]**2)
            if amp > amp_bound:
                scale = amp_bound / amp
                I_arr[i] *= scale
                Q_arr[i] *= scale
        
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
            for i in range(N - 1):
                dI = I_arr[i+1] - I_arr[i]
                dQ = Q_arr[i+1] - Q_arr[i]
                if np.sqrt(dI**2 + dQ**2) / dt > slew_bound + 1e-10:
                    valid = False
                    break
        if valid:
            if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / dt > slew_bound + 1e-10:
                valid = False
            if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / dt > slew_bound + 1e-10:
                valid = False
        if valid:
            break
    
    return I_arr, Q_arr

def _make_pulse(family, width_param, alpha_param, amp_scale):
    """Generate a pulse based on family and parameters."""
    t = np.arange(_N) * _DT
    center = _N * _DT / 2.0
    
    if family == 'gaussian':
        sigma = width_param
        env = np.exp(-0.5 * ((t - center) / sigma)**2)
        # Offset to zero at edges? No, standard Gaussian is fine, we scale area.
        # But we need to ensure it fits in time.
        I = env
        Q = np.zeros(_N)
        
    elif family == 'cosine':
        # Cosine window over the full duration or width?
        # Let's use a cosine bump of width = 2*width_param centered at center
        # Actually, simple cosine over the whole 40ns is a good candidate.
        # Let's try cosine over the full pulse duration T=40ns.
        I = np.cos(np.pi * (t / _T - 0.5)) # Cycles 0 to pi? 
        # Better: 0.5 * (1 + cos(2*pi*t/T)) is a cosine window.
        # Let's use a raised cosine: 0.5 * (1 - cos(2 * pi * t / T))
        I = 0.5 * (1 - np.cos(2 * np.pi * t / _T))
        Q = np.zeros(_N)
        
    elif family == 'boxcar':
        # Constant amplitude, clipped by slew
        I = np.ones(_N)
        Q = np.zeros(_N)
        
    elif family == 'drag':
        # DRAG: Gaussian + alpha * derivative
        sigma = width_param
        env = np.exp(-0.5 * ((t - center) / sigma)**2)
        # Derivative of Gaussian: - (t-center)/sigma^2 * env
        deriv = -(t - center) / (sigma**2) * env
        I = env
        Q = alpha_param * deriv
        
    else:
        I = np.zeros(_N)
        Q = np.zeros(_N)
    
    # Scale amplitude
    I = I * amp_scale
    Q = Q * amp_scale
    
    # Normalize area to approx Pi? 
    # The task says "seed is ... area pi". 
    # For robustness, we should probably normalize the integral to Pi for all shapes.
    # But if we change the shape, the "Pi" might not be the optimal area.
    # Let's just scale by amp_scale which is bounded by 0.4.
    # The constraints will clip it anyway.
    
    I, Q = _enforce_constraints(I, Q, _AMP, _SLEW, _DT)
    
    # Final amplitude check
    amps = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amps)
    if max_amp > 1e-10:
        # Don't rescale to 0.4, just ensure it's within bounds.
        # If we want to maximize drive, we might scale up to 0.4.
        # But slew might be violated. 
        # Let's leave it as is, the optimizer can vary amp_scale.
        pass
        
    return I, Q

def _liouvillian_step(dt, I, Q, detuning, scale):
    """Compute the Liouvillian superoperator for one step and return the propagator."""
    H = (detuning * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - _I3)) + 
         0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG)))
    
    # Coherent part
    L_coh = -1j * (np.kron(H, _I3) - np.kron(_I3, H.T))
    
    L_total = L_coh + _L_DISS
    
    # Propagator: expm(L_total * dt)
    return expm(L_total * dt)

def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return the final state for each initial basis state."""
    N = len(I_arr)
    
    # Precompute all propagators
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
    """Compute the score for a given pulse."""
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
                (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2), # for (|0>-|1>)/sqrt(2)
                (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2), # for (|0>+i|1>)/sqrt(2)
                (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),# for (|0>-i|1>)/sqrt(2)
            ]
            
            for rho_final, target in zip(results, targets):
                fid = _compute_fidelity(rho_final, target)
                all_fidelities.append(fid)
    
    all_fidelities = np.array(all_fidelities)
    mean_fidelity = np.mean(all_fidelities)
    max_fidelity = np.max(all_fidelities)
    
    infidelity_mean = 1.0 - mean_fidelity
    infidelity_max = 1.0 - max_fidelity
    
    score = infidelity_mean + infidelity_max
    return score

def pulse(target_gate, hardware_model):
    """Main entry point. Returns an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    best_I = None
    best_Q = None
    best_score = float('inf')
    
    # Define search space
    families = ['gaussian', 'cosine', 'boxcar', 'drag']
    
    # Parameters to sweep
    # width_param: sigma for gaussian, not used for cosine/boxcar, sigma for drag
    widths = [3.0, 4.0, 5.0, 6.0, 8.0] # ns
    # alpha_param: for DRAG
    alphas = [-0.5, -0.2, 0.0, 0.2, 0.5]
    # amp_scale: scaling factor
    amp_scales = [0.3, 0.4] # Max is 0.4
    
    # Initial best: Seed Gaussian
    I_seed, Q_seed = _make_pulse('gaussian', 40/6.0, 0.0, 0.4)
    score_seed = _score_pulse(I_seed, Q_seed)
    best_I, best_Q, best_score = I_seed, Q_seed, score_seed
    
    # Grid Search
    for family in families:
        for width in widths:
            for alpha in alphas:
                for amp in amp_scales:
                    if _time_remaining() < 10:
                        break
                    
                    I_test, Q_test = _make_pulse(family, width, alpha, amp)
                    
                    # Quick check: if pulse is zero, skip
                    if np.max(np.abs(I_test) + np.abs(Q_test)) < 1e-6:
                        continue
                    
                    score = _score_pulse(I_test, Q_test)
                    
                    if score < best_score:
                        best_score = score
                        best_I, best_Q = I_test, Q_test
                        
    # Local Refinement: Nelder-Mead on the best pulse found
    # We optimize the parameters of the best family? 
    # Or just perturb the I, Q arrays directly?
    # Direct perturbation is safer.
    
    if _time_remaining() > 30:
        # Simple local search
        current_I = best_I.copy()
        current_Q = best_Q.copy()
        current_score = best_score
        
        step_size = 0.05
        for _ in range(200):
            if _time_remaining() < 5:
                break
                
            improved = False
            # Try small random perturbations
            for _ in range(10):
                I_pert = current_I + np.random.randn(_N) * step_size
                Q_pert = current_Q + np.random.randn(_N) * step_size
                
                I_pert, Q_pert = _enforce_constraints(I_pert, Q_pert, _AMP, _SLEW, _DT)
                
                score_pert = _score_pulse(I_pert, Q_pert)
                
                if score_pert < current_score:
                    current_score = score_pert
                    current_I, current_Q = I_pert, Q_pert
                    improved = True
                    if score_pert < best_score:
                        best_score = score_pert
                        best_I, best_Q = current_I, current_Q
                    break # Restart perturbations from new best
                
            if not improved:
                step_size *= 0.8
            else:
                step_size = min(0.1, step_size * 1.1)

    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
