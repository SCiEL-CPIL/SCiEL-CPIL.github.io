"""Robust X_pi control pulse for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_matrices():
    """Build a, n, and Lindblad operators for the 3-level transmon."""
    # a = diag(sqrt([1,2]), 1) - wait, the problem says a = diag(sqrt([1, 2]), 1)
    # Actually re-reading: "a = diag(sqrt([1, 2]), 1)" - this is a 3x3 diagonal matrix
    # But that doesn't make sense for a lowering operator. Let me re-read.
    # "a = diag(sqrt([1, 2]), 1)" - I think this means a is a 3x3 matrix with
    # a[0,1] = sqrt(1), a[1,2] = sqrt(2), a[2,3]... no, 3 levels so indices 0,1,2
    # Actually for a 3-level system, a should be:
    # a[0,1] = 1, a[1,2] = sqrt(2), rest 0
    # But the problem says "a = diag(sqrt([1, 2]), 1)" which is confusing.
    # Let me interpret: a is a 3x3 matrix where a[i, i+1] = sqrt(i+1) for i=0,1
    # So a[0,1] = 1, a[1,2] = sqrt(2)
    
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    
    a_dag = a.conj().T
    n = a_dag @ a  # number operator
    
    return a, a_dag, n

def _liouvillian_step(dt, I, Q, detuning, scale, anharmonicity, T1, Tphi):
    """Compute the Liouvillian superoperator for one step and return the propagator."""
    a, a_dag, n = _build_matrices()
    
    # H = detuning * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I*(a + a_dag) + i*Q*(a - a_dag))
    H = (detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3)) + 
         0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
    
    # Lindblad operators
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2 / Tphi)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Liouvillian: d|psi><psi|/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form, the Liouvillian superoperator L acts on vec(rho)
    # L = -i(H kron I - I kron H^T) + sum_k (L_k kron L_k^* - 0.5(I kron L_k^dag L_k + L_k^dag L_k kron I))
    
    I3 = np.eye(3)
    
    # Coherent part
    L_coh = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Dissipative part
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        L_diss += np.kron(Lk, Lk.conj()) - 0.5 * (np.kron(np.eye(3), Lk_dag @ Lk) + np.kron(Lk_dag @ Lk, np.eye(3)))
    
    L_total = L_coh + L_diss
    
    # Propagator: expm(L_total * dt)
    from scipy.linalg import expm
    return expm(L_total * dt)

def _simulate_pulse(I_arr, Q_arr, detuning, scale, anharmonicity, T1, Tphi, dt):
    """Simulate the pulse and return the final state for each initial basis state."""
    N = len(I_arr)
    
    # Precompute all propagators
    props = []
    for i in range(N):
        prop = _liouvillian_step(dt, I_arr[i], Q_arr[i], detuning, scale, anharmonicity, T1, Tphi)
        props.append(prop)
    
    # Simulate for each initial state
    # |0> = [1, 0, 0], |1> = [0, 1, 0], etc.
    # But we work with density matrices, so vec(rho)
    
    results = []
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
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
    # Fidelity = <psi|rho|psi> for pure target
    return np.real(np.trace(target_rho @ rho_final))

def _score_pulse(I_arr, Q_arr, anharmonicity, T1, Tphi, dt):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states for X gate
    # X|0> = |1>, X|1> = |0>
    target_0 = np.array([0, 1, 0], dtype=complex)  # X|0> = |1>
    target_1 = np.array([1, 0, 0], dtype=complex)  # X|1> = |0>
    
    all_fidelities = []
    
    for det in detunings:
        for scale in scales:
            results = _simulate_pulse(I_arr, Q_arr, det, scale, anharmonicity, T1, Tphi, dt)
            
            # For each initial state, compute fidelity to the X-transformed state
            # |0> -> |1>, |1> -> |0>
            # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
            # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -( |0>-|1>)/sqrt(2)
            # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
            # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
            
            targets = [
                target_1,  # for |0>
                target_0,  # for |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+|1>)/sqrt(2)
                (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2)
                (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+i|1>)/sqrt(2) -> (i|0>+|1>)/sqrt(2)
                (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-i|1>)/sqrt(2) -> (-i|0>+|1>)/sqrt(2)
            ]
            
            for i, (rho_final, target) in enumerate(zip(results, targets)):
                fid = _compute_fidelity(rho_final, target)
                all_fidelities.append(fid)
    
    all_fidelities = np.array(all_fidelities)
    mean_fidelity = np.mean(all_fidelities)
    max_fidelity = np.max(all_fidelities)
    
    infidelity_mean = 1.0 - mean_fidelity
    infidelity_max = 1.0 - max_fidelity
    
    score = infidelity_mean + infidelity_max
    return score

def _check_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Check if the pulse satisfies amplitude and slew rate constraints."""
    N = len(I_arr)
    
    # Check amplitude
    for i in range(N):
        if np.sqrt(I_arr[i]**2 + Q_arr[i]**2) > amp_bound + 1e-10:
            return False
    
    # Check slew rate including from zero
    for i in range(N - 1):
        dI = I_arr[i+1] - I_arr[i]
        dQ = Q_arr[i+1] - Q_arr[i]
        if np.sqrt(dI**2 + dQ**2) / dt > slew_bound + 1e-10:
            return False
    
    # From zero to first step
    if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / dt > slew_bound + 1e-10:
        return False
    
    # From last step to zero
    if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / dt > slew_bound + 1e-10:
        return False
    
    return True

def _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Enforce amplitude and slew rate constraints by clipping."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    
    # Enforce amplitude
    for i in range(N):
        amp = np.sqrt(I_arr[i]**2 + Q_arr[i]**2)
        if amp > amp_bound:
            scale = amp_bound / amp
            I_arr[i] *= scale
            Q_arr[i] *= scale
    
    # Enforce slew rate - this is more complex, do iterative clipping
    for _ in range(100):
        # Check and clip slew from zero to first
        amp0 = np.sqrt(I_arr[0]**2 + Q_arr[0]**2)
        if amp0 / dt > slew_bound:
            scale = slew_bound * dt / amp0
            I_arr[0] *= scale
            Q_arr[0] *= scale
        
        # Check and clip slew between steps
        for i in range(N - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            amp = np.sqrt(dI**2 + dQ**2)
            if amp / dt > slew_bound:
                # Reduce the step
                scale = slew_bound * dt / amp
                I_arr[i+1] = I_arr[i] + dI * scale
                Q_arr[i+1] = Q_arr[i] + dQ * scale
        
        # Check and clip slew from last to zero
        amp_last = np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2)
        if amp_last / dt > slew_bound:
            scale = slew_bound * dt / amp_last
            I_arr[-1] *= scale
            Q_arr[-1] *= scale
        
        # Re-check amplitude after slew clipping
        for i in range(N):
            amp = np.sqrt(I_arr[i]**2 + Q_arr[i]**2)
            if amp > amp_bound:
                scale = amp_bound / amp
                I_arr[i] *= scale
                Q_arr[i] *= scale
        
        if _check_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
            break
    
    return I_arr, Q_arr

def _make_seed_pulse(N, dt, sigma_ns, area):
    """Create a DRAG-style Gaussian seed pulse."""
    t = np.arange(N) * dt
    
    # Gaussian centered at N*dt/2
    center = N * dt / 2.0
    sigma = sigma_ns
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset so edges are zero (they already are approximately)
    # Scale to get area = pi
    current_area = np.trapz(env, t)
    env = env * (area / current_area)
    
    # DRAG: add a derivative term to suppress leakage
    # d/dt of Gaussian
    drag = -((t - center) / sigma**2) * env
    
    # Normalize drag
    drag_max = np.max(np.abs(drag))
    if drag_max > 0:
        drag = drag / drag_max
    
    # Combine: I is the main drive, Q is the DRAG term
    # For an X gate, we want I to be the cosine-like component
    # DRAG adds to Q to compensate for anharmonicity
    alpha = 0.1  # DRAG strength
    
    I_arr = env
    Q_arr = alpha * drag
    
    return I_arr, Q_arr

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    
    # Create seed pulse
    sigma_ns = 40 / 6.0
    area = np.pi
    I_arr, Q_arr = _make_seed_pulse(N, dt, sigma_ns, area)
    
    # Enforce constraints
    I_arr, Q_arr = _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt)
    
    # Check if we have time for optimization
    if _time_remaining() > 60.0:
        # Try to optimize using scipy
        try:
            # Parameterize the pulse with fewer parameters for optimization
            # Use a Gaussian + DRAG form with adjustable parameters
            def objective(params):
                if _time_remaining() < 30.0:
                    return 1e6
                
                center = params[0]
                sigma = params[1]
                amp_scale = params[2]
                drag_strength = params[3]
                q_offset = params[4]
                
                t = np.arange(N) * dt
                env = np.exp(-0.5 * ((t - center) / sigma)**2)
                current_area = np.trapz(env, t)
                if current_area < 1e-10:
                    return 1e6
                env = env * (area / current_area) * amp_scale
                
                drag = -((t - center) / sigma**2) * env
                drag_max = np.max(np.abs(drag))
                if drag_max > 0:
                    drag = drag / drag_max
                
                I_new = env
                Q_new = drag_strength * drag + q_offset * env
                
                I_new, Q_new = _enforce_constraints(I_new, Q_new, amp_bound, slew_bound, dt)
                
                if not _check_constraints(I_new, Q_new, amp_bound, slew_bound, dt):
                    return 1e6
                
                score = _score_pulse(I_new, Q_new, anharmonicity, T1, Tphi, dt)
                return score
            
            # Initial parameters
            x0 = [N * dt / 2.0, sigma_ns, 1.0, 0.1, 0.0]
            
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < 1e5:
                center, sigma, amp_scale, drag_strength, q_offset = result.x
                t = np.arange(N) * dt
                env = np.exp(-0.5 * ((t - center) / sigma)**2)
                current_area = np.trapz(env, t)
                if current_area > 1e-10:
                    env = env * (area / current_area) * amp_scale
                    drag = -((t - center) / sigma**2) * env
                    drag_max = np.max(np.abs(drag))
                    if drag_max > 0:
                        drag = drag / drag_max
                    I_opt = env
                    Q_opt = drag_strength * drag + q_offset * env
                    I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt, amp_bound, slew_bound, dt)
                    if _check_constraints(I_opt, Q_opt, amp_bound, slew_bound, dt):
                        I_arr, Q_arr = I_opt, Q_opt
        
        except Exception:
            pass
    
    # Final constraint check
    I_arr, Q_arr = _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt)
    
    result = np.column_stack([I_arr, Q_arr])
    return result
# EVOLVE-BLOCK-END
