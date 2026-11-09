"""Improved program for robust X-pi gate control pulse."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def _build_hamiltonian(detuning, anharmonicity, scale, I, Q):
    """Build the 3x3 Hamiltonian in rad/ns units."""
    # Ladder operators for 3-level system
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3)) 
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build the Liouvillian superoperator for the density matrix evolution."""
    # Lindblad operators
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    def liouvillian(rho_flat):
        rho = rho_flat.reshape(3, 3)
        # d(rho)/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
        d_rho = -1j * (H @ rho - rho @ H)
        for L in [L1, L2]:
            Ldag = L.conj().T
            d_rho += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
        return d_rho.flatten()
    
    return liouvillian

def _evolve_state(rho0, H, T1, Tphi, dt, n_steps):
    """Evolve the density matrix through n_steps with Hamiltonian H."""
    rho = rho0.copy()
    for _ in range(n_steps):
        # Use a simple RK4 step for the Liouvillian
        liouv = _build_liouvillian(H, T1, Tphi)
        k1 = liouv(rho.flatten())
        k2 = liouv((rho + 0.5 * dt * k1.reshape(3, 3)).flatten())
        k3 = liouv((rho + 0.5 * dt * k2.reshape(3, 3)).flatten())
        k4 = liouv((rho + dt * k3.reshape(3, 3)).flatten())
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4).reshape(3, 3)
    return rho

def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure state."""
    # Fidelity = <psi|rho|psi> for pure state |psi>
    return np.real(np.vdot(target_state, rho @ target_state))

def _compute_score(pulse_array, hardware_model):
    """Compute the score for a given pulse array."""
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    dt = 0.5  # ns
    n_steps = 80
    
    # Detuning and scale test cases
    detunings = [-2 * np.pi * 0.002e-3, 0.0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_plus_i = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_minus_i = (psi0 - 1j * psi1) / np.sqrt(2)
    
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    # Target: X gate (swap |0> and |1>)
    # X|0> = |1>, X|1> = |0>, X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2), etc.
    # For the subspace {|0>, |1>}, X is the Pauli-X gate
    def target_state(psi):
        # Apply X gate in the {|0>, |1>} subspace
        if np.allclose(psi, psi0):
            return psi1
        elif np.allclose(psi, psi1):
            return psi0
        elif np.allclose(psi, psi_plus):
            return psi_plus  # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
        elif np.allclose(psi, psi_minus):
            return psi_minus  # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
        elif np.allclose(psi, psi_plus_i):
            return psi_minus_i  # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
        elif np.allclose(psi, psi_minus_i):
            return psi_plus_i  # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
        else:
            return psi
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            case_infidelities = []
            for psi in initial_states:
                psi_t = target_state(psi)
                rho0 = np.outer(psi, psi.conj())
                # Evolve with the pulse
                rho_final = _evolve_with_pulse(rho0, pulse_array, det, anharmonicity, scale, T1, Tphi, dt, n_steps)
                fid = _fidelity(rho_final, psi_t)
                infid = 1.0 - fid
                case_infidelities.append(infid)
            mean_infid = np.mean(case_infidelities)
            max_infid = np.max(case_infidelities)
            infidelities.append(mean_infid + max_infid)
    
    return np.mean(infidelities)

def _evolve_with_pulse(rho0, pulse_array, detuning, anharmonicity, scale, T1, Tphi, dt, n_steps):
    """Evolve the density matrix through the pulse."""
    rho = rho0.copy()
    I_vals = pulse_array[:, 0]
    Q_vals = pulse_array[:, 1]
    
    for i in range(n_steps):
        H = _build_hamiltonian(detuning, anharmonicity, scale, I_vals[i], Q_vals[i])
        liouv = _build_liouvillian(H, T1, Tphi)
        k1 = liouv(rho.flatten())
        k2 = liouv((rho + 0.5 * dt * k1.reshape(3, 3)).flatten())
        k3 = liouv((rho + 0.5 * dt * k2.reshape(3, 3)).flatten())
        k4 = liouv((rho + dt * k3.reshape(3, 3)).flatten())
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4).reshape(3, 3)
    
    return rho

def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    t_center = 20.0
    sigma = 40.0 / 6.0
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to make edges zero
    env -= env[0]
    
    # Scale to have area pi
    area = np.sum(env) * dt
    if abs(area) > 1e-10:
        env = env * (np.pi / area)
    
    I = env
    Q = np.zeros(n_steps)
    
    return np.column_stack([I, Q])

def _check_constraints(pulse_array):
    """Check if pulse satisfies amplitude and slew rate constraints."""
    dt = 0.5
    n_steps = len(pulse_array)
    
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amplitudes = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.max(amplitudes) > 0.4 + 1e-9:
        return False
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero into step 1
    p0 = np.array([0.0, 0.0])
    p1 = pulse_array[0]
    if np.linalg.norm(p1 - p0) / dt > 0.12 + 1e-9:
        return False
    
    # Between steps
    for i in range(n_steps - 1):
        diff = np.linalg.norm(pulse_array[i+1] - pulse_array[i])
        if diff / dt > 0.12 + 1e-9:
            return False
    
    # From last step to zero
    plast = pulse_array[-1]
    if np.linalg.norm(plast - p0) / dt > 0.12 + 1e-9:
        return False
    
    return True

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second deadline
    
    # Start with seed pulse
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse, hardware_model)
    
    # Try to improve with optimization
    # Use a parameterized basis: Gaussian with adjustable width, center, and DRAG-like Q component
    def params_to_pulse(params):
        sigma = np.exp(params[0])  # Positive width
        t_center = params[1]  # Center time
        drag_coeff = params[2]  # DRAG coefficient
        amplitude_scale = np.exp(params[3])  # Overall scale
        
        dt = 0.5
        n_steps = 80
        t = np.arange(n_steps) * dt
        
        env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        env -= env[0]
        
        area = np.sum(env) * dt
        if abs(area) > 1e-10:
            env = env * (np.pi / area)
        
        I = amplitude_scale * env
        Q = drag_coeff * amplitude_scale * np.gradient(env)
        
        return np.column_stack([I, Q])
    
    # Initial parameters
    sigma_init = np.log(40.0 / 6.0)
    t_center_init = 20.0
    drag_init = 0.0
    amp_scale_init = np.log(1.0)
    
    x0 = np.array([sigma_init, t_center_init, drag_init, amp_scale_init])
    
    # Try L-BFGS-B optimization
    try:
        def objective(x):
            p = params_to_pulse(x)
            if not _check_constraints(p):
                return 1e6  # Penalty for constraint violation
            return _compute_score(p, hardware_model)
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success and result.fun < best_score:
            best_pulse = params_to_pulse(result.x)
            best_score = result.fun
        
        # Check deadline
        if time.time() > deadline - 60.0:
            return best_pulse
        
    except Exception as e:
        pass
    
    # Try direct optimization on pulse values with projection
    try:
        x0 = best_pulse.flatten()
        
        def objective_direct(x):
            p = x.reshape(80, 2)
            if not _check_constraints(p):
                return 1e6
            return _compute_score(p, hardware_model)
        
        # Use Nelder-Mead for direct optimization
        result = minimize(objective_direct, x0, method='Nelder-Mead',
                         options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.success and result.fun < best_score:
            best_pulse = result.x.reshape(80, 2)
            best_score = result.fun
        
        # Check deadline
        if time.time() > deadline - 30.0:
            return best_pulse
            
    except Exception as e:
        pass
    
    return best_pulse
# EVOLVE-BLOCK-END
