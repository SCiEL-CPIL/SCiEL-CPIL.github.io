# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
    drive_scale_bounds = (0.97, 1.03)
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    N = 80  # number of steps
    total_time = N * dt  # 40 ns
    
    # Build the 3-level Hamiltonian components
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator diag(0, 1, 2)
    I3 = np.eye(3)
    
    # Precompute Liouvillian components
    # H = detuning n + 0.5 chi n(n-1) + 0.5 scale (I (a+a^dag) + i Q (a-a^dag))
    # L = -i (H rho - rho H) + ...
    # We separate L = L_det + L_chi + I * L_I + Q * L_Q
    # Note: detuning and scale are fixed for a specific simulation case, but we can precompute the operators
    # Actually, the Liouvillian depends on detuning and scale.
    # We will precompute the "drive" part operators L_I and L_Q which are independent of det/scale.
    
    def build_liouvillian_base(detuning, scale):
        """Build the Liouvillian for I=0, Q=0."""
        H_det = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        L1 = a / np.sqrt(T1)
        L2 = n_op * np.sqrt(2.0 / Tphi)
        
        H_left = np.kron(I3, H_det)
        H_right = np.kron(H_det.T, I3)
        
        L1_dag_L1 = L1.conj().T @ L1
        L2_dag_L2 = L2.conj().T @ L2
        
        L1_term = np.kron(L1, L1.T)
        L2_term = np.kron(L2, L2.T)
        
        L1_diss_left = np.kron(I3, L1_dag_L1)
        L1_diss_right = np.kron(L1_dag_L1.T, I3)
        
        L2_diss_left = np.kron(I3, L2_dag_L2)
        L2_diss_right = np.kron(L2_dag_L2.T, I3)
        
        L_base = -1j * (H_left - H_right) + L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                        + L2_term - 0.5 * (L2_diss_left + L2_diss_right)
        return L_base

    def build_liouvillian_drive(scale):
        """Build the Liouvillian components for I and Q drives."""
        # H_drive = 0.5 scale (I (a+a^dag) + i Q (a-a^dag))
        # L_drive = -i [H_drive, rho]
        # This is linear in I and Q.
        # H_I = 0.5 scale (a + a^dag)
        # H_Q = 0.5 scale i (a - a^dag)
        
        H_I = 0.5 * scale * (a + a_dag)
        H_Q = 0.5 * scale * 1j * (a - a_dag)
        
        L_I = -1j * (np.kron(I3, H_I) - np.kron(H_I.T, I3))
        L_Q = -1j * (np.kron(I3, H_Q) - np.kron(H_Q.T, I3))
        
        return L_I, L_Q

    # Precompute for the 9 cases? No, scale changes.
    # But we can precompute the kron structures once.
    # Let's just compute L_base and L_I, L_Q on the fly for each det/scale pair, 
    # but cache them if possible. Since there are only 9 pairs, we can precompute them all.
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    cached_liouvillians = {}
    for det in detunings:
        for scale in scales:
            key = (det, scale)
            L_base = build_liouvillian_base(det, scale)
            L_I, L_Q = build_liouvillian_drive(scale)
            cached_liouvillians[key] = (L_base, L_I, L_Q)

    def simulate_pulse(pulse_IQ, detuning, scale):
        """
        Simulate the pulse and return the final density matrix.
        """
        rho = np.zeros((9,), dtype=complex)
        rho[0] = 1.0  # |0><0| is first element
        
        key = (detuning, scale)
        L_base, L_I, L_Q = cached_liouvillians[key]
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L = L_base + I_val * L_I + Q_val * L_Q
            
            U = expm(dt * L)
            rho = U @ rho
        
        return rho.reshape(3, 3)

    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        psi = target_state.flatten()
        return np.real(np.conj(psi) @ rho @ psi)
    
    def compute_infidelity(pulse_IQ):
        """Compute mean + max infidelity over the 9 test cases."""
        
        # Test states
        states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),  # (|0>+|1>)/sqrt(2)
            (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),  # (|0>-|1>)/sqrt(2)
            (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>+i|1>)/sqrt(2)
            (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>-i|1>)/sqrt(2)
        ]
        
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                for state in states:
                    # Apply X_pi to the state: -i * swap |0> and |1>
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    
                    fid = compute_fidelity(rho_final, target)
                    infid = 1.0 - fid
                    infidelities.append(infid)
        
        infidelities = np.array(infid)
        return np.mean(infid) + np.max(infid)
    
    def check_validity(pulse_IQ):
        """Check if the pulse satisfies all constraints."""
        # Amplitude bound
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        
        # Slew rate bound
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-12:
                return False
            prev = curr
        
        # Last step to zero
        diff = np.linalg.norm(prev - np.zeros(2))
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True
    
    def make_drag_pulse(alpha, sigma_factor=1.0):
        """Create a DRAG-style Gaussian pulse."""
        t = np.arange(N) * dt
        sigma = (40.0 / 6.0) * sigma_factor
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        # I component
        I = gaussian.copy()
        
        # DRAG Q component: Q = alpha * dI/dt
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity by scaling down if necessary
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                scale_factor = amplitude_bound / max_amp
                pulse_IQ *= scale_factor
            # Check slew
            prev = np.zeros(2)
            max_slew = 0
            for i in range(N):
                curr = pulse_IQ[i]
                diff = np.linalg.norm(curr - prev)
                max_slew = max(max_slew, diff / dt)
                prev = curr
            diff = np.linalg.norm(prev)
            max_slew = max(max_slew, diff / dt)
            
            if max_slew > slew_bound:
                scale_factor = slew_bound / max_slew
                pulse_IQ *= scale_factor
        
        return pulse_IQ

    # Optimization
    start_time = time.time()
    time_limit = 300.0
    
    best_pulse = make_drag_pulse(0.0)
    best_score = compute_infidelity(best_pulse)
    
    # Coarse grid search on alpha and sigma_factor
    alphas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, -0.5, -1.0, -2.0]
    sigma_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for alpha in alphas:
        for sf in sigma_factors:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 60:
                break
            
            p = make_drag_pulse(alpha, sf)
            if check_validity(p):
                s = compute_infidelity(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
    
    # Refine with Nelder-Mead
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e10
        alpha = params[0]
        sf = params[1]
        p = make_drag_pulse(alpha, sf)
        if not check_validity(p):
            return 1e10
        return compute_infidelity(p)
    
    # Initial guess from best grid point
    # We need to recover alpha and sf from best_pulse? 
    # It's easier to just restart NM from a few good points or the best one found.
    # Let's assume the best grid point is a good start.
    # We don't track the params that gave best_pulse, so let's just run NM from (0, 1) and maybe a few others.
    
    # Run NM from (0, 1)
    try:
        res = minimize(objective, [0.0, 1.0], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 200})
        if res.fun < best_score:
            best_score = res.fun
            best_pulse = make_drag_pulse(res.x[0], res.x[1])
    except Exception:
        pass
    
    # Run NM from best alpha found in grid? 
    # Let's just try a few starting points
    for start_alpha in [0.5, 1.0, 2.0, -1.0]:
        for start_sf in [0.9, 1.0, 1.1]:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 30:
                break
            try:
                res = minimize(objective, [start_alpha, start_sf], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 100, 'maxfev': 100})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pulse = make_drag_pulse(res.x[0], res.x[1])
            except Exception:
                pass
            if time.time() - start_time > time_limit - 30:
                break
    
    return best_pulse
# EVOLVE-BLOCK-END
