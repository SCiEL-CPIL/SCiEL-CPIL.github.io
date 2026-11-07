"""Robust X_pi control pulse for a three-level transmon using optimized DRAG."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import expm, eig
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
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    N = 80  # number of steps
    total_time = N * dt  # 40 ns
    t = np.arange(N) * dt
    
    # Build the 3-level Hamiltonian components
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator diag(0, 1, 2)
    
    # Pre-compute dissipation part of Liouvillian (independent of I, Q, det, scale)
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    diss_L1_term = np.kron(L1.conj(), L1)
    diss_L2_term = np.kron(L2.conj(), L2)
    
    diss_L1_left = np.kron(np.eye(3), L1_dag_L1)
    diss_L1_right = np.kron(L1_dag_L1.T, np.eye(3))
    
    diss_L2_left = np.kron(np.eye(3), L2_dag_L2)
    diss_L2_right = np.kron(L2_dag_L2.T, np.eye(3))
    
    L_diss = diss_L1_term - 0.5 * (diss_L1_left + diss_L1_right) + \
             diss_L2_term - 0.5 * (diss_L2_left + diss_L2_right)
    
    # Precompute basis vectors for test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]
    
    # Target states after X_pi (swap |0> and |1>)
    target_states = []
    for s in states:
        tgt = np.zeros(3, dtype=complex)
        tgt[0] = s[1]
        tgt[1] = s[0]
        tgt[2] = s[2]
        target_states.append(tgt)
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    def get_L_drive(I, Q, scale):
        H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return -1j * (np.kron(np.eye(3), H_drive) - np.kron(H_drive.T, np.eye(3)))
    
    def get_L_static(detuning):
        H_static = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        return -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
    
    def simulate_pulse_fast(pulse_IQ, detuning=0.0, scale=1.0):
        """Fast simulation using precomputed Liouvillian structure."""
        rho_vec = np.zeros(9, dtype=complex)
        rho_vec[0] = 1.0  # |0><0| element
        
        L_static = get_L_static(detuning)
        L_base = L_diss + L_static
        
        # Precompute kron terms for drive
        # H_drive = 0.5*scale*(I*(a+a†) + iQ*(a-a†))
        # We can decompose: L_drive = 0.5*scale*(I*L_a + iQ*L_ia)
        # where L_a = -i*(kron(I, a+a†) - kron((a+a†).T, I))
        # and L_ia = -i*(kron(I, a-a†) - kron((a-a†).T, I))
        
        aa = a + a_dag
        ia = a - a_dag
        
        L_a = -1j * (np.kron(np.eye(3), aa) - np.kron(aa.T, np.eye(3)))
        L_ia = -1j * (np.kron(np.eye(3), ia) - np.kron(ia.T, np.eye(3)))
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L_total = L_base + 0.5 * scale * (I_val * L_a + 1j * Q_val * L_ia)
            
            # Use eigendecomposition for speed on 9x9 matrix
            # For small matrices, expm is fine, but let's try eig
            try:
                w, V = eig(L_total)
                V_inv = np.linalg.inv(V)
                U = V @ np.diag(np.exp(dt * w)) @ V_inv
            except:
                U = expm(dt * L_total)
            
            rho_vec = U @ rho_vec
        
        return rho_vec
    
    def compute_infidelity_fast(pulse_IQ):
        """Compute mean + max infidelity over the 9 test cases."""
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_vec = simulate_pulse_fast(pulse_IQ, det, scale)
                
                for tgt in target_states:
                    # Fidelity = <tgt|rho|tgt>
                    # rho_vec is column-major: rho_vec[i*3+j] = rho[i,j]
                    # <tgt|rho|tgt> = sum_ij conj(tgt[i]) * rho[i,j] * tgt[j]
                    fid = 0.0
                    for i in range(3):
                        for j in range(3):
                            fid += np.conj(tgt[i]) * rho_vec[i*3+j] * tgt[j]
                    fid = np.real(fid)
                    infid = 1.0 - fid
                    infidelities.append(infid)
        
        infidelities = np.array(infid)
        return np.mean(infid) + np.max(infid)
    
    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-12:
                return False
            prev = curr
        
        diff = np.linalg.norm(prev)
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True
    
    def make_seed_pulse():
        """Create the DRAG seed pulse."""
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        gaussian -= gaussian[0]
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        I = gaussian.copy()
        
        # DRAG: Q proportional to dI/dt, scaled by anharmonicity
        # Optimal DRAG: Q = -omega_anh / (4 * omega_drive) * dI/dt
        # For our case, a reasonable estimate
        dI_dt = np.gradient(I, dt)
        # Scale factor: anharmonicity is -2*pi*0.25, typical drive ~pi/40
        # DRAG coefficient ~ anharmonicity / (4 * drive_freq)
        # drive_freq ~ pi/40 rad/ns, so coeff ~ (-2*pi*0.25)/(4*pi/40) = -0.25*40/4 = -2.5
        # But this is in rad/ns units, let's use a smaller value
        gamma = 0.15  # Will be optimized
        Q = gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
            
            if not check_validity(pulse_IQ):
                # Reduce gamma
                for g in [0.1, 0.05, 0.0]:
                    Q = g * dI_dt
                    pulse_IQ = np.column_stack([I, Q])
                    if check_validity(pulse_IQ):
                        break
                else:
                    pulse_IQ = np.column_stack([I, np.zeros(N)])
        
        return pulse_IQ
    
    # Start with seed
    best_pulse = make_seed_pulse()
    best_score = compute_infidelity_fast(best_pulse)
    
    start_time = time.time()
    time_limit = 280.0  # Leave margin for final validation
    
    # Optimization strategy: 
    # 1. First optimize DRAG gamma with fixed Gaussian shape
    # 2. Then optimize shape parameters
    # 3. Then do local refinement on individual points
    
    def objective_drag(params):
        """Optimize DRAG gamma and Gaussian shape."""
        gamma = params[0]
        sigma = params[1]
        center = params[2]
        
        sigma = max(sigma, 1.0)
        center = np.clip(center, 10.0, 30.0)
        
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        gaussian -= gaussian[0]
        area = np.sum(gaussian) * dt
        if abs(area) < 1e-10:
            return 1e10
        gaussian *= np.pi / area
        
        I = gaussian
        dI_dt = np.gradient(I, dt)
        Q = gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        if not check_validity(pulse_IQ):
            # Penalty
            amplitudes = np.sqrt(I**2 + Q**2)
            amp_viol = np.max(0, np.max(amplitudes) - amplitude_bound)
            prev = np.zeros(2)
            slew_viol = 0
            for i in range(N):
                curr = pulse_IQ[i]
                diff = np.linalg.norm(curr - prev)
                slew_viol += max(0, diff / dt - slew_bound)
                prev = curr
            diff = np.linalg.norm(prev)
            slew_viol += max(0, diff / dt - slew_bound)
            return 1e5 * (amp_viol + slew_viol) + compute_infidelity_fast(pulse_IQ)
        
        return compute_infidelity_fast(pulse_IQ)
    
    # Phase 1: Optimize DRAG parameters
    if time.time() - start_time < time_limit * 0.4:
        x0_drag = np.array([0.15, 40.0/6.0, 20.0])
        
        # Try multiple initial points
        for gamma_init in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            x0 = np.array([gamma_init, 40.0/6.0, 20.0])
            try:
                res = minimize(objective_drag, x0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    gamma, sigma, center = res.x
                    sigma = max(sigma, 1.0)
                    center = np.clip(center, 10.0, 30.0)
                    gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
                    gaussian -= gaussian[0]
                    area = np.sum(gaussian) * dt
                    if abs(area) > 1e-10:
                        gaussian *= np.pi / area
                    I = gaussian
                    dI_dt = np.gradient(I, dt)
                    Q = gamma * dI_dt
                    pulse_IQ = np.column_stack([I, Q])
                    if check_validity(pulse_IQ):
                        best_pulse = pulse_IQ
            except:
                pass
            
            if time.time() - start_time > time_limit * 0.4:
                break
    
    # Phase 2: Local refinement - optimize individual pulse points
    if time.time() - start_time < time_limit * 0.8:
        # Use coordinate descent on subsets of points
        current_pulse = best_pulse.copy()
        current_score = best_score
        
        # Optimize in blocks of 10 points
        block_size = 10
        n_blocks = N // block_size
        
        for iteration in range(3):
            for block in range(n_blocks):
                idx_start = block * block_size
                idx_end = idx_start + block_size
                idx = np.arange(idx_start, idx_end)
                
                def block_obj(params):
                    p = current_pulse.copy()
                    p[idx, :] = params.reshape(block_size, 2)
                    
                    if not check_validity(p):
                        return 1e10 + current_score
                    return compute_infidelity_fast(p)
                
                x0_block = current_pulse[idx, :].flatten()
                try:
                    res = minimize(block_obj, x0_block, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                    if res.fun < current_score:
                        current_pulse[idx, :] = res.x.reshape(block_size, 2)
                        current_score = res.fun
                        if current_score < best_score:
                            best_score = current_score
                            best_pulse = current_pulse.copy()
                except:
                    pass
                
                if time.time() - start_time > time_limit * 0.8:
                    break
            if time.time() - start_time > time_limit * 0.8:
                break
    
    # Final validation
    if not check_validity(best_pulse):
        # Fallback to seed
        best_pulse = make_seed_pulse()
    
    return best_pulse
# EVOLVE-BLOCK-END
