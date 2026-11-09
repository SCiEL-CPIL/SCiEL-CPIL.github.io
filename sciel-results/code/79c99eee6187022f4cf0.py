"""Rewritten program for robust control pulses for a three-level transmon."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    det_bounds = [-2*np.pi*0.002, 2*np.pi*0.002]
    scale_bounds = [0.97, 1.03]
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    
    # Time grid
    t = np.arange(N) * dt + dt/2  # center of each step, from 0.25 to 39.75
    
    # Basis parameters
    sigma = 40.0 / 6.0  # ~6.67 ns
    tc = 20.0  # center time
    
    # Define basis functions
    def basis_funcs(t_arr, tc_val, sigma_val):
        """Return 3 basis functions evaluated at t_arr."""
        x = (t_arr - tc_val) / sigma_val
        h0 = np.exp(-x**2 / 2.0)
        h1 = x * np.exp(-x**2 / 2.0)
        h2 = (1.0 - x**2) * np.exp(-x**2 / 2.0)
        return h0, h1, h2
    
    def build_pulse(params):
        """Build pulse from 7 parameters: a0, a1, a2, b0, b1, b2, tc_offset."""
        a0, a1, a2, b0, b1, b2, tc_off = params
        tc_val = tc + tc_off
        h0, h1, h2 = basis_funcs(t, tc_val, sigma)
        I = a0*h0 + a1*h1 + a2*h2
        Q = b0*h0 + b1*h1 + b2*h2
        return np.column_stack([I, Q])
    
    def check_constraints(pulse_arr):
        """Check amplitude and slew constraints. Return True if valid."""
        # Amplitude constraint
        if np.any(np.sqrt(pulse_arr[:,0]**2 + pulse_arr[:,1]**2) > amp_bound + 1e-9):
            return False
        # Slew constraint: from 0 to step 0, between steps, from last step to 0
        p_prev = np.zeros(2)
        for i in range(N):
            p_curr = pulse_arr[i]
            if np.linalg.norm(p_curr - p_prev) / dt > slew_bound + 1e-9:
                return False
            p_prev = p_curr
        # From last step to zero
        if np.linalg.norm(np.zeros(2) - p_prev) / dt > slew_bound + 1e-9:
            return False
        return True
    
    # Quantum simulation
    def simulate(pulse_arr, detuning, scale):
        """Simulate the 3-level system and return final density matrix."""
        # Ladder operator a in 3-level basis
        a = np.zeros((3,3), dtype=complex)
        a[0,1] = 1.0
        a[1,2] = np.sqrt(2.0)
        a_dag = a.conj().T
        n = a_dag @ a
        
        # Hamiltonian
        H = detuning * n + 0.5 * alpha * n @ (n - np.eye(3)) 
        # Drive term: 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
        # This is time-dependent, so we need to handle it per step
        
        # Lindblad operators
        L1 = a / np.sqrt(T1)
        L2 = n * np.sqrt(2.0/Tphi)
        
        # Initial state: |0><0|
        rho = np.zeros((3,3), dtype=complex)
        rho[0,0] = 1.0
        
        for i in range(N):
            I_val = pulse_arr[i, 0]
            Q_val = pulse_arr[i, 1]
            
            # Drive Hamiltonian for this step
            H_drive = 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
            H_step = H + H_drive
            
            # Build Liouvillian superoperator
            # d(rho)/dt = -i[H,rho] + sum_k D[Lk] rho
            # Vectorize: vec(d(rho)/dt) = L_super * vec(rho)
            # L_super = -i*(H⊗I - I⊗H^T) + sum_k (Lk⊗Lk^* - 0.5*(Lk†Lk)⊗I - 0.5*(I⊗(Lk†Lk)^T))
            
            I3 = np.eye(3, dtype=complex)
            
            # Coherent part
            L_coh = -1j * (np.kron(H_step, I3) - np.kron(I3, H_step.T))
            
            # Dissipative part
            L_diss = np.zeros((9,9), dtype=complex)
            for L in [L1, L2]:
                LdagL = L.conj().T @ L
                L_diss += np.kron(L, L.conj()) - 0.5*np.kron(LdagL, I3) - 0.5*np.kron(I3, LdagL.T)
            
            L_super = L_coh + L_diss
            
            # Evolve for dt
            expL = expm(L_super * dt)
            rho_vec = expL @ rho.reshape(9)
            rho = rho_vec.reshape(3,3)
        
        return rho
    
    def fidelity_to_X(rho_final, state_idx):
        """Compute fidelity of final state with X|state>."""
        # X gate: |0><1| + |1><0|
        X = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
        
        # Initial states
        if state_idx == 0:
            psi = np.array([1,0,0], dtype=complex)
        elif state_idx == 1:
            psi = np.array([0,1,0], dtype=complex)
        elif state_idx == 2:
            psi = np.array([1,1,0], dtype=complex) / np.sqrt(2)
        elif state_idx == 3:
            psi = np.array([1,-1,0], dtype=complex) / np.sqrt(2)
        elif state_idx == 4:
            psi = np.array([1,1j,0], dtype=complex) / np.sqrt(2)
        elif state_idx == 5:
            psi = np.array([1,-1j,0], dtype=complex) / np.sqrt(2)
        
        psi_initial = np.outer(psi, psi.conj())
        psi_target = X @ psi
        rho_target = np.outer(psi_target, psi_target.conj())
        
        # Fidelity: F = Tr(rho_final @ rho_target) for pure target
        # Or use Uhlmann: F = (Tr(sqrt(sqrt(rho)@rho_target@sqrt(rho))))^2
        # For simplicity, use Tr(rho_final @ rho_target) since target is pure
        F = np.real(np.trace(rho_final @ rho_target))
        F = max(0, min(1, F))
        return F
    
    def compute_infidelities(pulse_arr):
        """Compute infidelities for all 9 cases and 6 states."""
        det_vals = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
        scale_vals = [0.97, 1.0, 1.03]
        
        case_infidelities = []
        for det in det_vals:
            for scale in scale_vals:
                rho_final = simulate(pulse_arr, det, scale)
                infids = []
                for s in range(6):
                    F = fidelity_to_X(rho_final, s)
                    infids.append(1.0 - F)
                case_infidelities.append(np.mean(infids))
        
        case_infidelities = np.array(case_infidelities)
        return np.mean(case_infidelities), np.max(case_infidelities)
    
    def objective(params):
        """Objective function: mean + max infidelity."""
        pulse_arr = build_pulse(params)
        if not check_constraints(pulse_arr):
            return 1e6  # penalty for invalid
        mean_inf, max_inf = compute_infidelities(pulse_arr)
        return mean_inf + max_inf
    
    # Initial guess: DRAG-style
    # Area of h0 = sigma*sqrt(2*pi) * (sigma in time units)
    # We want integral of I dt = pi
    # Integral of h0(t-tc) dt from -inf to inf = sigma*sqrt(2*pi)
    # So a0 = pi / (sigma*sqrt(2*pi))
    a0_init = np.pi / (sigma * np.sqrt(2*np.pi))
    a1_init = 0.1  # small DRAG correction
    a2_init = 0.0
    b0_init = 0.0
    b1_init = 0.0
    b2_init = 0.0
    tc_init = 0.0
    
    initial_guess = [a0_init, a1_init, a2_init, b0_init, b1_init, b2_init, tc_init]
    
    # Bounds for parameters
    bounds = [
        (-0.5, 0.5),   # a0
        (-0.5, 0.5),   # a1
        (-0.5, 0.5),   # a2
        (-0.5, 0.5),   # b0
        (-0.5, 0.5),   # b1
        (-0.5, 0.5),   # b2
        (-5.0, 5.0),   # tc_offset
    ]
    
    # Time budget management
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # First, try to get a valid seed pulse
    seed_pulse = build_pulse(initial_guess)
    if not check_constraints(seed_pulse):
        # Scale down to satisfy constraints
        scale_factor = 0.5
        for _ in range(10):
            scaled_pulse = build_pulse([p*scale_factor for p in initial_guess[:6]] + [initial_guess[6]])
            if check_constraints(scaled_pulse):
                break
            scale_factor *= 0.7
        best_pulse = scaled_pulse
    else:
        best_pulse = seed_pulse
    
    # Run differential evolution
    best_result = None
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            init='sobol',
        )
        if result.success or result.fun < 1e6:
            best_params = result.x
            best_pulse_candidate = build_pulse(best_params)
            if check_constraints(best_pulse_candidate):
                best_pulse = best_pulse_candidate
                best_result = result
    except Exception as e:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        # Try a second optimization with different settings if time permits
        try:
            result2 = differential_evolution(
                objective,
                bounds,
                seed=123,
                maxiter=30,
                popsize=10,
                tol=1e-4,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=True,
                init='latinhypercube',
            )
            if result2.fun < 1e6 and (best_result is None or result2.fun < best_result.fun):
                best_params = result2.x
                best_pulse_candidate = build_pulse(best_params)
                if check_constraints(best_pulse_candidate):
                    best_pulse = best_pulse_candidate
        except Exception:
            pass
    
    return best_pulse
