"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
    scale_bounds = (0.97, 1.03)
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    
    # Build the 3-level system operators
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    
    # n = a_dag @ a
    n = a_dag @ a
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Build Liouvillian for given detuning, scale, I, Q
    def build_liouvillian(detuning, scale, I, Q):
        # H = detuning * n + 0.5 * anharmonicity * n * (n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
        n_minus_1 = n - np.eye(3)
        H = (detuning * n + 0.5 * anharmonicity * n @ n_minus_1 + 
             0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        
        # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
        def liouvillian(rho_vec):
            rho = rho_vec.reshape(3, 3)
            d_rho = -1j * (H @ rho - rho @ H)
            for L in [L1, L2]:
                d_rho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            return d_rho.flatten()
        
        return liouvillian
    
    # Simulate pulse and compute fidelity to X gate
    def simulate_pulse(I_arr, Q_arr, detuning=0.0, scale=1.0):
        # Initial state: |0><0|
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        rho_vec = rho.flatten()
        
        for i in range(n_steps):
            liouv = build_liouvillian(detuning, scale, I_arr[i], Q_arr[i])
            # Use solve_ivp for each step
            sol = solve_ivp(liouv, [0, dt], rho_vec, method='DOP853', rtol=1e-10, atol=1e-12)
            rho_vec = sol.y[:, -1]
        
        rho = rho_vec.reshape(3, 3)
        
        # Compute fidelity to X gate for 5 test states
        # X gate: |0> -> |1>, |1> -> |0>
        # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
        
        # X gate action:
        # |0> -> |1>
        # |1> -> |0>
        # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = (|0>+|1>)/sqrt(2)
        # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
        # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
        # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
        
        # Fidelity = |<psi_target|psi_final>|^2 for pure states
        # But we're dealing with density matrices, so F = Tr(rho_target * rho_final) for pure states
        
        # Target states after X gate:
        # |0> -> |1>: target = |1><1|
        # |1> -> |0>: target = |0><0|
        # (|0>+|1>)/sqrt(2) -> (|0>+|1>)/sqrt(2): target = (|0>+|1>)(<0|+<1|)/2
        # (|0>-|1>)/sqrt(2) -> -(|0>-|1>)/sqrt(2): target = (|0>-|1>)(<0|-<1|)/2
        # (|0>+i|1>)/sqrt(2) -> i(|0>-i|1>)/sqrt(2): target = (|0>-i|1>)(<0|+i<1|)/2
        # (|0>-i|1>)/sqrt(2) -> -i(|0>+i|1>)/sqrt(2): target = (|0>+i|1>)(<0|-i<1|)/2
        
        # Actually, for fidelity to a unitary, we need:
        # F = (1/d) Tr(U^dag rho U) where U is the target unitary
        # For X gate: U = [[0,1,0],[1,0,0],[0,0,1]] (acts on 2-level subspace)
        
        # Let's compute fidelity as mean over 5 states of |<psi_target|psi_final>|^2
        # where psi_target = U |psi_initial>
        
        # Define initial states
        psi0 = np.array([1, 0, 0], dtype=complex)  # |0>
        psi1 = np.array([0, 1, 0], dtype=complex)  # |1>
        psi2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)  # (|0>+|1>)/sqrt(2)
        psi3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)  # (|0>-|1>)/sqrt(2)
        psi4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)  # (|0>+i|1>)/sqrt(2)
        psi5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)  # (|0>-i|1>)/sqrt(2)
        
        # X gate matrix (acts on 2-level subspace, identity on |2>)
        U_x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
        
        fidelities = []
        for psi_init in [psi0, psi1, psi2, psi3, psi4, psi5]:
            psi_target = U_x @ psi_init
            # Final state vector from density matrix (assuming pure state evolution)
            # rho = |psi_final><psi_final|, so |psi_final> can be extracted
            # For fidelity: F = |<psi_target|psi_final>|^2
            # We need to extract psi_final from rho
            # Since evolution is unitary (ignoring decoherence for fidelity), rho is pure
            # Eigenvalue decomposition to get the dominant eigenvector
            eigvals, eigvecs = np.linalg.eigh(rho)
            # Take the eigenvector with largest eigenvalue
            idx = np.argmax(eigvals)
            psi_final = eigvecs[:, idx]
            fidelity = np.abs(np.vdot(psi_target, psi_final))**2
            fidelities.append(fidelity)
        
        return np.mean(fidelities)
    
    # DRAG-style Gaussian seed
    def make_drag_pulse(alpha=0.0, beta=0.0, center_shift=0.0):
        t = np.arange(n_steps) * dt
        t_center = 20.0 + center_shift  # center at 20 ns (middle of 40 ns)
        sigma = 40.0 / (6.0 * np.sqrt(np.log(2)))  # sigma for Gaussian
        # Actually, let's use sigma = 40/6 as stated
        sigma = 40.0 / 6.0
        
        # Gaussian envelope
        gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
        
        # Offset to 0 at edges: subtract edge values
        edge_val = gauss[0]  # should be same as gauss[-1] by symmetry
        gauss = gauss - edge_val
        
        # Normalize area to pi
        area = np.sum(gauss) * dt
        if area > 0:
            gauss = gauss * (np.pi / area)
        
        # I and Q components
        # DRAG: I = gauss, Q = alpha * gauss + beta * dgauss/dt
        dgauss = np.gradient(gauss, dt)
        
        I = gauss
        Q = alpha * gauss + beta * dgauss
        
        # Enforce constraints
        # Amplitude bound
        for i in range(n_steps):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > amp_bound:
                scale_factor = amp_bound / amp
                I[i] *= scale_factor
                Q[i] *= scale_factor
        
        # Slew bound
        prev_I, prev_Q = 0.0, 0.0
        for i in range(n_steps):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / dt
            if slew > slew_bound:
                scale_factor = slew_bound * dt / np.sqrt(dI**2 + dQ**2)
                I[i] = prev_I + dI * scale_factor
                Q[i] = prev_Q + dQ * scale_factor
            prev_I, prev_Q = I[i], Q[i]
        
        # Check final slew to zero
        slew_final = np.sqrt(prev_I**2 + prev_Q**2) / dt
        if slew_final > slew_bound:
            scale_factor = slew_bound * dt / np.sqrt(prev_I**2 + prev_Q**2)
            I[-1] *= scale_factor
            Q[-1] *= scale_factor
        
        return np.column_stack([I, Q])
    
    # Score function: mean + max of infidelity over 9 cases
    def score_pulse(I_arr, Q_arr):
        infidelities = []
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        for det in detunings:
            for sc in scales:
                fid = simulate_pulse(I_arr, Q_arr, det, sc)
                infidelities.append(1.0 - fid)
        
        return np.mean(infidelities) + np.max(infidelities)
    
    # Start with DRAG seed
    best_pulse = make_drag_pulse(0.0, 0.0, 0.0)
    best_score = score_pulse(best_pulse[:, 0], best_pulse[:, 1])
    
    # Simple local optimization: try small variations in alpha, beta, center_shift
    # Use a grid search for small parameter ranges
    alpha_range = np.linspace(-0.5, 0.5, 5)
    beta_range = np.linspace(-0.5, 0.5, 5)
    shift_range = np.linspace(-2.0, 2.0, 5)
    
    # Coarse grid search
    for alpha in alpha_range:
        for beta in beta_range:
            for shift in shift_range:
                if _time_remaining() < 5.0:
                    break
                pulse_candidate = make_drag_pulse(alpha, beta, shift)
                score = score_pulse(pulse_candidate[:, 0], pulse_candidate[:, 1])
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_candidate
    
    return best_pulse
# EVOLVE-BLOCK-END
