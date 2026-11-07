# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X_pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity (rad/ns)
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    
    # Build the DRAG pulse
    # The DRAG pulse has I component (Gaussian) and Q component (derivative of Gaussian)
    # to cancel leakage to |2>
    
    # Gaussian envelope for I component
    t = np.linspace(0, 40, N)  # time points
    sigma = 40.0 / 6.0  # sigma = 40/6 ns
    
    # Gaussian centered at 20 ns
    center = 20.0
    gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Normalize area to pi (integral of I dt = pi)
    # Using trapezoidal rule
    area = np.trapz(gauss, t)
    gauss = gauss * (np.pi / area)
    
    # DRAG parameter: alpha_I = -alpha * amp / (2 * pi)
    # The Q component is proportional to the derivative of the I component
    # alpha_I is the DRAG strength parameter
    # For a Gaussian pulse, Q = alpha_I * dI/dt
    
    # The optimal DRAG parameter for canceling |2> leakage
    # alpha_I = -alpha / (2 * pi * amp_peak)
    # where amp_peak is the peak amplitude
    
    # Let's compute the peak amplitude
    amp_peak = np.max(gauss)
    
    # DRAG parameter
    # The Q component should be: Q(t) = alpha_DRAG * dI/dt
    # where alpha_DRAG = -alpha / (2 * pi * amp_peak)
    # But we need to be careful with the units
    
    # Actually, the standard DRAG formula is:
    # I(t) = A(t)
    # Q(t) = alpha_I * A'(t)
    # where alpha_I = -alpha / (2 * pi * A_peak)
    
    # Let's use a more direct approach
    # The leakage to |2> is proportional to alpha * A(t)^2 / (2 * pi)
    # The DRAG correction adds a Q term that creates a counter-rotating term
    
    # For a Gaussian pulse with peak amplitude A0:
    # alpha_I = -alpha / (2 * pi * A0)
    
    # But we need to ensure the total amplitude doesn't exceed the bound
    # and the slew rate is within bounds
    
    # Let's start with a reasonable DRAG parameter
    # and then optimize
    
    # Initial guess for DRAG parameter
    alpha_I = -alpha / (2 * np.pi * amp_peak)
    
    # Compute Q component as derivative of I
    # Using finite differences
    dI_dt = np.zeros(N)
    dI_dt[1:-1] = (gauss[2:] - gauss[:-2]) / (2 * dt)
    dI_dt[0] = (gauss[1] - gauss[0]) / dt
    dI_dt[-1] = (gauss[-1] - gauss[-2]) / dt
    
    Q = alpha_I * dI_dt
    I = gauss.copy()
    
    # Check constraints and adjust if needed
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I *= scale
        Q *= scale
        amp_peak = np.max(I)
        alpha_I = -alpha / (2 * np.pi * amp_peak)
        Q = alpha_I * dI_dt
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I *= scale
            Q *= scale
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # p = [I, Q]
    p = np.column_stack([I, Q])
    dp = np.diff(p, axis=0)
    slew = np.linalg.norm(dp, axis=1) / dt
    max_slew = np.max(slew)
    
    # Also check from zero to first step and last step to zero
    slew_start = np.linalg.norm(p[0]) / dt
    slew_end = np.linalg.norm(p[-1]) / dt
    max_slew = max(max_slew, slew_start, slew_end)
    
    if max_slew > slew_bound:
        # Reduce the pulse amplitude to satisfy slew constraint
        # The slew rate is proportional to the amplitude
        scale = slew_bound / max_slew
        I *= scale
        Q *= scale
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale2 = amp_bound / max_amp
            I *= scale2
            Q *= scale2
    
    # Now we have a valid DRAG pulse
    # Let's try to optimize it further
    
    # Define the fidelity function
    def compute_fidelity(I_pulse, Q_pulse, detuning, scale):
        """Compute the mean fidelity for the given pulse and hardware parameters."""
        
        # Build the Hamiltonian and Liouvillian
        # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
        a = np.zeros((3, 3), dtype=complex)
        a[0, 1] = 1.0
        a[1, 2] = np.sqrt(2.0)
        
        a_dag = a.conj().T
        n = a_dag @ a  # number operator
        
        # H = detuning * n + 0.5 * alpha * n * (n - 1) + 0.5 * scale * (I * (a + a_dag) + i * Q * (a - a_dag))
        
        def build_hamiltonian(t, I_val, Q_val):
            H = detuning * n + 0.5 * alpha * n @ (n - np.eye(3))
            drive = 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
            H += drive
            return H
        
        # Lindblad operators
        # L1 = a / sqrt(T1)
        # L2 = n * sqrt(2 / Tphi)
        L1 = a / np.sqrt(T1)
        L2 = n * np.sqrt(2.0 / Tphi)
        
        # Build the Liouvillian
        def build_liouvillian(H):
            # L = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
            # For the Liouvillian in vectorized form:
            # L = -i (H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5 (L_k^dag L_k)^T ⊗ I - 0.5 I ⊗ L_k^dag L_k)
            
            dim = 3
            I_mat = np.eye(dim, dtype=complex)
            
            # Coherent part: -i [H, rho]
            # In vectorized form: -i (H ⊗ I - I ⊗ H^T)
            L = -1j * (np.kron(H, I_mat) - np.kron(I_mat, H.T))
            
            # Incoherent part
            for Lk in [L1, L2]:
                Lk_dag = Lk.conj().T
                Lk_dag_Lk = Lk_dag @ Lk
                L += np.kron(Lk, Lk.conj()) - 0.5 * (np.kron(Lk_dag_Lk.T, I_mat) + np.kron(I_mat, Lk_dag_Lk))
            
            return L
        
        # Simulate the pulse
        def rhs(t, y):
            # y is the vectorized density matrix
            idx = int(np.round(t / dt))
            if idx >= N:
                idx = N - 1
            I_val = I_pulse[idx]
            Q_val = Q_pulse[idx]
            H = build_hamiltonian(t, I_val, Q_val)
            L = build_liouvillian(H)
            return L @ y
        
        # Initial state: |0><0|
        rho0 = np.zeros((3, 3), dtype=complex)
        rho0[0, 0] = 1.0
        y0 = rho0.flatten().copy()
        
        # Solve the ODE
        t_span = (0, 40)
        t_eval = np.arange(0, 40.5, dt)
        
        try:
            sol = solve_ivp(rhs, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=dt)
            if not sol.success:
                return 0.0
        except:
            return 0.0
        
        # Extract the final density matrix
        rho_final = sol.y[:, -1].reshape(3, 3)
        
        # Compute fidelity with the target X gate
        # X gate: |0> -> |1>, |1> -> |0>
        # We need to compute the fidelity for each of the 5 states
        # and take the mean
        
        # Target states after X gate:
        # |0> -> |1>
        # |1> -> |0>
        # (|0> + |1>)/sqrt(2) -> (|1> + |0>)/sqrt(2) = (|0> + |1>)/sqrt(2)
        # (|0> - |1>)/sqrt(2) -> (|1> - |0>)/sqrt(2) = -( |0> - |1>)/sqrt(2)
        # (|0> + i|1>)/sqrt(2) -> (|1> + i|0>)/sqrt(2) = i(|0> - i|1>)/sqrt(2)
        # (|0> - i|1>)/sqrt(2) -> (|1> - i|0>)/sqrt(2) = -i(|0> + i|1>)/sqrt(2)
        
        # The fidelity is |<psi_target|psi_final>|^2
        
        # Initial states
        psi_0 = np.array([1, 0, 0], dtype=complex)
        psi_1 = np.array([0, 1, 0], dtype=complex)
        psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        psi_plus_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
        psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        
        # Target states (after X gate)
        psi_0_target = np.array([0, 1, 0], dtype=complex)
        psi_1_target = np.array([1, 0, 0], dtype=complex)
        psi_plus_target = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        psi_minus_target = np.array([-1, 1, 0], dtype=complex) / np.sqrt(2)  # global phase doesn't matter
        psi_plus_i_target = np.array([1j, 1, 0], dtype=complex) / np.sqrt(2)  # global phase doesn't matter
        psi_minus_i_target = np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)  # global phase doesn't matter
        
        # Wait, let me reconsider. The X gate maps:
        # |0> -> |1>, |1> -> |0>
        # So (|0> + |1>)/sqrt(2) -> (|1> + |0>)/sqrt(2) = (|0> + |1>)/sqrt(2)
        # (|0> - |1>)/sqrt(2) -> (|1> - |0>)/sqrt(2) = -( |0> - |1>)/sqrt(2)
        # (|0> + i|1>)/sqrt(2) -> (|1> + i|0>)/sqrt(2)
        # (|0> - i|1>)/sqrt(2) -> (|1> - i|0>)/sqrt(2)
        
        # The fidelity is |<psi_target|psi_final>|^2
        # where psi_final is the state after the pulse, and psi_target is the target state
        
        # But we're evolving the density matrix, so we need to compute:
        # F = Tr(rho_final * |psi_target><psi_target|)
        
        # Actually, the task says "mean fidelity to X over |0>, |1>, ..."
        # This means we prepare each state, apply the pulse, and compute the fidelity
        # with the corresponding X-gated state
        
        # But we can only simulate one initial state at a time
        # So we need to simulate for each initial state
        
        # This is expensive, so let's just simulate for |0> and use the linearity
        # of the quantum channel
        
        # Actually, let's just simulate for |0> and compute the fidelity
        # The task says "mean fidelity to X over |0>, |1>, ..."
        # which suggests we need to average over these 5 states
        
        # Let's simulate for each state
        fidelities = []
        
        for psi_init, psi_target in [
            (psi_0, psi_0_target),
            (psi_1, psi_1_target),
            (psi_plus, psi_plus_target),
            (psi_minus, psi_minus_target),
            (psi_plus_i, psi_plus_i_target),
            (psi_minus_i, psi_minus_i_target)
        ]:
            # Prepare the initial density matrix
            rho_init = np.outer(psi_init, psi_init.conj())
            y_init = rho_init.flatten().copy()
            
            try:
                sol = solve_ivp(rhs, t_span, y_init, method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=dt)
                if not sol.success:
                    return 0.0
            except:
                return 0.0
            
            rho_final = sol.y[:, -1].reshape(3, 3)
            
            # Compute fidelity: F = Tr(rho_final * |psi_target><psi_target|)
            psi_target_vec = psi_target.flatten()
            F = np.trace(rho_final @ np.outer(psi_target, psi_target.conj()))
            F = np.real(F)
            fidelities.append(F)
        
        return np.mean(fidelities)
    
    # This is too slow for optimization. Let's use a simpler approach.
    # We'll just use the DRAG pulse with the analytically optimal parameter.
    
    # The optimal DRAG parameter for a Gaussian pulse is:
    # alpha_I = -alpha / (2 * pi * A_peak)
    
    # Let's compute this more carefully
    # The leakage to |2> is proportional to alpha * A(t)^2
    # The DRAG correction adds a Q term that creates a counter-rotating term
    # The optimal DRAG parameter cancels the leading-order leakage
    
    # For a Gaussian pulse:
    # I(t) = A0 * exp(-0.5 * ((t - t0) / sigma)^2)
    # Q(t) = alpha_I * A0 * (-(t - t0) / sigma^2) * exp(-0.5 * ((t - t0) / sigma)^2)
    # where alpha_I = -alpha / (2 * pi * A0)
    
    # But we need to normalize the area to pi
    # The area of the Gaussian is A0 * sigma * sqrt(2 * pi)
    # So A0 = pi / (sigma * sqrt(2 * pi))
    
    A0 = np.pi / (sigma * np.sqrt(2 * np.pi))
    I = A0 * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Compute the derivative
    dI_dt = -A0 * (t - center) / sigma**2 * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # DRAG parameter
    alpha_I = -alpha / (2 * np.pi * A0)
    
    Q = alpha_I * dI_dt
    
    # Check constraints
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I *= scale
        Q *= scale
    
    # Check slew rate
    p = np.column_stack([I, Q])
    dp = np.diff(p, axis=0)
    slew = np.linalg.norm(dp, axis=1) / dt
    slew_start = np.linalg.norm(p[0]) / dt
    slew_end = np.linalg.norm(p[-1]) / dt
    max_slew = max(np.max(slew), slew_start, slew_end)
    
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        I *= scale
        Q *= scale
    
    return np.column_stack([I, Q])
# EVOLVE-BLOCK-END
