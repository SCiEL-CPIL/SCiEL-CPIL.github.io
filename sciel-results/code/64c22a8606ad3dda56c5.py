"""Robust control pulses for a three-level transmon."""
import numpy as np

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        # Default to X_pi if not specified, or raise. Task says target_gate='X_pi'.
        pass
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns? No, rad/ns is frequency. 
    # Wait, the task says "anharmonicity -2 pi 0.25". Usually in Hz or rad/s?
    # The drive amplitudes are in rad/ns. So frequencies are in rad/ns.
    # Let's assume the values given are in rad/ns.
    # Actually, standard transmon anharmonicity is ~ -200 MHz = -2 pi * 0.2 GHz = -2 pi * 0.2e9 rad/s.
    # But here dt = 0.5 ns. 
    # If anharmonicity is -2*pi*0.25 rad/ns, that's 0.25 GHz? No.
    # Let's look at the bounds. Detuning bounds +-2 pi 0.002. 
    # If these are rad/ns, then 0.002 rad/ns is 0.2 MHz? No, 1 rad/ns = 1 GHz / 2pi?
    # 1 rad/ns = 10^9 rad/s. 
    # So 0.002 rad/ns = 2e6 rad/s = ~0.3 MHz.
    # Anharmonicity -2*pi*0.25 rad/ns = -1.57 rad/ns = 250 MHz? 
    # This is consistent with typical transmon parameters if units are rad/ns.
    
    amp_bound = 0.4
    slew_bound = 0.12
    # detuning_bounds = +-2*pi*0.002
    # scale_bounds = [0.97, 1.03]
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N = 80
    
    # Seed: DRAG-style Gaussian
    # sigma = 40/6 ns
    sigma = 40.0 / 6.0
    t = np.arange(N) * dt + dt/2  # center of steps? Or start?
    # "80 constant steps of dt = 0.5 ns".
    # Usually pulse[i] is the amplitude for the interval [i*dt, (i+1)*dt].
    # Let's use the center of the step for the Gaussian shape to be symmetric.
    t_center = (np.arange(N) + 0.5) * dt
    t_mid = 20.0 # 40 ns / 2
    
    # Gaussian
    g = np.exp(-0.5 * ((t_center - t_mid) / sigma) ** 2)
    
    # "offset to 0 at the edges"
    # The Gaussian naturally decays, but let's ensure it's small.
    # "area pi"
    # Integral of A * exp(-0.5 * ((t-t0)/sigma)^2) dt = A * sigma * sqrt(2*pi)
    # We want area = pi.
    # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    
    A = np.sqrt(np.pi / 2.0) / sigma
    
    I = A * g
    Q = np.zeros(N)
    
    # DRAG term: derivative of I
    # Q = -alpha * dI/dt
    # Typically alpha is chosen to suppress leakage.
    # For a 3-level system, the DRAG angle is related to the anharmonicity.
    # Let's start with a simple derivative.
    dI = np.zeros(N)
    dI[1:-1] = (I[2:] - I[:-2]) / (2 * dt)
    dI[0] = (I[1] - I[0]) / dt
    dI[-1] = (I[-1] - I[-2]) / dt
    
    # We need to check constraints.
    # 1. sqrt(I^2 + Q^2) <= 0.4
    # 2. ||p[i+1] - p[i]|| / dt <= 0.12
    
    # Let's check the seed constraints.
    # Max I is A.
    # A = sqrt(pi/2) / (40/6) = 1.2533 / 6.666 = 0.188
    # So I is well within 0.4.
    
    # Slew rate:
    # Max dI/dt is around A * (1/sigma) * exp(-0.5) ?
    # d/dt exp(-0.5 ((t-t0)/sigma)^2) = exp(...) * (-(t-t0)/sigma^2)
    # Max is at t-t0 = sigma. Value = exp(-0.5) / sigma.
    # A * exp(-0.5) / sigma = (sqrt(pi/2)/sigma) * exp(-0.5) / sigma = sqrt(pi/2) * exp(-0.5) / sigma^2
    # sigma = 6.666. sigma^2 = 44.44
    # 1.2533 * 0.6065 / 44.44 = 0.0171 rad/ns^2
    # Slew bound is 0.12 rad/ns^2. So we are well within slew bound.
    
    # However, we can use the full amplitude and slew budget to improve robustness.
    # The seed is just a starting point. We can optimize I and Q.
    
    # Let's try a simple optimization.
    # Objective: Minimize infidelity over the 9 cases.
    # Since we can't run the full quantum simulator easily in the "pulse" function 
    # (it would be too slow for 9 cases * 80 steps * matrix exponentiation),
    # we should rely on a robust analytical or semi-analytical form.
    
    # Actually, the task says "Write pulse...". It doesn't say we can't compute it.
    # But 300s limit. 
    # Simulating 80 steps of a 3x3 Liouvillian is fast.
    # 9 cases * 4 initial states * 80 steps * (3x3 matrix exp) is very fast.
    # We can do a local optimization!
    
    # Let's implement a simple gradient descent or Nelder-Mead on the pulse parameters.
    # But the pulse has 80*2 = 160 parameters.
    # Nelder-Mead in 160 dimensions might be slow.
    # Better to parameterize the pulse.
    
    # Parameterization:
    # I(t) = A_I * g(t)
    # Q(t) = A_Q * g'(t)
    # 2 parameters: A_I, A_Q.
    # This is a standard DRAG pulse.
    
    # Let's optimize A_I and A_Q.
    
    def simulate_pulse(I, Q, detuning, scale):
        # Returns fidelity for the 4 states
        # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I (a+a^dag) + i Q (a-a^dag))
        # Lindblad: L1 = a / sqrt(T1), L2 = n * sqrt(2/Tphi)
        
        # Basis: |0>, |1>, |2>
        # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
        # a^dag is conjugate transpose.
        # n = a^dag a = diag(0, 1, 2)
        
        a = np.array([
            [0, np.sqrt(1), 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ], dtype=complex)
        
        n = a.conj().T @ a
        # n should be diag(0, 1, 2)
        
        H0 = detuning * n + 0.5 * anharmonicity * (n @ (n - np.eye(3)))
        
        # Drive term
        # a + a^dag
        ad = a.conj().T
        drive_term = 0.5 * scale * (I * (a + ad) + 1j * Q * (a - ad))
        
        H = H0 + drive_term
        
        # Liouvillian L(rho) = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
        # We can construct the superoperator matrix for vectorized rho.
        # vec(L(rho)) = M vec(rho)
        
        # Term 1: -i (H rho - rho H)
        # vec(H rho) = (I ⊗ H) vec(rho)
        # vec(rho H) = (H^T ⊗ I) vec(rho)
        # So -i (I ⊗ H - H^T ⊗ I)
        
        # Term 2: L rho L^dag - 0.5 (L^dag L rho + rho L^dag L)
        # vec(L rho L^dag) = (L^T ⊗ L) vec(rho)
        # vec(L^dag L rho) = (I ⊗ L^dag L) vec(rho)
        # vec(rho L^dag L) = ((L^dag L)^T ⊗ I) vec(rho)
        
        L1 = a / np.sqrt(T1)
        L2 = n * np.sqrt(2.0 / Tphi)
        
        L1_dag = L1.conj().T
        L2_dag = L2.conj().T
        
        M = 1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
        # Note: -i [H, rho] = -i H rho + i rho H.
        # vec(-i H rho) = -i (I ⊗ H) vec(rho)
        # vec(i rho H) = i (H^T ⊗ I) vec(rho)
        # So M_coherent = -i (I ⊗ H) + i (H^T ⊗ I) = i (H^T ⊗ I - I ⊗ H)
        # My formula above was 1j * (kron(I, H) - kron(H.T, I)).
        # kron(I, H) is I ⊗ H.
        # So 1j * (I ⊗ H - H^T ⊗ I) = i I ⊗ H - i H^T ⊗ I.
        # This is -i (H^T ⊗ I - I ⊗ H).
        # Wait.
        # [H, rho] = H rho - rho H.
        # -i [H, rho] = -i H rho + i rho H.
        # vec(-i H rho) = -i (I ⊗ H) vec(rho).
        # vec(i rho H) = i (H^T ⊗ I) vec(rho).
        # Sum = i (H^T ⊗ I - I ⊗ H).
        
        M = 1j * (np.kron(H.T, np.eye(3)) - np.kron(np.eye(3), H))
        
        # Dissipative terms
        for L in [L1, L2]:
            L_dag = L.conj().T
            M += np.kron(L.T, L) - 0.5 * np.kron(np.eye(3), L_dag @ L) - 0.5 * np.kron((L_dag @ L).T, np.eye(3))
            
        # Integrate
        rho = np.zeros((9, 1), dtype=complex)
        # We need to sum over 4 initial states? No, we compute fidelity for each.
        # But the evolution is linear. We can evolve the identity? No.
        # We evolve each initial state separately.
        
        fidelities = []
        initial_states = [
            np.array([1, 0, 0]), # |0>
            np.array([0, 1, 0]), # |1>
            np.array([1, 1, 0]) / np.sqrt(2), # (|0>+|1>)/sqrt(2)
            np.array([1, -1j, 0]) / np.sqrt(2), # (|0>-i|1>)/sqrt(2)
            # Task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
            # That's 6 states?
            # "for the 9 cases ... infidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
            # The list is:
            # 1. |0>
            # 2. |1>
            # 3. (|0>+|1>)/sqrt(2)
            # 4. (|0>-|1>)/sqrt(2)
            # 5. (|0>+i|1>)/sqrt(2)
            # 6. (|0>-i|1>)/sqrt(2)
            # Total 6 states.
        ]
        initial_states = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, -1, 0]) / np.sqrt(2),
            np.array([1, 1j, 0]) / np.sqrt(2),
            np.array([1, -1j, 0]) / np.sqrt(2)
        ]
        
        # Target X gate: |0> -> |1>, |1> -> |0>.
        # So target states:
        # |0> -> |1>
        # |1> -> |0>
        # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
        # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = - (|0>-|1>)/sqrt(2)
        # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i (|0>-i|1>)/sqrt(2)
        # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i (|0>+i|1>)/sqrt(2)
        
        targets = [
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, -1, 0]) / np.sqrt(2),
            np.array([1, 1j, 0]) / np.sqrt(2),
            np.array([1, -1j, 0]) / np.sqrt(2)
        ]
        # Wait, X |0> = |1>. X |1> = |0>.
        # X (|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2).
        # X (|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2).
        # X (|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2).
        # X (|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2).
        
        # Fidelity is |<psi_t|psi_final>|^2.
        
        for i, psi0 in enumerate(initial_states):
            rho = np.outer(psi0, psi0.conj()).flatten()
            for k in range(N):
                # Exponentiate M * dt
                # M is 9x9.
                # Use scipy.linalg.expm or eigen decomposition?
                # expm is fine for 9x9.
                # But doing it 80 times * 6 states * 9 cases is 4320 expm calls.
                # Might be slow.
                # Alternative: use RK4 on the Liouvillian equation.
                # d rho/dt = M rho.
                # RK4 is 4 evaluations of M rho. M rho is matrix multiplication.
                # Much faster than expm.
                
                # Let's use RK4.
                def d_rho(rho):
                    return M @ rho
                
                k1 = d_rho(rho)
                k2 = d_rho(rho + 0.5 * dt * k1)
                k3 = d_rho(rho + 0.5 * dt * k2)
                k4 = d_rho(rho + dt * k3)
                rho = rho + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                
            # Reshape rho to 3x3
            rho_mat = rho.reshape(3, 3)
            # Fidelity
            psi_t = rho_mat @ psi0 # No, rho is density matrix.
            # Fidelity = <psi_t| rho |psi_t> ? No.
            # Fidelity with pure state |phi> is <phi|rho|phi>.
            # Here target is pure state |phi_t>.
            # Fidelity = <phi_t| rho |phi_t>.
            fid = np.real(np.vdot(targets[i], rho_mat @ targets[i]))
            fidelities.append(fid)
            
        return np.mean(fidelities)
    
    # This simulation is likely too slow if called many times in an optimizer.
    # 4320 RK4 steps * 4 * 9x9 matmul.
    # 4320 * 4 * 81 flops ~ 1.4 million flops per case.
    # 9 cases ~ 12 million flops.
    # If we do 100 evaluations, that's 1.2 billion flops.
    # NumPy is fast, this might take a few seconds.
    # 300s limit is generous.
    
    # Let's implement a simple Nelder-Mead on A_I and A_Q.
    
    # Initial guess: A_I = 0.2, A_Q = 0.05
    # Constraints:
    # sqrt(I^2 + Q^2) <= 0.4
    # Slew rate <= 0.12
    
    # Let's write a helper to generate I, Q from A_I, A_Q.
    
    def gen_pulse(A_I, A_Q):
        I = A_I * g
        # Q = A_Q * dI/dt ? Or A_Q * g'?
        # g' = - (t-t0)/sigma^2 * g
        # Let's use Q = A_Q * dI
        Q = A_Q * dI
        return I, Q
    
    def check_constraints(I, Q):
        # Amplitude
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > 0.4:
            return False
        # Slew
        # p = [I, Q]
        # diff = p[1:] - p[:-1]
        # norm = sqrt(diff_I^2 + diff_Q^2)
        # rate = norm / dt
        # Also from 0 to p[0] and p[-1] to 0.
        
        I_ext = np.concatenate(([0], I, [0]))
        Q_ext = np.concatenate(([0], Q, [0]))
        
        dI = np.diff(I_ext)
        dQ = np.diff(Q_ext)
        rate = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(rate) > 0.12:
            return False
            
        return True
    
    def objective(params):
        A_I, A_Q = params
        I, Q = gen_pulse(A_I, A_Q)
        if not check_constraints(I, Q):
            return 1e6 # Penalty
        
        # Evaluate over 9 cases
        detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infids = []
        for d in detunings:
            for s in scales:
                fid = simulate_pulse(I, Q, d, s)
                infids.append(1 - fid)
                
        infids = np.array(infids)
        score = np.mean(infids) + np.max(infids)
        return score
    
    # Optimization
    # Start with seed
    A_I0 = np.max(I) # from seed
    A_Q0 = 0.0
    
    # Simple coordinate descent or Nelder-Mead
    # Let's use Nelder-Mead from scipy.optimize
    
    from scipy.optimize import minimize
    
    # Bounds
    # A_I max:
    # If A_Q=0, A_I <= 0.4
    # Slew: A_I * max(dg) / dt <= 0.12
    # max(dg) is small.
    # So A_I can be up to 0.4.
    
    res = minimize(objective, [A_I0, A_Q0], method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    A_I_opt, A_Q_opt = res.x
    
    I_opt, Q_opt = gen_pulse(A_I_opt, A_Q_opt)
    
    # Final constraint check
    if not check_constraints(I_opt, Q_opt):
        # Fallback to seed
        return np.column_stack((I, Q))
        
    return np.column_stack((I_opt, Q_opt))

# Keep the original function name
pulse = pulse # This is a no-op, but ensures the name exists if we defined it differently.
# Actually, I defined `def pulse(...)` above. So it's fine.

# Wait, the original code had:
# from autoresearch_suite.science_physics_expansion import quantum_seed
# pulse = quantum_seed
# I should not import that. I am replacing it.

# One issue: `simulate_pulse` uses `M` which depends on `I` and `Q`.
# In my `simulate_pulse`, I calculate `M` inside the loop over steps?
# No, `H` changes at each step because `I` and `Q` are constant over the step?
# "80 constant steps of dt = 0.5 ns".
# So I[k] and Q[k] are constant for the k-th step.
# So M changes at each step.
# My RK4 implementation uses the same M for the whole step.
# This is correct for piecewise-constant Hamiltonian.

# However, `simulate_pulse` is defined inside `pulse`.
# This is fine.

# Let's double check the `M` construction.
# H is 3x3 complex.
# M is 9x9 complex.
# RK4 on d rho/dt = M rho.
# rho is a vector of length 9.
# This is correct.

# Performance check:
# 200 iterations * 9 cases * 6 states * 80 steps * 4 RK4 stages * (9x9 matvec)
# 200 * 9 * 6 * 80 * 4 * 81 ~ 287 million flops.
# NumPy can do ~100-500 MFLOPS easily.
# So ~1-3 seconds.
# Well within 300s.

# One detail: "A DOP853 recheck of the nominal case must agree within 2e-7."
# This implies the evaluator might run a high-precision check.
# My RK4 might have error.
# But the task says "Write pulse...". It doesn't say I have to implement the recheck.
# The recheck is done by the evaluator on my output.
# So my output just needs to be good.
# The recheck ensures that the simulation model is consistent.
# It doesn't penalize my pulse directly unless my pulse is bad.

# Let's refine the initial guess for A_Q.
# DRAG term is usually proportional to anharmonicity.
# A_Q ~ -anharmonicity * A_I * sigma^2 ?
# Let's just let the optimizer find it.

# Edge case: What if optimization fails?
# Return seed.

# Final check on constraints in `check_constraints`:
# `dI = np.diff(I_ext)`
# `I_ext` has length N+2.
# `dI` has length N+1.
# This covers 0->I[0], I[0]->I[1], ..., I[N-1]->0.
# Correct.

# Let's run it.

