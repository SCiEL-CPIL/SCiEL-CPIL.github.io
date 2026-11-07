"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using direct constrained optimization.
    
    Strategy:
    1. Use a fast unitary simulation (ignoring Lindblad operators) to evaluate
       infidelity for the 9 worst-case conditions (detuning/scale grid).
    2. Optimize the full 80-step I/Q vector directly using Nelder-Mead.
    3. Enforce amplitude and slew rate constraints via a penalty function.
    4. Return the best valid pulse found within the time budget.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    
    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    H_anh = 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
    
    # Simulation function for a given pulse and (delta, scale)
    def simulate_infidelity(pulse_vec, delta, scale):
        """
        Returns mean infidelity for the 5 test states.
        Uses unitary evolution (Lindblad ignored for speed).
        """
        I = pulse_vec[:n_steps]
        Q = pulse_vec[n_steps:]
        
        # Pre-compute interaction terms for speed? 
        # H(t) = H0 + 0.5 * scale * (I(t)(a+adag) + iQ(t)(a-adag))
        # H0 = delta * n + H_anh
        
        H0 = delta * n_op + H_anh
        H_I_term = 0.5 * (a + adag)
        H_Q_term = 0.5 * 1j * (a - adag)
        
        # Test states: |0>, |1>, (|0>+|1>)/sqrt2, (|0>-|1>)/sqrt2, (|0>+i|1>)/sqrt2, (|0>-i|1>)/sqrt2
        # Actually task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2). That's 6 states.
        # Wait, "mean fidelity to X over ...". X gate maps |0><->|1>.
        # Fidelity F = |<psi_target|psi_final>|^2.
        # For X gate:
        # |0> -> |1>
        # |1> -> |0>
        # (|0>+|1>)/sqrt2 -> (|1>+|0>)/sqrt2 = same state (up to phase? No, X|+> = |+>)
        # (|0>-|1>)/sqrt2 -> (|1>-|0>)/sqrt2 = -(|0>-|1>)/sqrt2 (phase -1, fidelity 1)
        # (|0>+i|1>)/sqrt2 -> (|1>+i|0>)/sqrt2 = i(|0>-i|1>)/sqrt2 ? 
        # Let's just define the 6 states and their targets.
        
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        ]
        
        # Targets for X_pi gate
        # X = [[0,1],[1,0]] in 2D. In 3D: |0><1| + |1><0|
        # X|0> = |1>
        # X|1> = |0>
        # X|+> = |+>
        # X|-> = -|-> (Global phase doesn't affect fidelity)
        # X|+i> = |1> + i|0> = i(|0> - i|1>) = i|-i> ? 
        # Let's check: X (|0> + i|1>)/sqrt2 = (|1> + i|0>)/sqrt2.
        # Target state for input |0> + i|1> is |1> + i|0>.
        
        targets = [
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 0, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([-1, 1, 0], dtype=complex) / np.sqrt(2), # X|-> = -|->, so target is -|->
            np.array([1j, 1, 0], dtype=complex) / np.sqrt(2), # X|+i> = i|0> + |1>
            np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2) # X|-i> = -i|0> + |1>
        ]
        
        fidelities = []
        
        # We can evolve all states simultaneously by stacking them?
        # Or just loop. 6 states * 80 steps * 9 cases = 4320 expm calls.
        # ExpM of 3x3 is fast.
        
        # To speed up, we can pre-compute the evolution operator for the full 40ns?
        # No, H(t) varies.
        
        # Optimization: Since we are optimizing the pulse, we call this many times.
        # We should vectorize the state evolution if possible, but 3x3 matrices are small.
        
        for psi0, psi_t in zip(states, targets):
            psi = psi0.copy()
            for i in range(n_steps):
                H_int = scale * (I[i] * H_I_term + Q[i] * H_Q_term)
                H_t = H0 + H_int
                # Evolve
                # Use scipy.linalg.expm
                U = opt.linalg.expm(-1j * H_t * dt) # Wait, scipy.linalg.expm is in scipy.linalg
                # Let's use np.linalg.eigh for speed on 3x3
                eigvals, eigvecs = np.linalg.eigh(H_t)
                U = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
                psi = U @ psi
            
            fid = np.abs(np.vdot(psi_t, psi))**2
            fidelities.append(fid)
        
        return 1.0 - np.mean(fidelities)

    def compute_score(pulse_vec):
        """
        Computes the score: mean + max infidelity over the 9 cases.
        Also checks validity constraints and returns a large penalty if invalid.
        """
        # Check validity
        I = pulse_vec[:n_steps]
        Q = pulse_vec[n_steps:]
        
        # Amplitude bound
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            return 1e6 + (np.max(amp) - amp_bound)
            
        # Slew rate bound
        p = np.column_stack([I, Q])
        # Check from zero to first
        if np.linalg.norm(p[0]) / dt > slew_bound:
            return 1e6 + (np.linalg.norm(p[0]) / dt - slew_bound)
        # Check between steps
        diffs = np.diff(p, axis=0)
        slews = np.linalg.norm(diffs, axis=1) / dt
        if np.max(slews) > slew_bound:
            return 1e6 + (np.max(slews) - slew_bound)
        # Check from last to zero
        if np.linalg.norm(p[-1]) / dt > slew_bound:
            return 1e6 + (np.linalg.norm(p[-1]) / dt - slew_bound)
            
        # Compute infidelities for 9 cases
        deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infs = []
        for d in deltas:
            for s in scales:
                inf = simulate_infidelity(pulse_vec, d, s)
                infs.append(inf)
                
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)

    # Initial guess: Gaussian DRAG
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    sigma = 6.0
    I0 = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I0) * dt
    I0 = I0 * (np.pi / area)
    if np.max(I0) > amp_bound:
         I0 *= amp_bound / np.max(I0)
    dI = np.gradient(I0, dt)
    alpha = 0.5 # Initial guess for DRAG
    Q0 = alpha * dI
    
    p0 = np.concatenate([I0, Q0])
    
    # Ensure initial guess is valid
    if compute_score(p0) > 1e5:
        # Scale down if needed
        amp = np.sqrt(I0**2 + Q0**2)
        if np.max(amp) > amp_bound:
            scale_factor = amp_bound / np.max(amp) * 0.9
            p0 *= scale_factor
        # Check slew
        p = np.column_stack([p0[:n_steps], p0[n_steps:]])
        diffs = np.diff(p, axis=0)
        slews = np.linalg.norm(diffs, axis=1) / dt
        s0 = np.linalg.norm(p[0]) / dt
        s_end = np.linalg.norm(p[-1]) / dt
        max_slew = np.max([np.max(slews), s0, s_end])
        if max_slew > slew_bound:
             scale_factor = slew_bound / max_slew * 0.9
             p0 *= scale_factor
             
    # Optimization
    start_time = time.time()
    time_limit = 300 # 300s budget, leave margin
    best_score = compute_score(p0)
    best_pulse = p0.copy()
    
    # Use Nelder-Mead
    # To handle the 160 dimensions, we might need to limit iterations or use a simpler method.
    # Nelder-Mead can be slow in high dimensions.
    # Alternative: Coordinate descent or Simulated Annealing?
    # Let's try Nelder-Mead with a reasonable maxiter.
    
    # We will run the optimization in a loop, checking time.
    # Since scipy.optimize.minimize doesn't easily support early termination based on wall clock,
    # we can wrap the objective to check time and return a large value to stop?
    # Or just run a fixed number of iterations and check time between restarts?
    # Better: Use a custom loop or just set maxiter high and rely on the fact that 
    # if it's too slow, we return the best found so far.
    
    # Actually, 160 dimensions is quite high for Nelder-Mead.
    # Let's try to reduce dimensions? No, the task allows any shape.
    # Maybe use `scipy.optimize.dual_annealing`? No, that's stochastic.
    # Let's stick to Nelder-Mead but with a tight maxiter and multiple restarts if time permits.
    
    # To make it faster, we can simplify the simulation.
    # The simulation is the bottleneck.
    # 9 cases * 6 states * 80 steps = 4320 matrix exponentiations.
    # Each expm/eigh is ~10-50 us. Total ~0.1s per evaluation.
    # Nelder-Mead needs ~N*10 evaluations = 1600 evaluations.
    # 1600 * 0.1s = 160s. This is tight but feasible.
    
    # Let's optimize the simulation speed.
    # We can pre-compute the eigen-decomposition of H_I and H_Q? No, they don't commute with H0.
    # But we can avoid constructing the full matrix if we use sparse? No, 3x3 is dense.
    
    # Let's try to speed up `simulate_infidelity`.
    # We can batch the 6 states into a (6, 3) matrix and evolve them all at once?
    # U @ psi_matrix. Yes.
    
    def simulate_infidelity_fast(pulse_vec, delta, scale):
        I = pulse_vec[:n_steps]
        Q = pulse_vec[n_steps:]
        
        H0 = delta * n_op + H_anh
        H_I_term = 0.5 * (a + adag)
        H_Q_term = 0.5 * 1j * (a - adag)
        
        states = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
            [1/np.sqrt(2), 1j/np.sqrt(2), 0],
            [1/np.sqrt(2), -1j/np.sqrt(2), 0]
        ], dtype=complex)
        
        targets = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
            [-1/np.sqrt(2), 1/np.sqrt(2), 0],
            [1j/np.sqrt(2), 1/np.sqrt(2), 0],
            [-1j/np.sqrt(2), 1/np.sqrt(2), 0]
        ], dtype=complex)
        
        psi_matrix = states.T # (3, 6)
        
        for i in range(n_steps):
            H_int = scale * (I[i] * H_I_term + Q[i] * H_Q_term)
            H_t = H0 + H_int
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            psi_matrix = U @ psi_matrix
            
        # Compute fidelities
        # F_k = | <t_k| psi_k> |^2
        # psi_k is column k of psi_matrix.
        # t_k is row k of targets? No, targets are rows.
        # vdot(t_k, psi_k) = sum(t_k^* * psi_k)
        # We can do element-wise multiply and sum.
        # targets.T is (3, 6). psi_matrix is (3, 6).
        # inner_products = np.sum(np.conj(targets.T) * psi_matrix, axis=0)
        inner_products = np.sum(np.conj(targets.T) * psi_matrix, axis=0)
        fidelities = np.abs(inner_products)**2
        
        return 1.0 - np.mean(fidelities)

    def compute_score_fast(pulse_vec):
        I = pulse_vec[:n_steps]
        Q = pulse_vec[n_steps:]
        
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            return 1e6 + (np.max(amp) - amp_bound)
            
        p = np.column_stack([I, Q])
        if np.linalg.norm(p[0]) / dt > slew_bound:
            return 1e6 + (np.linalg.norm(p[0]) / dt - slew_bound)
        diffs = np.diff(p, axis=0)
        slews = np.linalg.norm(diffs, axis=1) / dt
        if np.max(slews) > slew_bound:
            return 1e6 + (np.max(slews) - slew_bound)
        if np.linalg.norm(p[-1]) / dt > slew_bound:
            return 1e6 + (np.linalg.norm(p[-1]) / dt - slew_bound)
            
        deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infs = []
        for d in deltas:
            for s in scales:
                inf = simulate_infidelity_fast(pulse_vec, d, s)
                infs.append(inf)
                
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)

    # Re-initialize with fast score
    best_score = compute_score_fast(p0)
    
    # Run optimization
    # We will use a simple loop of Nelder-Mead steps or just call minimize with maxiter.
    # To be safe with time, we will run minimize and break if time runs out.
    # However, minimize doesn't support easy breaking.
    # We can use a custom objective that checks time and returns a constant to stop progress?
    # Or just run a fixed number of iterations.
    
    # Let's try running minimize with maxiter=500.
    # If it's too slow, we return the best found.
    
    # To handle the time limit strictly, we can run the optimization in a background thread?
    # No, let's just be conservative.
    
    # We will do a few restarts if time allows.
    
    result = None
    try:
        # Use Nelder-Mead
        # maxiter=1000 might be too many. Let's start with 500.
        # xatol, fatol for convergence.
        result = opt.minimize(compute_score_fast, p0, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-8, 'disp': False})
        
        if result.fun < best_score:
            best_score = result.fun
            best_pulse = result.x
    except Exception as e:
        pass # Keep best_pulse from initial guess or previous restart

    # If time remains, try a second restart with a different initial guess
    elapsed = time.time() - start_time
    if elapsed < 250:
        # Try a different shape: e.g. Sech^2
        t_centered = t - total_time / 2.0
        # Sech pulse
        I1 = 1.0 / np.cosh(t_centered / 4.0)
        area = np.sum(I1) * dt
        I1 = I1 * (np.pi / area)
        if np.max(I1) > amp_bound:
            I1 *= amp_bound / np.max(I1)
        dI1 = np.gradient(I1, dt)
        Q1 = 0.5 * dI1
        p1 = np.concatenate([I1, Q1])
        
        # Ensure valid
        score1 = compute_score_fast(p1)
        if score1 < 1e5:
            try:
                result2 = opt.minimize(compute_score_fast, p1, method='Nelder-Mead', 
                                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8, 'disp': False})
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_pulse = result2.x
            except Exception:
                pass

    # Final validity check and cleanup
    p_final = best_pulse
    I_final = p_final[:n_steps]
    Q_final = p_final[n_steps:]
    
    # Enforce hard constraints one last time
    amp = np.sqrt(I_final**2 + Q_final**2)
    if np.max(amp) > amp_bound:
        scale_factor = amp_bound / np.max(amp) * 0.999
        p_final *= scale_factor
        I_final = p_final[:n_steps]
        Q_final = p_final[n_steps:]
        
    p = np.column_stack([I_final, Q_final])
    diffs = np.diff(p, axis=0)
    slews = np.linalg.norm(diffs, axis=1) / dt
    s0 = np.linalg.norm(p[0]) / dt
    s_end = np.linalg.norm(p[-1]) / dt
    max_slew = np.max([np.max(slews), s0, s_end])
    if max_slew > slew_bound:
        scale_factor = slew_bound / max_slew * 0.999
        p_final *= scale_factor
        I_final = p_final[:n_steps]
        Q_final = p_final[n_steps:]
        
    return np.column_stack([I_final, Q_final])

# EVOLVE-BLOCK-END
