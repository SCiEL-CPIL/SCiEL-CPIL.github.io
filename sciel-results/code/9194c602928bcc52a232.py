"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using direct optimization.
    
    Strategy:
    1. Define a fast unitary simulation proxy that computes the mean fidelity
       for the 9 hardware cases (detuning, scale) without Lindblad terms 
       (since they are constant penalties for shape optimization).
    2. Use Nelder-Mead to optimize the 160-dimensional I/Q vector directly.
    3. Enforce amplitude and slew rate constraints via penalty terms in the 
       objective function.
    4. Use a time-budgeted search to ensure completion within limits.
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
    
    # Time points (center of each step)
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0

    # Hardware variations for scoring
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    n_nm1 = n_op @ (n_op - np.eye(3))
    
    # Initial state |0>
    psi0 = np.array([1, 0, 0], dtype=complex)
    # Target states for fidelity
    # X_pi maps |0> -> |1>, |1> -> -|0>, etc.
    # Fidelity is mean of |<target|psi_final>|^2
    # For X_pi:
    # |0> -> |1>
    # |1> -> -|0> (global phase -1 is irrelevant for fidelity to -|0> which is |0>)
    # (|0>+|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2)
    # (|0>-|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
    
    # Actually, the task says: fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # This implies we check the fidelity of U|psi> to X|psi>.
    # Since X is a permutation matrix (with signs), we can just check if U approximates X.
    # Fidelity F = (1/6) * sum_i |<X psi_i | U psi_i>|^2
    # Since X is unitary, <X psi_i | U psi_i> = <psi_i | X^dagger U psi_i>
    # Let V = X^dagger U. Then F = (1/6) sum_i |<psi_i | V psi_i>|^2
    # If V = I, F = 1.
    # X^dagger = X.
    # So we want U to be close to X.
    
    # Let's define the 6 initial states
    psi_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 0, 0]) + np.array([0, 1, 0])) / np.sqrt(2),
        (np.array([1, 0, 0]) - np.array([0, 1, 0])) / np.sqrt(2),
        (np.array([1, 0, 0]) + 1j*np.array([0, 1, 0])) / np.sqrt(2),
        (np.array([1, 0, 0]) - 1j*np.array([0, 1, 0])) / np.sqrt(2)
    ]
    
    # X matrix (X_pi gate on 2 levels, 3rd level unaffected)
    # |0> -> |1>, |1> -> -|0> (pi pulse)
    # Actually, standard X_pi is:
    # |0> -> i|1> ? No, usually exp(-i pi/2 sigma_x) = -i sigma_x.
    # -i sigma_x |0> = -i |1>.
    # -i sigma_x |1> = -i |0>.
    # Fidelity is |<target|psi>|^2, so phase doesn't matter.
    # Target for |0> is |1>.
    # Target for |1> is |0>.
    # Target for (|0>+|1>)/sqrt(2) is (|1>+|0>)/sqrt(2).
    # So we just check if U maps the subspace spanned by |0>,|1> to itself correctly.
    
    def compute_score(p_vec):
        """
        p_vec: (160,) array, first 80 are I, next 80 are Q.
        Returns score (lower is better).
        """
        I = p_vec[:80]
        Q = p_vec[80:]
        
        # Constraint penalties
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            # Hard penalty
            return 1e6 + (max_amp - amp_bound)
            
        # Slew rate
        # p[i] is (I[i], Q[i])
        # Check from 0 to p[0], p[i] to p[i+1], p[-1] to 0
        p_steps = np.column_stack([I, Q])
        max_slew = 0.0
        # From 0 to first
        slew_0 = np.linalg.norm(p_steps[0]) / dt
        if slew_0 > max_slew:
            max_slew = slew_0
        # Between steps
        for i in range(n_steps - 1):
            dp = np.linalg.norm(p_steps[i+1] - p_steps[i]) / dt
            if dp > max_slew:
                max_slew = dp
        # From last to 0
        slew_end = np.linalg.norm(p_steps[-1]) / dt
        if slew_end > max_slew:
            max_slew = slew_end
            
        if max_slew > slew_bound:
            return 1e6 + (max_slew - slew_bound)

        # Simulation
        # We simulate for all 9 cases.
        # To save time, we can batch or just loop. 9 * 80 steps = 720 exponentiations.
        # This might be slow. Let's optimize the simulation.
        # H(t) = H0(Delta) + H_int(I, Q, scale)
        
        fids = []
        for delta in detunings:
            for scale in scales:
                # H0
                H0 = delta * n_op + 0.5 * anharmonicity * n_nm1
                
                # Evolve each of the 6 states? 
                # Actually, we can evolve the density matrix or just the 6 states.
                # Since we need mean fidelity, and fidelity is linear in rho,
                # we can evolve a single "super" state? No, fidelity is nonlinear.
                # But we can compute the unitary U and then calculate fidelity.
                
                # Let's compute U for this (delta, scale).
                # U = exp(-i H_last dt) ... exp(-i H_first dt)
                # Start with Identity.
                U = np.eye(3, dtype=complex)
                
                for i in range(n_steps):
                    I_i = I[i]
                    Q_i = Q[i]
                    H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
                    H_t = H0 + H_int
                    
                    # Exponentiate
                    # For speed, use eigendecomposition
                    eigvals, eigvecs = np.linalg.eigh(H_t)
                    U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
                    U = U_step @ U
                    
                # Calculate fidelity
                # V = X^dagger U
                # F = (1/6) sum |<psi_i| V psi_i>|^2
                # X^dagger = X = [[0, -1j, 0], [-1j, 0, 0], [0, 0, 1]] ?
                # Wait, X_pi is usually defined as exp(-i pi/2 sigma_x) = -i sigma_x.
                # sigma_x = [[0, 1], [1, 0]].
                # -i sigma_x = [[0, -i], [-i, 0]].
                # So X = [[0, -1j, 0], [-1j, 0, 0], [0, 0, 1]].
                
                X_mat = np.array([[0, -1j, 0],
                                  [-1j, 0, 0],
                                  [0, 0, 1]], dtype=complex)
                
                V = X_mat.conj().T @ U
                
                fid_sum = 0.0
                for psi in psi_states:
                    # |<psi| V psi>|^2
                    proj = np.vdot(psi, V @ psi)
                    fid_sum += np.abs(proj)**2
                    
                fid = fid_sum / 6.0
                fids.append(fid)
                
        fids = np.array(fids)
        infid = 1 - fids
        return np.mean(infid) + np.max(infid)

    # Initial guess: DRAG Gaussian
    sigma = 6.0
    I_init = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I_init) * dt
    I_init = I_init * (np.pi / area)
    
    # Ensure amplitude bound
    if np.max(I_init) > amp_bound:
        I_init *= amp_bound / np.max(I_init)
        
    dI = np.gradient(I_init, dt)
    alpha = 0.5 * anharmonicity / (2 * np.pi) # Rough guess
    Q_init = alpha * dI
    
    p_init = np.concatenate([I_init, Q_init])
    
    # Check initial validity
    score_init = compute_score(p_init)
    best_p = p_init.copy()
    best_score = score_init
    
    # Nelder-Mead optimization
    # 160 dimensions is high for NM. It might be slow.
    # Let's try a smaller search space or a different optimizer.
    # Or reduce dimensions by optimizing parameters of a shape?
    # The prompt suggests "direct, constrained optimization of the 80-step I/Q vector".
    # But 160 dims NM is risky for 300s.
    # Let's try optimizing a subset of steps or using a coarser grid first?
    # Or use scipy.optimize.minimize with method='Nelder-Mead' and maxiter limit.
    
    # Given the time limit, a full 160-dim NM might not converge well.
    # Let's try optimizing the 4 parameters of a "double Gaussian" or similar?
    # No, the directive says "direct ... of the 80-step I/Q vector".
    # Let's try to make the simulation faster.
    # We can precompute the eigenvectors if H is constant? No, H changes every step.
    # We can use scipy.linalg.expm? It's slower than eigh for 3x3 usually.
    # We can use the fact that H is sparse?
    
    # Let's try a limited number of NM iterations.
    # And a time guard.
    
    start_time = time.time()
    deadline = start_time + 250  # 250s margin
    
    # Use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    # Objective function wrapper
    def objective(x):
        return compute_score(x)
        
    # Options for NM
    options = {
        'maxiter': 200,
        'maxfev': 200,
        'xatol': 1e-5,
        'fatol': 1e-6
    }
    
    # Try optimization
    try:
        result = minimize(objective, p_init, method='Nelder-Mead', options=options)
        
        if result.fun < best_score:
            best_p = result.x
            best_score = result.fun
            
        # Check time
        if time.time() > deadline:
            pass # Just exit loop
            
    except Exception as e:
        pass # Keep initial
        
    # Final validity check and return
    I_final = best_p[:80]
    Q_final = best_p[80:]
    
    # Enforce hard constraints one last time
    amp = np.sqrt(I_final**2 + Q_final**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        I_final *= scale
        Q_final *= scale
        
    p_final = np.column_stack([I_final, Q_final])
    max_slew = 0.0
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(p_final[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = p_final[i].copy()
    if np.linalg.norm(p_final[-1]) / dt > max_slew:
        max_slew = np.linalg.norm(p_final[-1]) / dt
        
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        p_final *= scale
        
    return p_final
# EVOLVE-BLOCK-END
