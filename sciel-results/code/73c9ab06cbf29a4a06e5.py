"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using analytic compensation.
    
    Strategy:
    1. Detuning Robustness: Set pulse area to A = pi / Delta_nominal.
       This ensures that for a given detuning Delta, the rotation angle 
       is approximately pi * (Delta_nominal / Delta). 
       For Delta = Delta_nominal, angle is pi.
       The first-order error in angle due to detuning fluctuation dDelta is 
       proportional to dDelta * dAngle/dDelta. By choosing A ~ 1/Delta, 
       we minimize the sensitivity of the effective Rabi frequency to 
       detuning in the relevant range? Actually, standard DRAG uses 
       fixed area pi. The 'robust' part usually comes from the shape.
       
       However, a key insight for robustness against detuning in a 
       fixed-time gate is to use a pulse shape that minimizes the 
       sensitivity to detuning. A Gaussian is standard. 
       
       Let's look at the error expansion.
       U = exp(-i * Omega_eff * T/2 * sigma_x) approx.
       Omega_eff = sqrt(omega_r^2 + Delta^2).
       For small Delta, Omega_eff ≈ omega_r + Delta^2/(2 omega_r).
       The phase error is Delta^2 * T / (4 omega_r).
       This is second order.
       
       The leakage to |2> is first order in Delta (via the DRAG term) 
       and second order in the pulse amplitude (via the anharmonicity 
       coupling).
       Standard DRAG sets Q = alpha * dI/dt.
       The optimal alpha for a Gaussian pulse with anharmonicity chi 
       is alpha = chi / (2 * omega_r) ? No, alpha = chi / (2 * pi) ?
       
       Let's use the standard result: 
       alpha = chi / (2 * omega_r) is not dimensionally correct if 
       omega_r is in rad/ns.
       The DRAG amplitude Q(t) = - (chi / 2) * (dI/dt) / omega_r ?
       Actually, the condition to cancel the |1> -> |2> leakage 
       induced by detuning Delta is:
       Q(t) = - (Delta / omega_r) * I(t) ? No, that's for adiabatic.
       
       For a resonant pulse (Delta=0), the leakage is due to the 
       non-adiabaticity and the anharmonicity.
       The standard DRAG correction for a Gaussian pulse is:
       Q(t) = alpha * dI/dt
       where alpha = chi / (2 * pi) is often cited? 
       Let's check dimensions. I is rad/ns. dI/dt is rad/ns^2. 
       Q is rad/ns. So alpha is ns.
       chi is rad/ns.
       So alpha ~ 1/chi * something?
       
       Actually, the optimal DRAG parameter for a Gaussian pulse 
       to minimize leakage to the second excited state is:
       alpha = chi / (2 * omega_r) ? No.
       
       Let's derive quickly or use the known optimal value.
       For a Gaussian pulse I(t) = A exp(-t^2/2tau^2), 
       the optimal DRAG amplitude is Q(t) = - (chi / 2) * (dI/dt) / A ?
       
       A common heuristic is Q(t) = (chi / 2) * dI/dt.
       Let's assume the drive is resonant (Delta=0) for the nominal 
       case, but the pulse must be robust.
       
       The most robust pulse against detuning is often one that 
       satisfies the "compensation" condition.
       
       Let's stick to the Gaussian shape but optimize the DRAG 
       coefficient alpha specifically for the hardware parameters 
       provided (anharmonicity -2 pi 0.25).
       
       The anharmonicity is chi = -2 pi * 0.25.
       The optimal DRAG coefficient for a Gaussian pulse is often 
       cited as alpha = chi / (2 * omega_r) ? 
       Wait, the standard DRAG paper (O'Brien et al., 2003) suggests 
       alpha = chi / (2 * pi) ? No.
       
       Let's use the formula: Q(t) = - (chi / 2) * (dI/dt) / omega_r ?
       If omega_r is the peak amplitude, this is tricky.
       
       Let's use the numerical optimization result from literature: 
       For a Gaussian pulse, the optimal DRAG coefficient is 
       alpha = chi / (2 * pi) is NOT correct.
       
       Let's use a simple, robust heuristic: 
       Q(t) = k * dI/dt
       We want to cancel the leakage. 
       The leakage is proportional to the integral of I(t) * (something).
       
       Let's try alpha = 0.5 / chi ? No.
       
       Actually, a very robust pulse is the "DRAG" pulse with 
       alpha = chi / (2 * omega_r) ?
       
       Let's look at the units again.
       H_int = I (a + a^dagger) + i Q (a - a^dagger).
       The |1> -> |2> transition is driven by the term proportional 
       to sqrt(2) Q (a - a^dagger) ? No, the matrix element for 
       |1> -> |2> is <2| H |1>.
       <2| a |1> = sqrt(2). <2| a^dagger |1> = 0.
       So H_21 = 0.5 * scale * (I * sqrt(2) + i Q * (-sqrt(2))) ?
       H = 0.5 scale (I (a + a^dagger) + i Q (a - a^dagger)).
       <2| H |1> = 0.5 scale (I <2|a|1> + i Q <2|(-a^dagger)|1> ? No.
       a |1> = |0>. a^dagger |1> = sqrt(2) |2>.
       So <2| a |1> = 0.
       <2| a^dagger |1> = sqrt(2).
       <2| -a^dagger |1> = -sqrt(2).
       So H_21 = 0.5 scale (I * 0 + i Q * (-sqrt(2))) = -i Q sqrt(2) / 2 * scale.
       Wait, the term is i Q (a - a^dagger).
       i Q (a - a^dagger) |1> = i Q (|0> - sqrt(2)|2>).
       <2| ... > = -i Q sqrt(2).
       So H_21 = 0.5 * scale * (-i Q sqrt(2)) = -i Q scale / sqrt(2).
       
       The detuning term H_det = Delta n.
       <2| H_det |1> = 0.
       But the detuning causes the |0> -> |1> and |1> -> |2> 
       frequencies to shift.
       
       The leakage to |2> comes from the Q component.
       To minimize leakage, we want Q to be as small as possible 
       while maintaining the X gate. But Q is needed to cancel 
       the "leakage" caused by the non-adiabaticity? 
       No, for a resonant pulse, Q=0 is optimal for leakage? 
       No, the standard DRAG says Q != 0 is better.
       
       The DRAG effect: The pulse shape I(t) has a finite width. 
       The "sudden" turn-on/off causes transitions to |2>. 
       The Q term, being the derivative, acts as a "filter" that 
       cancels these transitions.
       
       The optimal alpha for a Gaussian pulse is alpha = chi / (2 * pi) ?
       Let's try alpha = 0.5 * chi / (2 * pi) ?
       
       Let's use the value: alpha = chi / (2 * omega_r) is not 
       dimensionally consistent if omega_r is in rad/ns and alpha 
       is in ns.
       
       Let's use the standard result from the literature for 
       Gaussian DRAG:
       alpha = chi / (2 * pi) ? No.
       
       Let's calculate alpha numerically for the nominal case 
       using a fast grid search over a few values of alpha, 
       evaluating the fidelity for the nominal case (Delta=0, 
       scale=1). This is fast (10-20 evaluations).
       
       Then, to handle detuning, we rely on the fact that the 
       Gaussian DRAG pulse is already quite robust.
       
       To further improve robustness against detuning, we can 
       adjust the pulse area. 
       If we set Area = pi / (1 + Delta^2 / omega_r^2) ? 
       This is complex.
       
       Let's stick to:
       1. Gaussian I(t) with area pi.
       2. Q(t) = alpha * dI/dt.
       3. Optimize alpha for nominal case.
       4. This is a structural change from sech^2 to Gaussian 
          with optimized DRAG, and it uses a fast local search 
          for the DRAG coefficient.
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
    
    # Time points
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Gaussian pulse parameters
    # We need to choose sigma.
    # A typical sigma for 40ns is around 5-8 ns.
    # Let's pick sigma = 6.0 ns (similar to seed).
    sigma = 6.0
    
    # I(t)
    I = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I) * dt
    I = I * (np.pi / area)
    
    # Check amplitude bound
    if np.max(I) > amp_bound:
        I *= amp_bound / np.max(I)
        # Re-normalize area? No, if we scale down, area changes.
        # But we want area pi. If amp bound is hit, we can't have 
        # area pi with this sigma. We must increase sigma.
        # Let's increase sigma until max(I) <= amp_bound.
        while np.max(I) > amp_bound:
            sigma *= 1.1
            I = np.exp(-t_centered**2 / (2 * sigma**2))
            area = np.sum(I) * dt
            I = I * (np.pi / area)
    
    # Check slew rate
    dI = np.gradient(I, dt)
    # Q = alpha * dI
    # We need to find optimal alpha.
    
    # Fast simulation function for fidelity
    def simulate_fidelity(pulse):
        # pulse: (80, 2) array
        I = pulse[:, 0]
        Q = pulse[:, 1]
        
        # Build Hamiltonian
        # H = Delta n + 0.5 chi n(n-1) + 0.5 scale (I (a+adag) + i Q (a-adag))
        # For nominal case: Delta=0, scale=1
        Delta = 0.0
        scale = 1.0
        
        # Lindblad
        T1 = 25000.0
        Tphi = 40000.0
        gamma1 = 1.0 / T1
        gamma_phi = 1.0 / Tphi - 0.5 / T1 # Pure dephasing? 
        # Task says: Lindblad operators a / sqrt(T1) and n sqrt(2 / Tphi)
        # L1 = a / sqrt(T1)
        # L2 = n * sqrt(2 / Tphi)
        
        # We can use the density matrix evolution.
        # But for speed, we can use the state vector if we ignore 
        # dephasing? No, dephasing is in the score.
        # However, for optimizing alpha, we can ignore dephasing 
        # as it's a constant penalty. We only care about leakage 
        # and detuning error.
        
        # Let's use a simple unitary evolution for the 3 levels.
        # H is 3x3.
        # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
        # adag = a.T
        # n = adag @ a = diag(0, 1, 2)
        
        a = np.array([[0, 1, 0],
                      [0, 0, np.sqrt(2)],
                      [0, 0, 0]], dtype=complex)
        adag = a.conj().T
        n = adag @ a
        
        H0 = Delta * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
        
        # Initial state: |0>
        psi = np.array([1, 0, 0], dtype=complex)
        
        # Target state: X_pi |0> = |1>
        # Fidelity = |<1|psi>|^2
        
        # Time step dt = 0.5 ns
        # H(t) = H0 + 0.5 * scale * (I(t) (a + adag) + i Q(t) (a - adag))
        
        for i in range(n_steps):
            I_i = I[i]
            Q_i = Q[i]
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = H0 + H_int
            
            # Exponentiate
            # H_t is Hermitian.
            # Use scipy.linalg.expm or diagonalization.
            # For speed, we can use the fact that H_t is 3x3.
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            psi = U @ psi
        
        # Fidelity
        psi_1 = psi[1]
        fid = np.abs(psi_1)**2
        return fid
    
    # Grid search for alpha
    alphas = np.linspace(-0.5, 0.5, 20)
    best_alpha = 0.0
    best_fid = 0.0
    
    for alpha in alphas:
        Q = alpha * dI
        p = np.column_stack([I, Q])
        
        # Check validity
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            continue
        
        # Slew check
        valid = True
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > slew_bound:
                valid = False
                break
            p_prev = p[i].copy()
        if np.linalg.norm(p[-1]) / dt > slew_bound:
            valid = False
        
        if not valid:
            continue
        
        fid = simulate_fidelity(p)
        if fid > best_fid:
            best_fid = fid
            best_alpha = alpha
    
    # Use best alpha
    Q = best_alpha * dI
    p = np.column_stack([I, Q])
    
    # Final validity check
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        p *= scale
        I = p[:, 0]
        Q = p[:, 1]
    
    # Slew check
    max_slew = 0.0
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(p[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = p[i].copy()
    if np.linalg.norm(p[-1]) / dt > max_slew:
        max_slew = np.linalg.norm(p[-1]) / dt
    
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        p *= scale
    
    return p
# EVOLVE-BLOCK-END
