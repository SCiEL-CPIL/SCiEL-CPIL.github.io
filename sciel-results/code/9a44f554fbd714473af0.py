"""Gradient-based pulse optimization for 3-level transmon X-pi gate using adjoint method."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute Liouvillian terms that don't depend on H
# Dissipative part is constant
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

# Target operator X
X3 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 1]], dtype=complex)

# Test states
STATES = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
]

def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian matrix."""
    NN1 = N_MAT @ (N_MAT - I3)
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

def build_liouvillian_full(H):
    """Build full Liouvillian for given Hamiltonian."""
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def check_validity(pulse):
    """Check amplitude and slew constraints."""
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev - np.array([0.0, 0.0])) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def enforce_constraints(pulse):
    """Project pulse onto feasible set."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    # Enforce slew bound forward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Enforce slew bound backward (from last to zero)
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Re-enforce amplitude after slew projection
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    return pulse

def seed_pulse():
    """Generate a DRAG-style seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 5.0
    amp = 0.2
    center = 20.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    # Offset to 0 at edges
    edge_val = g[0]
    g = g - edge_val
    
    # Normalize area to pi
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
        
    # DRAG term
    dg = (t - center) / (sigma**2) * g
    if np.max(np.abs(dg)) > 0:
        dg = dg / np.max(np.abs(dg))
        
    I = amp * g
    Q = amp * (-0.1) * dg
    
    pulse = np.column_stack((I, Q))
    return enforce_constraints(pulse)

def compute_score_and_gradient(pulse):
    """
    Compute the score (mean + max infidelity) and its gradient with respect to the pulse.
    Uses adjoint method for gradient.
    """
    # Flatten pulse for gradient
    p_flat = pulse.flatten()
    
    total_score = 0.0
    total_grad = np.zeros_like(p_flat)
    max_infidelity = -np.inf
    max_case_idx = -1
    case_infideltas = []
    
    # First pass: compute infidelities to find max
    for ci, (detuning, scale) in enumerate(zip(np.repeat(DETUNING_VALUES, 3), np.tile(SCALE_VALUES, 3))):
        # Forward pass
        exp_mats = []
        exp_mats_dagger = []
        
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            exp_mats.append(sla.expm(L * DT))
            exp_mats_dagger.append(exp_mats[-1].conj().T)
            
        # Compute fidelity for each state
        infidelities = []
        for psi0 in STATES:
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.reshape(-1).copy()
            
            for i in range(N_STEPS):
                rho_vec = exp_mats[i] @ rho_vec
            
            rho = rho_vec.reshape(3, 3)
            psi_target = X3 @ psi0
            f = np.vdot(psi_target, rho @ psi_target).real
            f = np.clip(f, 0, 1)
            infidelities.append(1.0 - f)
            
        mean_inf = np.mean(infidelities)
        case_infideltas.append(mean_inf)
        if mean_inf > max_infidelity:
            max_infidelity = mean_inf
            max_case_idx = ci
            
    mean_inf = np.mean(case_infideltas)
    score = mean_inf + max_infidelity
    
    # Gradient calculation
    # Score = sum_c (1/9) I_c + I_max
    # Grad Score = sum_c (1/9) Grad I_c + Grad I_max
    
    for ci, (detuning, scale) in enumerate(zip(np.repeat(DETUNING_VALUES, 3), np.tile(SCALE_VALUES, 3))):
        weight = 1.0 / 9.0
        if ci == max_case_idx:
            weight += 1.0
            
        if weight == 0: continue
        
        # Adjoint method for this case
        # We need dI_c/dpulse
        # I_c = 1 - mean_s <psi_target_s| rho_f_s |psi_target_s>
        # dI_c = - mean_s <psi_target_s| d rho_f_s |psi_target_s>
        
        # Backward pass (adjoint)
        # Let chi_i be the adjoint variable at step i
        # d rho_f = sum_k (prod_{j=k+1}^{N-1} E_j) d E_k rho_{k-1} ...
        # This is complex. Let's use a simpler finite difference for gradient if adjoint is too slow/complex?
        # No, adjoint is better.
        
        # Let's implement the adjoint recursion properly.
        # rho_{i} = E_i rho_{i-1}
        # d rho_i = d E_i rho_{i-1} + E_i d rho_{i-1}
        # We want sum_i <chi_i| d rho_i>
        # d E_i = (d/dt E_i) * dt * d u_i
        # d/dt E(t) = -i (dH/dt) E(t) ... wait, E = exp(L dt). dE = -i (dH) E dt? No.
        # dE = (dL) E dt = -i (dH \otimes I - I \otimes dH^T) E dt
        
        # Let's store forward states
        rho_states = []
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            E = sla.expm(L * DT)
            if i == 0:
                rho0 = np.outer(STATES[0], STATES[0].conj()) # Placeholder, will loop states
                # Actually we need to do this for each state.
                pass
            
        # To save time, let's just do finite difference for gradient? 
        # 160 params * 9 cases * 6 states * 80 steps * expm... 
        # Finite difference: 2 * 160 evaluations of score.
        # Score eval: 9 * 6 * 80 expm = 4320 expm.
        # 320 * 4320 = 1.3M expm. Too slow.
        
        # Adjoint is necessary.
        # Let's implement adjoint for one case.
        
        # Precompute E and E^dagger for this case
        E_list = []
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            E_list.append(sla.expm(L * DT))
            
        # Adjoint for each state in this case
        grad_case = np.zeros_like(p_flat)
        
        for psi0 in STATES:
            rho0 = np.outer(psi0, psi0.conj()).reshape(-1)
            psi_target = X3 @ psi0
            psi_target_vec = psi_target.reshape(-1)
            
            # Forward: store rho_i
            rho_hist = [rho0]
            for i in range(N_STEPS):
                rho_hist.append(E_list[i] @ rho_hist[-1])
                
            # Backward: 
            # Objective term for this state: - <psi_t| rho_f |psi_t>
            # Let lambda_i be adjoint for rho_i
            # lambda_{N-1} corresponds to d rho_{N-1} term?
            # d J = sum_i <lambda_i| d rho_i>
            # d rho_i = dE_i rho_{i-1} + E_i d rho_{i-1}
            # <lambda_i| d rho_i> = <lambda_i| dE_i rho_{i-1}> + <lambda_i| E_i d rho_{i-1}>
            # = <E_i^dagger lambda_i| d rho_{i-1}> + <lambda_i| dE_i rho_{i-1}>
            # So lambda_{i-1} = E_i^dagger lambda_i
            # And the term contributing to d u_i is <lambda_i| dE_i rho_{i-1}>
            
            lambda_i = np.zeros(9, dtype=complex)
            # Final term: d J = - <psi_t| rho_f |psi_t> = - <psi_t psi_t^dagger | rho_f>
            # In vec form: - Tr(P^T rho_f) where P = |psi_t><psi_t|
            # = - <vec(P)| vec(rho_f)> ? 
            # vec(A X B) = (B^T \otimes A) vec(X)
            # Tr(P^T rho) = vec(P^T)^T vec(rho) = vec(P)^T vec(rho)
            # So lambda_{N-1} (adjoint for rho_{N-1} which is rho_f) is - vec(P)
            P = np.outer(psi_target, psi_target.conj())
            lambda_N_minus_1 = -P.reshape(-1)
            
            # Backward recursion
            # lambda_{N-1} is set.
            # For i = N-1 down to 0:
            # Contribution to d u_i: <lambda_i| dE_i rho_{i-1}>
            # Then lambda_{i-1} = E_i^dagger lambda_i
            
            for i in range(N_STEPS - 1, -1, -1):
                # Compute gradient contribution for step i
                # dE_i = -i (dH_i \otimes I - I \otimes dH_i^T) E_i
                # dH_i = 0.5 scale (dI_i (A+Ad) + i dQ_i (A-Ad))
                
                I_val, Q_val = pulse[i]
                H = build_hamiltonian(detuning, scale, I_val, Q_val)
                
                # We need <lambda_i| dE_i rho_{i-1}>
                # = <lambda_i| -i (dH_i \otimes I - I \otimes dH_i^T) E_i rho_{i-1}>
                # = -i <lambda_i| (dH_i \otimes I - I \otimes dH_i^T) rho_i>
                
                rho_i = E_list[i] @ rho_hist[i] # This is rho_i
                # Note: rho_hist[i] is rho_{i-1}. rho_i = E_i rho_{i-1}.
                
                # Let M = (dH_i \otimes I - I \otimes dH_i^T) rho_i
                # dH_i is linear in dI_i and dQ_i.
                # dH_i = 0.5 scale (dI_i X_drive + i dQ_i Y_drive)
                # where X_drive = A+Ad, Y_drive = A-Ad
                
                X_drive = A_MAT + AD_MAT
                Y_drive = A_MAT - AD_MAT
                
                # M(dI, dQ) = 0.5 scale (dI_i (X_drive \otimes I - I \otimes X_drive^T) rho_i + 
                #               i dQ_i (Y_drive \otimes I - I \otimes Y_drive^T) rho_i)
                
                # We want <lambda_i| M(dI, dQ)>
                # = 0.5 scale [ dI_i <lambda_i| (X_drive \otimes I - I \otimes X_drive^T) rho_i> +
                #               i dQ_i <lambda_i| (Y_drive \otimes I - I \otimes Y_drive^T) rho_i> ]
                
                # Let A_I = <lambda_i| (X_drive \otimes I - I \otimes X_drive^T) rho_i>
                # Let A_Q = <lambda_i| (Y_drive \otimes I - I \otimes Y_drive^T) rho_i>
                
                # grad_I = 0.5 scale A_I
                # grad_Q = 0.5 scale i A_Q
                
                # Note: <lambda| M |rho> = lambda^dagger M rho.
                
                term_I = np.vdot(lambda_i, (np.kron(X_drive, I3) - np.kron(I3, X_drive.T)) @ rho_i)
                term_Q = np.vdot(lambda_i, (np.kron(Y_drive, I3) - np.kron(I3, Y_drive.T)) @ rho_i)
                
                dJ_dI = 0.5 * scale * term_I
                dJ_dQ = 0.5 * scale * 1j * term_Q
                
                # We want real gradient.
                # dJ is real.
                # grad_I = Real(dJ_dI)
                # grad_Q = Real(dJ_dQ)
                
                grad_case[i*2] += weight * np.real(dJ_dI)
                grad_case[i*2+1] += weight * np.real(dJ_dQ)
                
                # Update lambda for previous step
                if i > 0:
                    lambda_i = E_list[i].conj().T @ lambda_i
                else:
                    lambda_i = np.zeros(9, dtype=complex) # Not used further
            
            # Note: The weight was applied per state? 
            # Score = sum_c (1/9) I_c. I_c = mean_s (1 - F_s).
            # So d Score = sum_c (1/9) sum_s (1/6) d(1-F_s)
            # My loop over states inside case loop needs to average over states.
            # I accumulated grad_case for one state. I need to divide by 6 (number of states).
            grad_case /= 6.0
            
            total_grad += grad_case
        
    return score, total_grad

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Start with seed
    current_pulse = seed_pulse()
    best_pulse = current_pulse.copy()
    best_score, _ = compute_score_and_gradient(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Flatten for optimizer
    p_flat = current_pulse.flatten()
    
    def objective_and_grad(p):
        if time.time() > deadline:
            return best_score, np.zeros_like(p)
        
        pulse_2d = p.reshape(N_STEPS, 2)
        # Enforce constraints? 
        # L-BFGS-B works better if we project back to feasible set.
        # But gradients are for the unconstrained space.
        # Strategy: Project p to feasible set, compute score/grad on projected pulse?
        # This is tricky because projection is non-smooth.
        # Alternative: Use penalty method or just rely on L-BFGS-B bounds?
        # L-BFGS-B supports bounds.
        # But slew constraint is hard.
        
        # Let's just project and compute score. Gradient will be approximate if projection changes pulse significantly.
        # However, if we start near feasible and stay near, it's okay.
        
        pulse_proj = enforce_constraints(pulse_2d)
        s, g = compute_score_and_gradient(pulse_proj)
        
        # If pulse_proj != pulse_2d, the gradient g is for pulse_proj.
        # We return g as approximation for p.
        return s, g.flatten()
    
    # Bounds for L-BFGS-B
    # Amplitude bound: I^2 + Q^2 <= 0.4^2. 
    # Individual bounds: I in [-0.4, 0.4], Q in [-0.4, 0.4] is a box approximation.
    # Slew constraint: |p[i+1] - p[i]| <= 0.12 * 0.5 = 0.06.
    # This is a complex polytope. 
    # We will use the box bounds for I and Q and rely on the projection in the objective.
    # This might cause issues if projection is aggressive.
    
    bounds = [(-0.4, 0.4) for _ in range(2 * N_STEPS)]
    
    try:
        result = minimize(
            objective_and_grad,
            p_flat,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-5, 'maxls': 50}
        )
        
        if result.fun < best_score:
            opt_pulse = enforce_constraints(result.x.reshape(N_STEPS, 2))
            opt_score, _ = compute_score_and_gradient(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception as e:
        pass
    
    # Local refinement with Nelder-Mead on a subset of parameters? 
    # Or just return best found.
    
    return best_pulse
# EVOLVE-BLOCK-END
