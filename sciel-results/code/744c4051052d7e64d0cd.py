"""
Robust control pulses for a three-level transmon.
Implements a DRAG-style seed with optimization via Legendre polynomial basis.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# Constants
DT = 0.5  # ns
N_STEPS = 80  # 40 ns total
T_TOTAL = N_STEPS * DT  # 40 ns
ANHARMONICITY = -2 * np.pi * 0.25  # GHz
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = 2 * np.pi * 0.002
SCALE_MIN = 0.97
SCALE_MAX = 1.03
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

def generate_seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    # Gaussian envelope
    env = np.exp(-t**2 / (2 * sigma**2))
    # Offset to make area = pi
    area = np.trapz(env, t)
    env = env * (np.pi / area)
    # Ensure zero at edges
    env[0] = 0.0
    env[-1] = 0.0
    # I component (real), Q component (imaginary) for DRAG
    I = env
    Q = np.zeros_like(I)
    # DRAG term: proportional to derivative of envelope
    denv = np.gradient(env, DT)
    Q = -0.1 * denv  # Small DRAG coefficient
    return np.column_stack([I, Q])

def hamiltonian(t, y, I_val, Q_val, detuning, scale):
    """
    Compute the Hamiltonian for the 3-level system.
    y is not used (time-independent within step).
    """
    # Ladder operators for 3 levels
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    
    n = adag @ a  # Number operator
    
    # Free Hamiltonian
    H_free = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3))
    
    # Drive term
    H_drive = 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    
    return H_free + H_drive

def lindblad_operators():
    """Create Lindblad operators for T1 and Tphi decay."""
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    
    L1 = a / np.sqrt(T1)  # T1 relaxation
    L2 = n * np.sqrt(2.0 / Tphi)  # Dephasing
    
    return [L1, L2]

def liouvillian(H, L_ops):
    """
    Compute the Liouvillian superoperator.
    H is the Hamiltonian, L_ops is a list of Lindblad operators.
    Returns the Liouvillian matrix acting on the density matrix (vectorized).
    """
    dim = H.shape[0]
    # Commutator term: -i [H, rho]
    L_comm = np.zeros((dim*dim, dim*dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # (H rho - rho H)_{ij} = sum_k (H_{ik} rho_{kj} - rho_{ik} H_{kj})
                    pass  # We'll use a more efficient approach
    
    # More efficient: use Kronecker products
    I_dim = np.eye(dim, dtype=complex)
    
    # -i (H ⊗ I - I ⊗ H^T)
    L_comm = -1j * (np.kron(H, I_dim) - np.kron(I_dim, H.T))
    
    # Dissipative terms: sum_l (L_l rho L_l^† - 0.5 {L_l^† L_l, rho})
    L_diss = np.zeros((dim*dim, dim*dim), dtype=complex)
    for L in L_ops:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        
        # L rho L^†
        term1 = np.kron(L, L.conj().T)
        
        # -0.5 (L^† L rho + rho L^† L)
        term2 = -0.5 * (np.kron(LdagL, I_dim) + np.kron(I_dim, LdagL.T))
        
        L_diss += term1 + term2
    
    return L_comm + L_diss

def evolve_density_matrix(rho0, I_vals, Q_vals, detuning, scale):
    """
    Evolve the density matrix through all time steps.
    rho0: initial density matrix (3x3)
    I_vals, Q_vals: arrays of length N_STEPS
    Returns final density matrix
    """
    rho = rho0.copy()
    L_ops = lindblad_operators()
    
    for i in range(N_STEPS):
        H = hamiltonian(0, None, I_vals[i], Q_vals[i], detuning, scale)
        L = liouvillian(H, L_ops)
        
        # Vectorize rho
        rho_vec = rho.flatten()
        
        # Exponentiate: rho_{i+1} = exp(L * DT) rho_i
        # Use matrix exponential
        from scipy.linalg import expm
        L_exp = expm(L * DT)
        rho_vec = L_exp @ rho_vec
        
        rho = rho_vec.reshape((3, 3))
    
    return rho

def fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure state."""
    # Fidelity = <psi| rho |psi>
    return np.real(np.vdot(target_state, rho @ target_state))

def compute_score(I_vals, Q_vals):
    """
    Compute the score for given pulse.
    Score = mean + max of infidelities over 9 cases.
    """
    target_states = []
    # |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j*psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j*psi1) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    detunings = [-DETUNING_BOUNDS, 0.0, DETUNING_BOUNDS]
    scales = [SCALE_MIN, 1.0, SCALE_MAX]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            # For each target state, compute infidelity
            fid_sum = 0.0
            for psi in target_states:
                rho0 = np.outer(psi, psi.conj())
                rho_final = evolve_density_matrix(rho0, I_vals, Q_vals, det, scale)
                # Target is X gate: |0> -> |1>, |1> -> -|0> (or |0> -> |1> with phase)
                # X gate: X|0> = |1>, X|1> = |0>
                # So target states after X gate:
                # |0> -> |1>
                # |1> -> |0>
                # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2)
                # etc.
                # Fidelity to X|psi>
                psi_x = np.zeros(3, dtype=complex)
                if np.allclose(psi, psi0):
                    psi_x = psi1
                elif np.allclose(psi, psi1):
                    psi_x = psi0
                elif np.allclose(psi, psi_plus):
                    psi_x = psi_plus  # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
                elif np.allclose(psi, psi_minus):
                    psi_x = psi_minus  # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -psi_minus
                    # Actually X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
                    # So fidelity to -psi_minus is same as to psi_minus
                elif np.allclose(psi, psi_i_plus):
                    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
                    psi_x = (psi1 + 1j*psi0) / np.sqrt(2)
                elif np.allclose(psi, psi_i_minus):
                    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
                    psi_x = (psi1 - 1j*psi0) / np.sqrt(2)
                
                fid = fidelity(rho_final, psi_x)
                fid_sum += fid
            
            mean_fid = fid_sum / len(target_states)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def check_constraints(I_vals, Q_vals):
    """Check amplitude and slew constraints."""
    # Amplitude bound
    amplitudes = np.sqrt(I_vals**2 + Q_vals**2)
    if np.any(amplitudes > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew bound
    # From zero to step 1
    p0 = np.array([0.0, 0.0])
    p1 = np.array([I_vals[0], Q_vals[0]])
    if np.linalg.norm(p1 - p0) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        pi = np.array([I_vals[i], Q_vals[i]])
        pi1 = np.array([I_vals[i+1], Q_vals[i+1]])
        if np.linalg.norm(pi1 - pi) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    plast = np.array([I_vals[-1], Q_vals[-1]])
    if np.linalg.norm(p0 - plast) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def legendre_basis(t, n_terms=6):
    """Generate Legendre polynomial basis on [0, 1]."""
    from numpy.polynomial.legendre import legvander
    t_norm = t / T_TOTAL  # Normalize to [0, 1]
    return legvander(t_norm, n_terms - 1)

def optimize_pulse():
    """Optimize the pulse using Legendre polynomial basis."""
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second limit
    
    # Generate seed pulse
    seed_pulse = generate_seed_pulse()
    I_seed = seed_pulse[:, 0]
    Q_seed = seed_pulse[:, 1]
    
    # Initial score
    best_I = I_seed.copy()
    best_Q = Q_seed.copy()
    best_score = compute_score(best_I, best_Q)
    
    # Legendre basis
    t = np.arange(N_STEPS) * DT
    n_terms = 6
    basis = legendre_basis(t, n_terms)
    
    # Fit seed to Legendre basis
    I_coeffs = np.zeros(n_terms)
    Q_coeffs = np.zeros(n_terms)
    for k in range(n_terms):
        I_coeffs[k] = np.sum(I_seed * basis[:, k]) / np.sum(basis[:, k]**2)
        Q_coeffs[k] = np.sum(Q_seed * basis[:, k]) / np.sum(basis[:, k]**2)
    
    def params_to_pulse(params):
        """Convert Legendre coefficients to pulse."""
        I_vals = basis @ params[:n_terms]
        Q_vals = basis @ params[n_terms:]
        return I_vals, Q_vals
    
    def objective(params):
        """Objective function for optimization."""
        I_vals, Q_vals = params_to_pulse(params)
        
        # Check constraints
        if not check_constraints(I_vals, Q_vals):
            return 1e6  # Penalty for constraint violation
        
        return compute_score(I_vals, Q_vals)
    
    # Initial parameters
    x0 = np.concatenate([I_coeffs, Q_coeffs])
    
    # Bounds for coefficients (rough)
    bounds = [(-1.0, 1.0)] * (2 * n_terms)
    
    # Optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.fun < best_score:
            I_opt, Q_opt = params_to_pulse(result.x)
            if check_constraints(I_opt, Q_opt):
                best_I = I_opt
                best_Q = Q_opt
                best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        return best_I, best_Q
    
    # Try Nelder-Mead for refinement
    try:
        result2 = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result2.fun < best_score:
            I_opt, Q_opt = params_to_pulse(result2.x)
            if check_constraints(I_opt, Q_opt):
                best_I = I_opt
                best_Q = Q_opt
                best_score = result2.fun
    except Exception:
        pass
    
    return best_I, best_Q

def pulse(target_gate, hardware_model):
    """
    Main entry point. Returns (80, 2) array of I and Q amplitudes.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate supported")
    
    # Optimize the pulse
    I_vals, Q_vals = optimize_pulse()
    
    return np.column_stack([I_vals, Q_vals])

# EVOLVE-BLOCK-END
