"""Robust control pulses for a three-level transmon - differentiable ODE with adjoint gradients."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm, expm_frechet
import time

# Hardware constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
T2 = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
DIM = 3

# Precompute operators
A_OP = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
A_DAG = A_OP.conj().T
N_OP = A_DAG @ A_OP  # diag(0,1,2)
N2_OP = N_OP @ (N_OP - np.eye(DIM))  # n(n-1)
I3 = np.eye(DIM)
I9 = np.eye(DIM * DIM)

# Lindblad terms (constant)
L1 = A_OP / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T2)
L1dag = L1.conj().T
L2dag = L2.conj().T
L1dagL1 = L1dag @ L1
L2dagL2 = L2dag @ L2

# Precompute constant parts of Liouvillian superoperator
# L_super = -i(H⊗I - I⊗H†) + sum_k (L_k⊗L_k* - 0.5(L_k†L_k⊗I + I⊗L_k†L_k))
# The constant part (independent of H):
CONST_LIND = (np.kron(L1, L1.conj().T) - 0.5 * (np.kron(L1dagL1, I3) + np.kron(I3, L1dagL1))
             + np.kron(L2, L2.conj().T) - 0.5 * (np.kron(L2dagL2, I3) + np.kron(I3, L2dagL2)))

# ∂L/∂I: -i(∂H/∂I ⊗ I - I ⊗ ∂H/∂I†)
# H_drive = 0.5*scale*(I*(A+A†) + iQ*(A-A†))
# ∂H/∂I = 0.5*scale*(A+A†)
# ∂H/∂Q = 0.5*scale*i*(A-A†)


def build_liouvillian(detuning, scale, I_val, Q_val):
    """Build the full Liouvillian superoperator (9x9 complex)."""
    H = detuning * N_OP + 0.5 * ANHARMONICITY * N2_OP
    H += 0.5 * scale * (I_val * (A_OP + A_DAG) + 1j * Q_val * (A_OP - A_DAG))
    H_term = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    return H_term + CONST_LIND


def dliouvillian_dI(detuning, scale, I_val, Q_val):
    """∂L/∂I superoperator."""
    dH_dI = 0.5 * scale * (A_OP + A_DAG)
    return -1j * (np.kron(dH_dI, I3) - np.kron(I3, dH_dI.conj().T))


def dliouvillian_dQ(detuning, scale, I_val, Q_val):
    """∂L/∂Q superoperator."""
    dH_dQ = 0.5 * scale * 1j * (A_OP - A_DAG)
    return -1j * (np.kron(dH_dQ, I3) - np.kron(I3, dH_dQ.conj().T))


# Target states
PSI0 = np.array([1, 0, 0], dtype=complex)
PSI1 = np.array([0, 1, 0], dtype=complex)
PSI_PLUS = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
PSI_IPLUS = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
PSI_IMINUS = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
TARGET_STATES = [PSI0, PSI1, PSI_PLUS, PSI_MINUS, PSI_IPLUS, PSI_IMINUS]

DETUNINGS = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]


def vec_rho(rho):
    """Vectorize density matrix (column-major)."""
    return rho.reshape(DIM * DIM)


def unvec_rho(v):
    """Unvectorize."""
    return v.reshape(DIM, DIM)


def fidelity_pure(rho, psi):
    """Fidelity F = psi† rho psi (real for Hermitian rho)."""
    return np.real(np.vdot(psi, rho @ psi))


def forward_evolve(pulse_I, pulse_Q, detuning, scale):
    """
    Forward evolve density matrix through the pulse.
    Returns array of shape (N_STEPS+1, DIM, DIM) with rho[0] = |0><0|.
    """
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[0, 0] = 1.0
    
    rhos = [rho.copy()]
    for k in range(N_STEPS):
        L_super = build_liouvillian(detuning, scale, pulse_I[k], pulse_Q[k])
        U = expm(DT * L_super)
        rho = (U @ vec_rho(rho)).reshape(DIM, DIM)
        rhos.append(rho.copy())
    
    return np.array(rhos)


def compute_case_infidelity(rho_final):
    """Compute mean infidelity over 6 target states."""
    fids = []
    for psi in TARGET_STATES:
        f = fidelity_pure(rho_final, psi)
        f = np.clip(f, 0, 1)
        fids.append(1 - f)
    return np.mean(fids)


def score_and_gradient(pulse_I, pulse_Q):
    """
    Compute score (mean + max over 9 cases) and gradient w.r.t. pulse.
    Uses adjoint method for exact gradients.
    """
    n_cases = 9
    case_infid = np.zeros(n_cases)
    case_max_fid = np.zeros(n_cases)
    
    # Store forward states for adjoint
    all_rhos = []
    case_idx = 0
    for det in DETUNINGS:
        for scl in SCALES:
            rhos = forward_evolve(pulse_I, pulse_Q, det, scl)
            all_rhos.append(rhos)
            infid = compute_case_infidelity(rhos[-1])
            case_infid[case_idx] = infid
            case_idx += 1
    
    mean_infid = np.mean(case_infid)
    max_infid = np.max(case_infid)
    score = mean_infid + max_infid
    
    # Adjoint: compute gradient
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    case_idx = 0
    for det in DETUNINGS:
        for scl in SCALES:
            rhos = all_rhos[case_idx]
            
            # d(score)/d(rho_N) for this case
            # score = mean + max, so d(score)/d(case_infid_c) = 1/9 + (1 if c==argmax else 0)
            is_max = (case_idx == np.argmax(case_infid))
            weight = 1.0 / 9.0 + (1.0 if is_max else 0.0)
            
            # d(case_infid)/d(rho_N)
            # case_infid = mean_j (1 - Re(psi_j† rho_N psi_j))
            # d/d(rho_N) = -mean_j Re(psi_j psi_j†) in appropriate sense
            # In vectorized form: dF/dvec(rho) where F = Re(psi† rho psi)
            # F = Re(sum_{ij} psi_i* rho_{ij} psi_j)
            # dF/drho_{ij} = Re(psi_i* psi_j)
            # So dF/dvec(rho) = vec(Re(psi psi†)) ... need to be careful with ordering
            
            # Actually: F = psi† rho psi = sum_{ij} psi_i* rho_{ij} psi_j
            # dF/drho_{ij} = psi_i* psi_j
            # In column-major vec: index (i,j) -> j*DIM + i
            # So dF/dvec(rho)[j*DIM+i] = psi_i* psi_j = (psi psi†)_{j,i}^* ... 
            # Let me just compute it directly:
            # dF/dvec(rho) = conj(psi) ⊗ psi (Kronecker) ... let me verify
            # vec(psi psi†) has entries (psi psi†)_{ij} = psi_i psi_j*
            # dF/drho_{ij} = psi_i* psi_j
            # So dF/dvec(rho) = conj(vec(psi psi†)) = vec(psi psi†).conj()
            
            d_infid_drho = np.zeros(DIM * DIM, dtype=complex)
            for psi in TARGET_STATES:
                # d(1-F)/dvec(rho) = -dF/dvec(rho) = -conj(vec(psi psi†))
                dF = np.conj(np.kron(psi, psi))  # This gives vec(psi psi†) conjugated? Let me check
                # np.kron(psi, psi)[j*DIM+i] = psi[j]*psi[i] = psi_j * psi_i
                # We want dF/drho_{ij} = psi_i* psi_j, so in vec index j*DIM+i: psi_i* psi_j
                # np.kron(psi.conj(), psi)[j*DIM+i] = psi.conj()[j]*psi[i] = psi_j* psi_i -- no
                # Let me just build it directly
                dF_vec = np.zeros(DIM * DIM, dtype=complex)
                for i in range(DIM):
                    for j in range(DIM):
                        dF_vec[j * DIM + i] = np.conj(psi[i]) * psi[j]
                d_infid_drho += -dF_vec / len(TARGET_STATES)
            
            # Adjoint initial condition: lambda_N = weight * d_infid_drho
            lambda_vec = weight * d_infid_drho
            
            # Run adjoint backward
            for k in range(N_STEPS - 1, -1, -1):
                L_super = build_liouvillian(det, scl, pulse_I[k], pulse_Q[k])
                U = expm(DT * L_super)
                
                # Gradient w.r.t. H_k (and thus I_k, Q_k)
                # d(f)/d(H_k) = lambda_k† * d(U_k rho_k)/dH_k
                # = lambda_k† * expm_frechet(DT*L, DT*dL/dH_k) * rho_k
                # But we need dL/dI and dL/dQ separately
                
                rho_k = vec_rho(rhos[k])
                
                dL_dI = dliouvillian_dI(det, scl, pulse_I[k], pulse_Q[k])
                dL_dQ = dliouvillian_dQ(det, scl, pulse_I[k], pulse_Q[k])
                
                # Fréchet derivative: expm(A, E) = derivative of expm in direction E
                # d(U rho)/dI = expm_frechet(DT*L_super, DT*dL_dI) @ rho_k
                dU_rho_dI = expm_frechet(DT * L_super, DT * dL_dI) @ rho_k
                dU_rho_dQ = expm_frechet(DT * L_super, DT * dL_dQ) @ rho_k
                
                # Gradient contribution
                grad_I[k] += np.real(np.vdot(lambda_vec, dU_rho_dI))
                grad_Q[k] += np.real(np.vdot(lambda_vec, dU_rho_dQ))
                
                # Update adjoint: lambda_k = U_k† lambda_{k+1}
                # In vectorized form with column-major: U acts as U @ vec(rho)
                # The adjoint is: lambda_k = U† @ lambda_{k+1} (conjugate transpose)
                lambda_vec = U.conj().T @ lambda_vec
            
            case_idx += 1
    
    return score, np.concatenate([grad_I, grad_Q])


def check_constraints(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    amplitudes = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    prev_I, prev_Q = 0.0, 0.0
    for i in range(N_STEPS):
        dI = pulse_I[i] - prev_I
        dQ = pulse_Q[i] - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
        prev_I, prev_Q = pulse_I[i], pulse_Q[i]
    
    dI = -prev_I
    dQ = -prev_Q
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-10:
        return False
    
    return True


def drag_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    
    A = np.sqrt(np.pi / 2) / sigma
    I = A * np.exp(-t**2 / (2 * sigma**2))
    I = I - I[0]
    I = np.maximum(I, 0)
    
    max_amp = np.max(np.sqrt(I**2))
    if max_amp > AMPLITUDE_BOUND:
        I = I * (AMPLITUDE_BOUND / max_amp)
    
    for _ in range(10):
        if check_constraints(I, np.zeros_like(I)):
            break
        I = I * 0.9
    
    Q = np.zeros_like(I)
    return I, Q


def _compute_pulse():
    """Compute the optimized pulse."""
    I_seed, Q_seed = drag_seed()
    x0 = np.concatenate([I_seed, Q_seed])
    
    # Try gradient-based optimization
    try:
        def objective(x):
            I = x[:N_STEPS]
            Q = x[N_STEPS:]
            
            # Soft penalty for constraint violations
            penalty = 0.0
            amplitudes = np.sqrt(I**2 + Q**2)
            viol_amp = np.maximum(amplitudes - AMPLITUDE_BOUND, 0)
            penalty += 100.0 * np.sum(viol_amp**2)
            
            prev_I, prev_Q = 0.0, 0.0
            slew_viol = []
            for i in range(N_STEPS):
                dI = I[i] - prev_I
                dQ = Q[i] - prev_Q
                slew = np.sqrt(dI**2 + dQ**2) / DT
                slew_viol.append(max(0, slew - SLEW_BOUND))
                prev_I, prev_Q = I[i], Q[i]
            dI = -prev_I
            dQ = -prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            slew_viol.append(max(0, slew - SLEW_BOUND))
            penalty += 100.0 * np.sum(np.array(slew_viol)**2)
            
            score, _ = score_and_gradient(I, Q)
            return score + penalty
        
        def gradient(x):
            I = x[:N_STEPS]
            Q = x[N_STEPS:]
            
            score, grad = score_and_gradient(I, Q)
            
            # Add gradient of penalty
            grad_pen = np.zeros_like(x)
            amplitudes = np.sqrt(I**2 + Q**2)
            for i in range(N_STEPS):
                if amplitudes[i] > AMPLITUDE_BOUND and amplitudes[i] > 1e-12:
                    factor = 200.0 * (amplitudes[i] - AMPLITUDE_BOUND) / amplitudes[i]
                    grad_pen[i] += factor * I[i]
                    grad_pen[N_STEPS + i] += factor * Q[i]
            
            prev_I, prev_Q = 0.0, 0.0
            for i in range(N_STEPS):
                dI = I[i] - prev_I
                dQ = Q[i] - prev_Q
                slew = np.sqrt(dI**2 + dQ**2) / DT
                if slew > SLEW_BOUND:
                    factor = 200.0 * (slew - SLEW_BOUND) / (slew * DT) if slew > 1e-12 else 0
                    grad_pen[i] += factor * dI
                    grad_pen[N_STEPS + i] += factor * dQ
                    grad_pen[i-1] -= factor * dI if i > 0 else 0
                    grad_pen[N_STEPS + i - 1] -= factor * dQ if i > 0 else 0
                prev_I, prev_Q = I[i], Q[i]
            
            return grad + grad_pen
        
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 50, 'gtol': 1e-5, 'maxls': 50})
        
        I_opt = result.x[:N_STEPS]
        Q_opt = result.x[N_STEPS:]
        
        if check_constraints(I_opt, Q_opt):
            return I_opt, Q_opt
        else:
            # Try to project back
            scale_factor = AMPLITUDE_BOUND / max(1e-12, np.max(np.sqrt(I_opt**2 + Q_opt**2)))
            I_opt *= scale_factor
            Q_opt *= scale_factor
            if check_constraints(I_opt, Q_opt):
                return I_opt, Q_opt
            
    except Exception as e:
        pass
    
    return I_seed, Q_seed


_I_cached, _Q_cached = _compute_pulse()


def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    return np.column_stack([_I_cached, _Q_cached])
