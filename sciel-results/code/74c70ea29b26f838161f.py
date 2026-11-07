"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
import sys

_DEADLINE = 310.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute Liouvillian components
# H = detuning*n + 0.5*anh*n*(n-1) + 0.5*scale*(I*(a+a^dag) + i*Q*(a-a^dag))
# We'll build the Liouvillian for each step

def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for a single step."""
    # Hamiltonian
    # n = a^dag a = diag(0,1,2)
    # n*(n-1) = diag(0,0,2)
    H_det = detuning * _NOP  # detuning * n
    H_anh = 0.5 * _ANH * (np.diag([0.0, 0.0, 2.0]))  # 0.5 * anh * n*(n-1)
    
    # Drive term: 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    aa = _A + _ADAG
    a_minus_adag = _A - _ADAG
    H_drive = 0.5 * scale * (I * aa + 1j * Q * a_minus_adag)
    
    H = H_det + H_anh + H_drive
    
    # Liouvillian: L(rho) = -i*[H, rho] + sum_k (L_k rho L_k^dag - 0.5*(L_k^dag L_k rho + rho L_k^dag L_k))
    # Vectorize: vec(L(rho)) = L_vec * vec(rho)
    # For 3x3 matrices, we work with 9x9 superoperators
    
    # -i*[H, rho] -> -i*(H⊗I - I⊗H^T) in vec form
    # Actually, vec(A rho B) = (B^T ⊗ A) vec(rho)
    # So vec([H, rho]) = vec(H rho - rho H) = (I ⊗ H) vec(rho) - (H^T ⊗ I) vec(rho)
    # = (I ⊗ H - H^T ⊗ I) vec(rho)
    # -i*[H, rho] -> -1j * (I ⊗ H - H^T ⊗ I)
    
    n_dim = 3
    I3 = np.eye(n_dim)
    
    L_comm = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative terms
    # For L_k: L_k rho L_k^dag - 0.5*(L_k^dag L_k rho + rho L_k^dag L_k)
    # vec(L_k rho L_k^dag) = (L_k^T ⊗ L_k) vec(rho)
    # vec(L_k^dag L_k rho) = (I ⊗ L_k^dag L_k) vec(rho)
    # vec(rho L_k^dag L_k) = ((L_k^dag L_k)^T ⊗ I) vec(rho)
    
    L_diss = np.zeros((n_dim*n_dim, n_dim*n_dim), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkdag = Lk.conj().T
        Lkdag_Lk = Lkdag @ Lk
        L_diss += np.kron(Lk.T, Lk) - 0.5 * np.kron(I3, Lkdag_Lk) - 0.5 * np.kron(Lkdag_Lk.T, I3)
    
    L_total = L_comm + L_diss
    return L_total

def _simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(_N):
        L = _build_liouvillian(detuning, scale, pulse_I[i], pulse_Q[i])
        L_exp = expm(L * _DT)
        rho_vec = rho.flatten()
        rho_vec_new = L_exp @ rho_vec
        rho = rho_vec_new.reshape((3, 3))
    
    return rho

def _fidelity_to_X(rho):
    """Compute fidelity to X gate: F = <0|rho|1> + <1|rho|0> should be 1 for X."""
    # X gate maps |0> -> |1>, |1> -> |0>
    # So after X, rho should have rho[0,1] = rho[1,0] = 1 (in the computational basis)
    # Fidelity = Re(rho[0,1]) for a pure state X
    # More generally, we want the overlap with the target state
    # Target: X|0> = |1>, X|1> = |0>
    # For input |0>, target is |1>, fidelity = <1|rho|1> = rho[1,1]
    # For input |1>, target is |0>, fidelity = <0|rho|0> = rho[0,0]
    # For superpositions, we need to be more careful
    
    # Actually, the task says "mean fidelity to X over |0>, |1>, (|0>+-|1>)/sqrt(2), (|0>+-i|1>)/sqrt(2)"
    # This means we simulate each of these 4 input states and compute fidelity to X|state>
    
    # But our simulation starts from |0>. We need to simulate from each input state.
    # Let me re-read: "infidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    
    # So we need 4 separate simulations with different initial states.
    pass

def _compute_score(pulse_I, pulse_Q):
    """Compute the score for a given pulse."""
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    # 4 input states
    # |0> = [1,0,0], |1> = [0,1,0], (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # Wait, the task says 4 states: |0>, |1>, (|0>+-|1>)/sqrt(2), (|0>+-i|1>)/sqrt(2)
    # That's actually 6 states? Let me re-read: "over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # The +- means both + and -, so that's 2 + 2 + 2 = 6 states? Or is it 4?
    # Looking at it again: "|0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # I think this means: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # That's 6 states. But the task says "mean fidelity to X over" these states.
    
    # Let me assume it's 4 states: |0>, |1>, |+> = (|0>+|1>)/sqrt(2), |+i> = (|0>+i|1>)/sqrt(2)
    # Actually, the +- notation likely means both signs, so 6 states total.
    
    input_states = [
        np.array([1.0, 0.0, 0.0], dtype=complex),  # |0>
        np.array([0.0, 1.0, 0.0], dtype=complex),  # |1>
        np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1.0, 1.0j, 0.0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1.0, -1.0j, 0.0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # Target states after X
    target_states = [
        np.array([0.0, 1.0, 0.0], dtype=complex),  # X|0> = |1>
        np.array([1.0, 0.0, 0.0], dtype=complex),  # X|1> = |0>
        np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2),  # X|+> = |+>
        np.array([-1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2),  # X|-> = |->  (actually X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2))
        np.array([1.0, -1.0j, 0.0], dtype=complex) / np.sqrt(2),  # X(|0>+i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2) = (-i)(|0>+i|1>)/sqrt(2)
        np.array([1.0, 1.0j, 0.0], dtype=complex) / np.sqrt(2),   # X(|0>-i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = (i)(|0>-i|1>)/sqrt(2)
    ]
    
    # Actually, fidelity for pure states is |<psi_target|psi_final>|^2
    # For the superposition states, the global phase doesn't matter for fidelity
    
    total_fidelity = 0.0
    count = 0
    
    for det in detunings:
        for scale in scales:
            for psi_in, psi_target in zip(input_states, target_states):
                rho_in = np.outer(psi_in, psi_in.conj())
                
                # Simulate
                rho = rho_in.copy()
                for i in range(_N):
                    L = _build_liouvillian(det, scale, pulse_I[i], pulse_Q[i])
                    L_exp = expm(L * _DT)
                    rho_vec = rho.flatten()
                    rho_vec_new = L_exp @ rho_vec
                    rho = rho_vec_new.reshape((3, 3))
                
                # Fidelity = |<psi_target|rho|psi_target>| for pure target
                # = Tr(rho * |psi_target><psi_target|)
                proj = np.outer(psi_target, psi_target.conj())
                fid = np.real(np.trace(rho @ proj))
                total_fidelity += fid
                count += 1
    
    mean_fidelity = total_fidelity / count
    infidelity = 1.0 - mean_fidelity
    return infidelity

def _make_seed_pulse(sigma_ratio=1.0):
    """Create a DRAG-style Gaussian seed pulse."""
    t = np.arange(_N) * _DT
    sigma = (40.0 / 6.0) * sigma_ratio  # sigma = 40/6 ns, scaled
    
    # Gaussian with area pi
    # Integral of A * exp(-t^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi) = pi
    # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A = np.sqrt(np.pi / 2.0) / sigma
    
    # Offset to 0 at edges
    center = _N // 2
    t_centered = t - center * _DT
    
    I = A * np.exp(-t_centered**2 / (2.0 * sigma**2))
    
    # DRAG: add derivative component
    # dI/dt = -A * t_centered / sigma^2 * exp(-t_centered^2/(2*sigma^2))
    # Q = lambda * dI/dt
    # For transmon, lambda = -anh/(2*omega_01) approximately, but let's use a small value
    lambda_drag = 0.1
    Q = lambda_drag * (-A * t_centered / sigma**2 * np.exp(-t_centered**2 / (2.0 * sigma**2)))
    
    # Enforce constraints
    I, Q = _enforce_constraints(I, Q)
    
    return I, Q

def _enforce_constraints(I, Q):
    """Enforce amplitude and slew rate constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > _AMP
    if np.any(mask):
        scale = _AMP / amp[mask]
        I[mask] *= scale
        Q[mask] *= scale
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    # p[i] = [I[i], Q[i]]
    # |p[1] - p[0]| / dt <= 0.12, where p[0] = [0, 0] (before pulse)
    # |p[i+1] - p[i]| / dt <= 0.12 for i = 0..78
    # |p[79] - [0,0]| / dt <= 0.12 (after pulse)
    
    max_delta = _SLEW * _DT  # 0.12 * 0.5 = 0.06
    
    # From zero to step 0
    delta = np.sqrt(I[0]**2 + Q[0]**2)
    if delta > max_delta:
        scale = max_delta / delta
        I[0] *= scale
        Q[0] *= scale
    
    # Between steps
    for i in range(_N - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        delta = np.sqrt(dI**2 + dQ**2)
        if delta > max_delta:
            scale = max_delta / delta
            I[i+1] = I[i] + dI * scale
            Q[i+1] = Q[i] + dQ * scale
    
    # From last step to zero
    delta = np.sqrt(I[-1]**2 + Q[-1]**2)
    if delta > max_delta:
        scale = max_delta / delta
        I[-1] *= scale
        Q[-1] *= scale
    
    return I, Q

def _optimize_pulse(I0, Q0):
    """Optimize the pulse using Nelder-Mead."""
    def objective(params):
        I = params[:_N]
        Q = params[_N:]
        I, Q = _enforce_constraints(I, Q)
        return _compute_score(I, Q)
    
    x0 = np.concatenate([I0, Q0])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
    
    I_opt = result.x[:_N]
    Q_opt = result.x[_N:]
    I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
    
    return I_opt, Q_opt, result.fun

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((_N, 2))
    
    # Create seed pulse
    I, Q = _make_seed_pulse()
    
    best_I, best_Q = I.copy(), Q.copy()
    best_score = _compute_score(I, Q)
    
    # Try optimization if time permits
    if _time_remaining() > 100.0:
        try:
            I_opt, Q_opt, score_opt = _optimize_pulse(I, Q)
            if score_opt < best_score:
                best_I, best_Q = I_opt, Q_opt
                best_score = score_opt
        except Exception:
            pass
    
    # Return as (80, 2) array
    return np.column_stack([best_I, best_Q])

# EVOLVE-BLOCK-END
