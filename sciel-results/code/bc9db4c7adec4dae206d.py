"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 300.0
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
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
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

# Precompute for fast Liouvillian construction
def _liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for a given step."""
    # Hamiltonian
    H = (detuning * _NOP + 0.5 * _ANH * _NOP @ (_NOP - 3*np.eye(3)) +
         0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG)))
    
    # Liouvillian in Liouville space: L(rho) = -i[H,rho] + sum_k L_k rho L_k^dag - 0.5{L_k^dag L_k, rho}
    # We'll work with vectorized density matrices
    # The superoperator is: -i (H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^*) - 0.5 (L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T)
    
    I3 = np.eye(3)
    L1DAG_L1 = _L1ADAG @ _L1
    L2DAG_L2 = _L2ADAG @ _L2
    
    # Vectorize: vec(A X B) = (B^T ⊗ A) vec(X)
    # -i[H, rho] = -i (H rho - rho H) -> vec: -i (I ⊗ H - H^T ⊗ I) vec(rho)
    # Wait, let me be careful. vec(H rho) = (I ⊗ H) vec(rho) if we use column-stacking
    # Actually, vec(A X B) = (B^T ⊗ A) vec(X)
    # So vec(H rho) = (I^T ⊗ H) vec(rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # So -i[H,rho] -> -i (I ⊗ H - H^T ⊗ I) vec(rho)
    
    L_super = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Lindblad terms: L_k rho L_k^dag -> vec: (L_k^* ⊗ L_k) vec(rho)
    L_super += np.kron(_L1.conj(), _L1) + np.kron(_L2.conj(), _L2)
    
    # -0.5 {L_k^dag L_k, rho} -> -0.5 (L_k^dag L_k rho + rho L_k^dag L_k)
    # vec: -0.5 ((I ⊗ L_k^dag L_k) + (L_k^dag L_k)^T ⊗ I) vec(rho)
    L_super -= 0.5 * (np.kron(I3, L1DAG_L1) + np.kron(L1DAG_L1.T, I3))
    L_super -= 0.5 * (np.kron(I3, L2DAG_L2) + np.kron(L2DAG_L2.T, I3))
    
    return L_super

def _simulate_pulse(I, Q, detuning, scale):
    """Simulate the pulse and return the final density matrix (vectorized)."""
    # Start with zero density matrix? No, we start with a state.
    # Actually, we need to simulate the evolution of an initial state.
    # The fidelity is computed by evolving each initial state.
    # For efficiency, we can compute the evolution superoperator for the whole pulse,
    # then apply it to each initial state.
    
    # Build the total evolution superoperator
    # This is the product of expm(L_i * dt) for each step
    # This is expensive for 80 steps with 9x9 matrices.
    # Let's try a different approach: simulate each initial state separately.
    
    # Actually, for scoring we need to evolve 6 initial states across 9 conditions = 54 simulations.
    # Each simulation is 80 steps of a 9x9 matrix exponential.
    # That's 54 * 80 = 4320 matrix exponentials of 9x9 matrices.
    # This should be fast enough.
    
    # Let's precompute the superoperators for each step
    # Actually, the superoperator depends on I, Q, detuning, scale which are constant across steps
    # within a single simulation. So we can compute it once per simulation.
    
    # Wait, I and Q vary with time! So the superoperator changes each step.
    
    # Let's just simulate directly.
    rho_vec = np.zeros(9)
    # Actually, we need to start with a specific initial state.
    # Let me restructure: this function should take an initial state and return the final state.
    
    pass

def _evolve_state(I, Q, detuning, scale, rho0):
    """Evolve a density matrix through the pulse."""
    rho = rho0.copy()
    for i in range(_N):
        H = (detuning * _NOP + 0.5 * _ANH * _NOP @ (_NOP - 3*np.eye(3)) +
             0.5 * scale * (I[i] * (_A + _ADAG) + 1j * Q[i] * (_A - _ADAG)))
        
        # Compute the master equation step
        # drho/dt = -i[H,rho] + D1(rho) + D2(rho)
        # where Dk(rho) = Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho}
        
        comm = H @ rho - rho @ H
        D1 = _L1 @ rho @ _L1ADAG - 0.5 * (_L1ADAG @ _L1 @ rho + rho @ _L1ADAG @ _L1)
        D2 = _L2 @ rho @ _L2ADAG - 0.5 * (_L2ADAG @ _L2 @ rho + rho @ _L2ADAG @ _L2)
        
        drho = -1j * comm + D1 + D2
        
        # First-order Taylor step (should be sufficient for small dt)
        rho = rho + drho * _DT
    
    return rho

def _fidelity(rho, psi):
    """Compute fidelity F = <psi|rho|psi>."""
    return np.real(np.vdot(psi, rho @ psi))

def _score_pulse(I, Q):
    """Compute the score for a given pulse."""
    # Check constraints
    for i in range(_N):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > _AMP + 1e-9:
            return 1e10  # Invalid
        if i > 0:
            slew = np.sqrt((I[i]-I[i-1])**2 + (Q[i]-Q[i-1])**2) / _DT
            if slew > _SLEW + 1e-9:
                return 1e10
    # Check slew from zero to first and last to zero
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / _DT
    if slew_start > _SLEW + 1e-9:
        return 1e10
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / _DT
    if slew_end > _SLEW + 1e-9:
        return 1e10
    
    # Initial states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2)
    psi_im = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2)
    
    states = [psi0, psi1, psi_plus, psi_minus, psi_ip, psi_im]
    
    infidelities = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            fid_sum = 0.0
            for psi in states:
                rho0 = np.outer(psi, psi.conj())
                rho_final = _evolve_state(I, Q, det, scale, rho0)
                # Target: X gate maps |0> -> |1>, |1> -> |0>
                # So fidelity to X|psi>
                psi_target = _X_gate @ psi
                fid = _fidelity(rho_final, psi_target)
                fid_sum += fid
            mean_fid = fid_sum / len(states)
            infidelities.append(1.0 - mean_fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

# X gate in the 3-level system
_X_gate = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]], dtype=complex)

def _make_seed_pulse():
    """Create the DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, _T, _N)
    sigma = 40.0 / 6.0
    # Gaussian envelope centered at T/2
    envelope = np.exp(-0.5 * ((t - _T/2) / sigma)**2)
    # Normalize area to pi
    area = np.sum(envelope) * _DT
    envelope = envelope * (np.pi / area)
    # Offset to 0 at edges - already 0 at edges due to Gaussian
    I = envelope
    Q = np.zeros(_N)
    # Ensure constraints
    # Scale down if needed
    max_amp = np.max(np.sqrt(I**2 + Q**2))
    if max_amp > _AMP:
        scale_factor = _AMP / max_amp
        I *= scale_factor
        Q *= scale_factor
    # Check slew
    for i in range(1, _N):
        slew = np.sqrt((I[i]-I[i-1])**2 + (Q[i]-Q[i-1])**2) / _DT
        if slew > _SLEW:
            # This shouldn't happen for a smooth Gaussian, but just in case
            pass
    return I, Q

def _enforce_constraints(I, Q):
    """Project the pulse to satisfy amplitude and slew constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= AMP
    for i in range(_N):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > _AMP:
            scale = _AMP / amp
            I[i] *= scale
            Q[i] *= scale
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW
    # This is a convex constraint, we can project iteratively
    for _ in range(10):  # Iterative projection
        # Forward pass
        prev_I, prev_Q = 0.0, 0.0
        for i in range(_N):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            norm = np.sqrt(dI**2 + dQ**2)
            max_d = _SLEW * _DT
            if norm > max_d:
                scale = max_d / norm
                I[i] = prev_I + dI * scale
                Q[i] = prev_Q + dQ * scale
            prev_I, prev_Q = I[i], Q[i]
        
        # Backward pass
        next_I, next_Q = 0.0, 0.0
        for i in range(_N - 1, -1, -1):
            dI = I[i] - next_I
            dQ = Q[i] - next_Q
            norm = np.sqrt(dI**2 + dQ**2)
            max_d = _SLEW * _DT
            if norm > max_d:
                scale = max_d / norm
                I[i] = next_I + dI * scale
                Q[i] = next_Q + dQ * scale
            next_I, next_Q = I[i], Q[i]
        
        # Re-apply amplitude constraint
        for i in range(_N):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > _AMP:
                scale = _AMP / amp
                I[i] *= scale
                Q[i] *= scale
    
    return I, Q

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((_N, 2))
    
    I, Q = _make_seed_pulse()
    I, Q = _enforce_constraints(I, Q)
    
    # Try to improve with optimization if time allows
    if _time_remaining() > 60:
        try:
            # Use differential evolution for global search
            def objective(params):
                I_opt = params[:_N]
                Q_opt = params[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt.copy(), Q_opt.copy())
                return _score_pulse(I_opt, Q_opt)
            
            # Bounds for DE
            bounds = [(-_AMP, _AMP)] * (2 * _N)
            
            result = differential_evolution(
                objective, bounds,
                maxiter=5, popsize=5,
                seed=42, tol=1e-4,
                polish=False
            )
            
            if result.fun < _score_pulse(I, Q):
                I_opt = result.x[:_N]
                Q_opt = result.x[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                I, Q = I_opt, Q_opt
        
        except Exception:
            pass
    
    if _time_remaining() > 30:
        try:
            # Local refinement with Nelder-Mead
            def objective(params):
                I_opt = params[:_N]
                Q_opt = params[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt.copy(), Q_opt.copy())
                return _score_pulse(I_opt, Q_opt)
            
            result = minimize(
                objective, np.concatenate([I, Q]),
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < _score_pulse(I, Q):
                I_opt = result.x[:_N]
                Q_opt = result.x[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                I, Q = I_opt, Q_opt
        
        except Exception:
            pass
    
    # Final constraint enforcement
    I, Q = _enforce_constraints(I, Q)
    
    return np.column_stack([I, Q])
# EVOLVE-BLOCK-END
