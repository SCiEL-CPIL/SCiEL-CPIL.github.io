"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses a DRAG-style Gaussian pulse with optimized parameters for robustness.
    """
    if target_gate != 'X_pi':
        # For now, only handle X_pi
        return np.zeros((80, 2))
    
    # Time parameters
    dt = 0.5  # ns
    n_steps = 80
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002e-3, 2 * np.pi * 0.002e-3)
    scale_bounds = (0.97, 1.03)
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # DRAG pulse parameters
    # Gaussian width: sigma = 40/6 ns ≈ 6.667 ns
    sigma = 40.0 / 6.0  # ns
    t_center = 20.0  # ns, center of 40 ns window
    
    # Start with the seed DRAG pulse
    # I(t) = A * exp(-0.5 * ((t - t_center) / sigma)^2)
    # Q(t) = -alpha * dI/dt = -alpha * A * (-(t - t_center) / sigma^2) * exp(-0.5 * ((t - t_center) / sigma)^2)
    #      = alpha * A * (t - t_center) / sigma^2 * exp(-0.5 * ((t - t_center) / sigma)^2)
    
    t = np.arange(n_steps) * dt  # 0 to 39.5 ns
    
    # Optimize amplitude A and alpha for robustness
    # The area of the Gaussian should be pi for an X-pi gate
    # Area = A * sigma * sqrt(2*pi) = pi => A = 1 / (sigma * sqrt(2))
    
    # Let's use a grid search over a few parameters
    # We'll optimize A (amplitude) and alpha (DRAG coefficient)
    
    best_score = np.inf
    best_I = None
    best_Q = None
    
    # Grid of A values around the nominal
    A_nominal = 1.0 / (sigma * np.sqrt(2.0))
    A_values = np.linspace(0.95 * A_nominal, 1.05 * A_nominal, 11)
    
    # Grid of alpha values
    alpha_values = np.linspace(0.0, 0.15, 8)
    
    # Deadline guard
    start_time = time.time()
    deadline = 320.0  # seconds, with safety margin
    
    for A in A_values:
        if time.time() > deadline:
            break
        for alpha in alpha_values:
            if time.time() > deadline:
                break
            
            # Generate pulse
            I = A * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
            Q = alpha * A * (t - t_center) / sigma**2 * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
            
            # Check amplitude constraint
            if np.any(np.sqrt(I**2 + Q**2) > amp_bound):
                continue
            
            # Check slew rate constraint
            # ||p[i+1] - p[i]|| / dt <= 0.12
            # Including from zero into step 1 and from last step to zero
            p = np.column_stack([I, Q])
            # Add zero at start and end
            p_ext = np.vstack([np.zeros(2), p, np.zeros(2)])
            diffs = np.diff(p_ext, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            if np.any(norms / dt > slew_bound):
                continue
            
            # Score this pulse
            score = _score_pulse(I, Q, dt, n_steps, anharmonicity, T1, Tphi)
            
            if score < best_score:
                best_score = score
                best_I = I.copy()
                best_Q = Q.copy()
    
    # If no valid pulse found, return the nominal DRAG pulse
    if best_I is None:
        I = A_nominal * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        Q = np.zeros_like(I)
    else:
        I = best_I
        Q = best_Q
    
    return np.column_stack([I, Q])


def _score_pulse(I, Q, dt, n_steps, anharmonicity, T1, Tphi):
    """Score a pulse by computing infidelity across 9 detuning/scale cases."""
    detunings = [-2 * np.pi * 0.002e-3, 0.0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    
    for det in detunings:
        for scale in scales:
            # Build the Liouvillian evolution
            U = _evolve_unitary(I, Q, dt, n_steps, det, scale, anharmonicity)
            
            # Compute fidelity for each test state
            fidelities = []
            test_states = [
                np.array([1, 0, 0], dtype=complex),  # |0>
                np.array([0, 1, 0], dtype=complex),  # |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0> + |1>)/sqrt(2)
                (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0> - |1>)/sqrt(2)
                (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0> + i|1>)/sqrt(2)
                (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0> - i|1>)/sqrt(2)
            ]
            
            for psi in test_states:
                # Evolve state
                psi_final = U @ psi
                # Target: X gate maps |0> -> |1>, |1> -> |0>, others accordingly
                # X = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
                psi_target = _x_gate @ psi
                # Fidelity = |<psi_target|psi_final>|^2
                fid = np.abs(np.vdot(psi_target, psi_final))**2
                fidelities.append(fid)
            
            infid = 1.0 - np.mean(fidelities)
            infidelities.append(infid)
    
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid


def _evolve_unitary(I, Q, dt, n_steps, det, scale, anharmonicity):
    """Evolve the system under the given pulse and return the unitary."""
    # a matrix: lowering operator for 3 levels
    # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    
    # n = a^dagger a
    n = a_dag @ a
    
    # H(t) = det * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    # Note: n*(n-1) is the anharmonic term
    
    n_sq_minus_n = n @ (n - np.eye(3))
    
    # Build the time-dependent Hamiltonian
    H_static = det * n + 0.5 * anharmonicity * n_sq_minus_n
    
    def H_of_t(t, y):
        idx = int(round(t / dt))
        if idx >= n_steps:
            idx = n_steps - 1
        I_val = I[idx]
        Q_val = Q[idx]
        H_drive = 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
        return H_static + H_drive
    
    # Use solve_ivp to evolve the state
    # We need to evolve the full density matrix or use the unitary directly
    # For simplicity, let's evolve a basis state and reconstruct U
    
    # Actually, let's use the Liouvillian approach for accuracy
    # Or just evolve each column of the identity matrix
    
    # Evolve each basis state
    U = np.zeros((3, 3), dtype=complex)
    
    for i in range(3):
        psi0 = np.zeros(3, dtype=complex)
        psi0[i] = 1.0
        
        def rhs(t, y):
            H = H_of_t(t, y)
            return -1j * H @ y
        
        sol = solve_ivp(rhs, [0, n_steps * dt], psi0, method='RK45', 
                        rtol=1e-10, atol=1e-12, max_step=dt)
        if sol.success:
            U[:, i] = sol.y[:, -1]
        else:
            # Fallback: use Euler
            U[:, i] = _euler_evolve(i, I, Q, dt, n_steps, det, scale, anharmonicity)
    
    return U


def _euler_evolve(i, I, Q, dt, n_steps, det, scale, anharmonicity):
    """Fallback Euler evolution for basis state i."""
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n = a_dag @ a
    n_sq_minus_n = n @ (n - np.eye(3))
    H_static = det * n + 0.5 * anharmonicity * n_sq_minus_n
    
    psi = np.zeros(3, dtype=complex)
    psi[i] = 1.0
    
    for step in range(n_steps):
        H_drive = 0.5 * scale * (I[step] * (a + a_dag) + 1j * Q[step] * (a - a_dag))
        H = H_static + H_drive
        psi = psi - 1j * dt * H @ psi
    
    return psi


# X gate matrix
_x_gate = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]], dtype=complex)

# EVOLVE-BLOCK-END
