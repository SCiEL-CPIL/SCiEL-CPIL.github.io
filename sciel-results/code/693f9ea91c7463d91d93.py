import numpy as np
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only 'X_pi' is supported")

    # Parse hardware model
    try:
        anharmonicity = hardware_model.get('anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        detuning_bounds = hardware_model.get('detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = hardware_model.get('drive_scale_bounds', (0.97, 1.03))
        T1 = hardware_model.get('T1', 25000)
        Tphi = hardware_model.get('Tphi', 40000)
    except AttributeError:
        anharmonicity = getattr(hardware_model, 'anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = getattr(hardware_model, 'amplitude_bound', 0.4)
        slew_bound = getattr(hardware_model, 'slew_bound', 0.12)
        detuning_bounds = getattr(hardware_model, 'detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = getattr(hardware_model, 'drive_scale_bounds', (0.97, 1.03))
        T1 = getattr(hardware_model, 'T1', 25000)
        Tphi = getattr(hardware_model, 'Tphi', 40000)

    dt = 0.5
    N_steps = 80
    t_total = N_steps * dt

    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    I_op = np.eye(3, dtype=complex)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    # Liouvillian superoperator construction helper
    def make_liouvillian(detuning, scale, I_val, Q_val):
        H_det = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - I_op)
        H_drive = 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
        H = H_det + H_drive

        S = -1j * (np.kron(I_op, H) - np.kron(H.T, I_op))

        for L in [L1, L2]:
            Ldag = L.conj().T
            term1 = np.kron(Ldag.conj().T, L)
            term2 = 0.5 * np.kron(I_op, Ldag @ L)
            term3 = 0.5 * np.kron((Ldag @ L).T, I_op)
            S += term1 - term2 - term3

        return S

    def simulate_pulse(pulse_flat, detuning, scale):
        P = pulse_flat.reshape(N_steps, 2)
        I_vals = P[:, 0]
        Q_vals = P[:, 1]

        # We compute the total superoperator for this (detuning, scale) case
        U_total = np.eye(9, dtype=complex)

        for i in range(N_steps):
            I_val = I_vals[i]
            Q_val = Q_vals[i]
            S = make_liouvillian(detuning, scale, I_val, Q_val)
            U_step = expm(S * dt)
            U_total = U_step @ U_total

        # Now apply U_total to the 4 input states
        # Input states: |0>, |1>, |+>, |+i>
        inputs = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) + 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)
        ]

        # Targets after X gate:
        # X|0> = |1>
        # X|1> = |0>
        # X|+> = |+>
        # X|+i> = |-i> = (|0> - i|1>)/sqrt(2)
        targets = [
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 0, 0], dtype=complex),
            (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) - 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)
        ]

        fidelities = []
        for idx in range(4):
            rho_in = np.outer(inputs[idx], inputs[idx].conj())
            rho_vec = rho_in.flatten()
            rho_out_vec = U_total @ rho_vec
            rho_out = rho_out_vec.reshape(3, 3)

            # Fidelity F(rho, |psi>) = <psi| rho |psi>
            F = np.real(np.vdot(targets[idx], rho_out @ targets[idx]))
            fidelities.append(F)

        return np.mean(fidelities)

    def objective(pulse_flat):
        infs = []
        detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]

        for d in detunings:
            for s in scales:
                fid = simulate_pulse(pulse_flat, d, s)
                inf = 1.0 - fid
                infs.append(inf)

        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)

    # Initial guess: Gaussian DRAG
    t_centers = (np.arange(N_steps) + 0.5) * dt
    t0 = t_total / 2.0
    sigma = 40.0 / 6.0
    A_gauss = np.sqrt(np.pi / 2.0) / sigma

    I_init = A_gauss * np.exp(-((t_centers - t0) ** 2) / (2 * sigma ** 2))
    coeff_drag = -anharmonicity / (2 * np.pi * 2 * sigma ** 2)
    Q_init = coeff_drag * (t_centers - t0) * I_init

    # Enforce constraints on initial guess
    I_init, Q_init = enforce_constraints(I_init, Q_init, amplitude_bound, slew_bound, dt)

    pulse_init = np.column_stack((I_init, Q_init)).flatten()

    # Optimize using Nelder-Mead
    # 160 dimensions is high, but Nelder-Mead can handle it if we limit iterations
    # and rely on the structure of the problem (smooth landscape near the seed)
    # Alternatively, we could optimize a lower-dimensional parameterization,
    # but the task asks to keep the interface and improve the program.
    # Let's try a few iterations to see if it improves the seed.

    # To stay within time limits, we'll limit maxiter.
    # 100 iterations * 9 cases * 80 steps * expm(9x9) is fast.
    # Let's try 50 iterations.

    from scipy.optimize import minimize

    res = minimize(objective, pulse_init, method='Nelder-Mead',
                   options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6})

    I_opt, Q_opt = res.x.reshape(N_steps, 2).T

    # Final constraint enforcement
    I_opt, Q_opt = enforce_constraints(I_opt, Q_opt, amplitude_bound, slew_bound, dt)

    pulse_array = np.column_stack((I_opt, Q_opt))
    return pulse_array

def enforce_constraints(I, Q, amp_bound, slew_bound, dt):
    """
    Scale I and Q to satisfy amplitude and slew rate constraints.
    """
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I *= scale
        Q *= scale

    # Slew rate constraint
    # Check internal steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew_internal = np.sqrt(dI**2 + dQ**2) / dt

    # Check start
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / dt
    # Check end
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / dt

    max_slew = np.max(np.concatenate((slew_internal, [slew_start, slew_end])))

    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        I *= scale
        Q *= scale

        # Re-check amplitude after slew scaling
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale2 = amp_bound / max_amp
            I *= scale2
            Q *= scale2

    return I, Q
