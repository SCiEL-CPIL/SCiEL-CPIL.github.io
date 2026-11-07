"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.

    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    # Time guard
    start_time = time.time()
    deadline = start_time + 310.0  # 20s safety margin

    # Time grid for constant steps (midpoint rule)
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Hamiltonian / Liouvillian setup
    # a is the lowering operator for a 3-level system (approx)
    # |0> -> |1> with 1, |1> -> |2> with sqrt(2)
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    eye3 = np.eye(3)
    L1c = L1.conj().T
    L2c = L2.conj().T

    # Lindblad contribution to Liouvillian (9x9)
    # L(rho) = L rho L^dag - 0.5 {L^dag L, rho}
    # Vectorization: vec(L rho L^dag) = (L^dag^T \otimes L) vec(rho)
    # vec({L^dag L, rho}) = (I \otimes L^dag L + (L^dag L)^T \otimes I) vec(rho)
    # Since L^dag L is Hermitian, (L^dag L)^T = L^dag L
    Lindblad_part = np.zeros((9, 9), dtype=complex)
    for L, Lc in [(L1, L1c), (L2, L2c)]:
        Lindblad_part += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, eye3)
                          - 0.5 * np.kron(eye3, L @ Lc))

    def hamiltonian_part(detuning, scale, I, Q):
        H = (detuning * n
             + 0.5 * anharmonicity * (n @ n - n)
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        Hc = H.conj().T
        return -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]

    # Pre-compute initial density matrices (vectorized)
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states])

    # Target states for fidelity
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]
    # Vectorize target states for dot product
    target_vecs = np.array([ts.reshape(9) for ts in target_states])

    def score_pulse(I, Q):
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]

        fids = []
        for d in detunings:
            for s in scales:
                rho_batch = rho0s.copy()

                for k in range(N):
                    H_part = hamiltonian_part(d, s, I[k], Q[k])
                    L_full = Lindblad_part + H_part
                    U = linalg.expm(L_full * dt)
                    rho_batch = (U @ rho_batch.T).T

                case_fids = []
                for i in range(6):
                    rho_final = rho_batch[i].reshape(3, 3)
                    psi_target = target_states[i]
                    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
                    case_fids.append(fid)
                fids.append(np.mean(case_fids))

        fids = np.array(fids)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def generate_pulse(params):
        """
        params: [amp, sigma, drag, shift]
        """
        amp, sigma, drag, shift = params
        sigma = max(sigma, 1.0) # Prevent division by zero or too narrow
        center = T / 2.0 + shift
        
        # Gaussian envelope
        g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        
        # Offset to 0 at edges? The task says "offset to 0 at the edges" for the seed.
        # We will enforce this by subtracting the value at t=0 and t=T (if non-zero) 
        # or simply using a window that is naturally small. 
        # For robustness, let's use a cosine window multiplied by gaussian? 
        # Or just subtract edge values.
        edge_val = 0.5 * (np.exp(-0.5 * ((0 - center) / sigma) ** 2) + 
                          np.exp(-0.5 * ((T - center) / sigma) ** 2))
        g = g - edge_val
        g = np.maximum(g, 0.0)
        
        # Normalize area to pi
        area = np.sum(g) * dt
        if area > 1e-9:
            g = g * (np.pi / area)
        else:
            return np.zeros(N), np.zeros(N)
        
        # Scale amplitude
        I = amp * g
        
        # DRAG term: derivative of envelope
        g_deriv = np.gradient(g, dt)
        Q = drag * g_deriv
        
        # Enforce constraints
        # 1. Amplitude
        amp_calc = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp_calc)
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I = I * factor
            Q = Q * factor
            
        # 2. Slew rate
        # Check slew from 0 to I[0], I[k] to I[k+1], I[-1] to 0
        def check_and_fix_slew(I, Q):
            # Calculate slews
            slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI**2 + dQ**2) / dt
            slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
            
            max_slew = max(slew0, np.max(slews) if len(slews)>0 else 0.0, slewN)
            
            if max_slew > slew_bound:
                factor = slew_bound / max_slew
                I = I * factor
                Q = Q * factor
            return I, Q
        
        I, Q = check_and_fix_slew(I, Q)
        
        return I, Q

    def objective(params):
        if time.time() > deadline:
            return 1e6
        I, Q = generate_pulse(params)
        return score_pulse(I, Q)

    # Initial guess
    # amp ~ 0.1, sigma ~ T/6, drag ~ 0.2, shift ~ 0
    x0 = [0.15, T/6.0, 0.2, 0.0]
    
    best_score = float('inf')
    best_I = None
    best_Q = None
    
    # Run Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-5, 'disp': False})
        if res.fun < best_score:
            best_score = res.fun
            best_I, best_Q = generate_pulse(res.x)
    except Exception:
        pass

    # If optimization failed or didn't find anything, try a few random restarts
    if best_I is None or best_score > 0.5:
        np.random.seed(42)
        for i in range(20):
            if time.time() > deadline:
                break
            # Random parameters
            amp = np.random.uniform(0.1, 0.3)
            sigma = np.random.uniform(T/10, T/4)
            drag = np.random.uniform(-0.5, 0.5)
            shift = np.random.uniform(-2, 2)
            params = [amp, sigma, drag, shift]
            I, Q = generate_pulse(params)
            sc = score_pulse(I, Q)
            if sc < best_score:
                best_score = sc
                best_I, best_Q = I, Q
                # Local refinement around this point
                try:
                    res2 = minimize(objective, params, method='Nelder-Mead', 
                                    options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5, 'disp': False})
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_I, best_Q = generate_pulse(res2.x)
                except Exception:
                    pass

    if best_I is None:
        # Fallback to seed
        best_I, best_Q = generate_pulse(x0)

    # Final validity check
    amp = np.sqrt(best_I ** 2 + best_Q ** 2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"

    def check_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
        return m <= slew_bound

    assert check_slew(best_I, best_Q), "Slew rate constraint violated"

    return np.column_stack([best_I, best_Q])

# EVOLVE-BLOCK-END
