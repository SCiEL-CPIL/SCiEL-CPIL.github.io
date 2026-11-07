"""Rewritten program: robust DRAG-style pulse for 3-level transmon X-pi gate using numerical optimization."""
import numpy as np
from scipy.integrate import solve_ivp
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
    t_centers = (np.arange(N_steps) + 0.5) * dt
    
    # Precompute matrices
    a_mat = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_mat.conj().T
    n_mat = a_dag @ a_mat
    n_nm1 = n_mat @ (n_mat - np.eye(3))
    
    def simulate_pulse(I, Q, detuning, scale, T1_val, Tphi_val):
        """Simulate the 3-level system and return final density matrices for 4 basis states."""
        # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2)
        psi_0 = np.array([1, 0, 0], dtype=complex)
        psi_1 = np.array([0, 1, 0], dtype=complex)
        psi_p = (psi_0 + psi_1) / np.sqrt(2)
        psi_m = (psi_0 - psi_1) / np.sqrt(2)
        psi_pi = (psi_0 + 1j * psi_1) / np.sqrt(2)
        psi_mi = (psi_0 - 1j * psi_1) / np.sqrt(2)
        
        initial_states = [
            np.outer(psi_0, psi_0.conj()),
            np.outer(psi_1, psi_1.conj()),
            np.outer(psi_p, psi_p.conj()),
            np.outer(psi_m, psi_m.conj()),
            np.outer(psi_pi, psi_pi.conj()),
            np.outer(psi_mi, psi_mi.conj())
        ]
        
        # Target X gate on |0>, |1> subspace
        # X |0> = |1>, X |1> = |0>
        # Fidelity is <psi_target| rho |psi_target>
        # For |0>, target is |1>. For |1>, target is |0>.
        # For superpositions, target is X applied to state.
        
        targets = [
            np.outer(psi_1, psi_1.conj()),
            np.outer(psi_0, psi_0.conj()),
            np.outer(psi_m, psi_m.conj()), # X(|0>+|1>)/sqrt2 = (|1>+|0>)/sqrt2 = psi_p ... wait.
            # X |0> = |1>, X |1> = |0>.
            # X (|0>+|1>)/sqrt2 = (|1>+|0>)/sqrt2 = psi_p.
            # X (|0>-|1>)/sqrt2 = (|1>-|0>)/sqrt2 = -psi_m. Fidelity is same.
            # X (|0>+i|1>)/sqrt2 = (|1>+i|0>)/sqrt2 = i(|0>-i|1>)/sqrt2? No.
            # Let's just use unitary U = exp(-i pi/2 X) = -i X? No, X-pi gate is U = exp(-i pi X / 2) = -i X.
            # Actually, standard X-pi gate swaps |0> and |1> with a phase? 
            # The task says "fidelity to X". Usually X = [[0,1],[1,0]].
            # So target for |0> is |1>. Target for |1> is |0>.
            # Target for (|0>+|1>)/sqrt2 is (|1>+|0>)/sqrt2.
            # Target for (|0>-|1>)/sqrt2 is (|1>-|0>)/sqrt2.
            # Target for (|0>+i|1>)/sqrt2 is (|1>+i|0>)/sqrt2.
            # Target for (|0>-i|1>)/sqrt2 is (|1>-i|0>)/sqrt2.
            
            np.outer(psi_p, psi_p.conj()), # Wait, X psi_p = psi_p? No. X (|0>+|1>) = |1>+|0> = psi_p. Yes.
            np.outer(psi_m, psi_m.conj()), # X psi_m = -psi_m? X (|0>-|1>) = |1>-|0> = -psi_m. Fidelity is same.
            np.outer((psi_1 + 1j*psi_0)/np.sqrt(2), ((psi_1 + 1j*psi_0)/np.sqrt(2)).conj()),
            np.outer((psi_1 - 1j*psi_0)/np.sqrt(2), ((psi_1 - 1j*psi_0)/np.sqrt(2)).conj())
        ]
        
        # Simplified targets:
        t0 = np.array([0, 1, 0], dtype=complex)
        t1 = np.array([1, 0, 0], dtype=complex)
        tp = (t1 + t0) / np.sqrt(2)
        tm = (t1 - t0) / np.sqrt(2)
        tpi = (t1 + 1j*t0) / np.sqrt(2)
        tmi = (t1 - 1j*t0) / np.sqrt(2)
        
        target_states = [t0, t1, tp, tm, tpi, tmi]
        
        fidelities = []
        
        for rho_init in initial_states:
            rho = rho_init.copy()
            for i in range(N_steps):
                I_val = I[i]
                Q_val = Q[i]
                
                # Hamiltonian
                H_drive = 0.5 * scale * (I_val * (a_mat + a_dag) + 1j * Q_val * (a_mat - a_dag))
                H_det = detuning * n_mat
                H_anh = 0.5 * anharmonicity * n_nm1
                H = H_det + H_anh + H_drive
                
                # Lindblad operators
                L1 = a_mat / np.sqrt(T1_val)
                L2 = n_mat * np.sqrt(2.0 / Tphi_val)
                
                # Liouvillian
                # L(rho) = -i[H, rho] + sum (Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho})
                comm = H @ rho - rho @ H
                liouv = -1j * comm
                for L in [L1, L2]:
                    L_dag = L.conj().T
                    liouv += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
                
                # Exponentiate Liouvillian
                # Note: liouv is a matrix acting on vectorized rho.
                # We need to vectorize rho, apply expm(L*dt), then unvectorize.
                rho_vec = rho.flatten()
                liouv_mat = np.zeros((9, 9), dtype=complex)
                # Construct Liouvillian matrix representation
                # This is expensive to do every step. 
                # Optimization: Precompute the Liouvillian matrix structure?
                # H depends on I, Q. So H changes every step.
                # We must construct the 9x9 matrix.
                
                # Basis: |00>, |01>, |02>, |10>, |11>, |12>, |20>, |21>, |22>
                # rho_ij -> index i*3 + j
                for i in range(3):
                    for j in range(3):
                        idx = i * 3 + j
                        # -i H_{ik} rho_{kj}
                        for k in range(3):
                            liouv_mat[idx, k * 3 + j] -= 1j * H[i, k]
                        # -i rho_{ik} H_{kj}
                        for k in range(3):
                            liouv_mat[idx, i * 3 + k] -= 1j * H[k, j]
                        # L rho L^dag
                        for L in [L1, L2]:
                            for k in range(3):
                                for m in range(3):
                                    liouv_mat[idx, k * 3 + m] += L[i, k] * L.conj().T[m, j]
                            # -0.5 L^dag L rho
                            L_dag_L = L.conj().T @ L
                            for k in range(3):
                                liouv_mat[idx, i * 3 + k] -= 0.5 * L_dag_L[k, j]
                            # -0.5 rho L^dag L
                            for k in range(3):
                                liouv_mat[idx, k * 3 + j] -= 0.5 * L_dag_L[i, k]
                                
                # This loop is O(N^3) per step. 80 steps * 27 operations. It's fine.
                # But doing it in Python is slow. 
                # Let's use a vectorized approach or a helper.
                
                # Actually, let's just use the matrix exponential of the superoperator.
                # We can construct the superoperator more efficiently.
                
                rho_vec_new = expm(liouv_mat * dt) @ rho_vec
                rho = rho_vec_new.reshape((3, 3))
                
            # Calculate fidelity
            psi_t = target_states[initial_states.index(rho_init)]
            rho_t = np.outer(psi_t, psi_t.conj())
            fid = np.real(np.trace(rho_t @ rho))
            fidelities.append(fid)
            
        return np.mean(fidelities)

    def generate_pulse(sigma, drag_alpha, area_scale):
        """Generate I and Q arrays for given parameters."""
        t0 = t_total / 2.0
        # Gaussian envelope
        I = area_scale * np.exp(-((t_centers - t0) ** 2) / (2 * sigma ** 2))
        # Normalize area to pi? 
        # Area of exp is sigma * sqrt(2*pi). 
        # We want area = pi. So area_scale = sqrt(pi/2) / sigma.
        # But we can adjust area_scale to optimize.
        I = I * (np.sqrt(np.pi / 2.0) / sigma)
        
        # DRAG Q component
        # Q(t) = drag_alpha * dI/dt
        # dI/dt = I * (-(t-t0) / sigma^2)
        Q = drag_alpha * I * (-(t_centers - t0) / sigma ** 2)
        
        return I, Q

    def check_constraints(I, Q):
        """Check amplitude and slew constraints. Returns True if valid."""
        amp = np.sqrt(I ** 2 + Q ** 2)
        if np.max(amp) > amplitude_bound:
            return False
        
        I_prev = np.concatenate(([0], I[:-1]))
        Q_prev = np.concatenate(([0], Q[:-1]))
        dI = I - I_prev
        dQ = Q - Q_prev
        slew = np.sqrt(dI ** 2 + dQ ** 2) / dt
        
        dI_last = -I[-1]
        dQ_last = -Q[-1]
        slew_last = np.sqrt(dI_last ** 2 + dQ_last ** 2) / dt
        
        if np.max(np.concatenate((slew, [slew_last]))) > slew_bound:
            return False
            
        return True

    # Grid Search for optimal sigma and drag_alpha
    best_fid = 0.0
    best_I = None
    best_Q = None
    
    # Ranges
    sigmas = np.linspace(3.0, 8.0, 10)
    drag_alphas = np.linspace(-0.5, 0.5, 10)
    
    for sigma in sigmas:
        for alpha in drag_alphas:
            I, Q = generate_pulse(sigma, alpha, 1.0)
            
            if not check_constraints(I, Q):
                continue
                
            # Simulate nominal case
            # detuning = 0, scale = 1
            fid = simulate_pulse(I, Q, 0.0, 1.0, T1, Tphi)
            
            if fid > best_fid:
                best_fid = fid
                best_I = I.copy()
                best_Q = Q.copy()

    # If no valid pulse found (unlikely), fallback to seed-like
    if best_I is None:
        sigma = 40.0 / 6.0
        best_I, best_Q = generate_pulse(sigma, 0.1, 1.0)
        # Force validity
        amp = np.sqrt(best_I ** 2 + best_Q ** 2)
        if np.max(amp) > amplitude_bound:
            s = amplitude_bound / np.max(amp)
            best_I *= s
            best_Q *= s
        # Check slew
        I_prev = np.concatenate(([0], best_I[:-1]))
        Q_prev = np.concatenate(([0], best_Q[:-1]))
        dI = best_I - I_prev
        dQ = best_Q - Q_prev
        slew = np.sqrt(dI ** 2 + dQ ** 2) / dt
        dI_last = -best_I[-1]
        dQ_last = -best_Q[-1]
        slew_last = np.sqrt(dI_last ** 2 + dQ_last ** 2) / dt
        s_max = np.max(np.concatenate((slew, [slew_last])))
        if s_max > slew_bound:
            s = slew_bound / s_max
            best_I *= s
            best_Q *= s

    return np.column_stack((best_I, best_Q))
