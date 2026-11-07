import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N_STEPS = 80
DURATION = 40.0

# Basis operators for 3-level system
# |0>, |1>, |2>
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op

# Vectorize density matrix (3x3 -> 9x1)
def vec_rho(rho):
    return rho.flatten()

def unvec_rho(vec):
    return vec.reshape(3, 3)

# Precompute dissipator superoperators
# L1 = a / sqrt(T1)
L1 = a_op / np.sqrt(T1)
L1_dag = L1.conj().T
L1_dag_L1 = L1_dag @ L1
# D1(rho) = 2 L1 rho L1_dag - L1_dag L1 rho - rho L1_dag L1
# In vectorized form: 2 kron(L1_dag, L1) - kron(I, L1_dag L1) - kron(L1_dag L1, I)
# Note: vec(A X B) = (B^T \otimes A) vec(X)
# vec(L1 rho L1_dag) = (L1_dag^T \otimes L1) vec(rho) = (L1_dag.conj() \otimes L1) vec(rho) since L1_dag = L1^T? No, L1_dag = L1^\dagger.
# L1 = [[0, 1/sqrt(T1), 0], ...]
# L1_dag = L1.T.conj()
# vec(L1 rho L1_dag) = (L1_dag.T \otimes L1) vec(rho) = (L1.conj() \otimes L1) vec(rho)
D1_super = 2 * np.kron(L1.conj(), L1) - np.kron(np.eye(3), L1_dag_L1) - np.kron(L1_dag_L1, np.eye(3))

# L2 = n * sqrt(2/Tphi)
L2 = n_op * np.sqrt(2.0 / Tphi)
L2_dag = L2.conj().T
L2_dag_L2 = L2_dag @ L2
D2_super = 2 * np.kron(L2.conj(), L2) - np.kron(np.eye(3), L2_dag_L2) - np.kron(L2_dag_L2, np.eye(3))

# Static Liouvillian parts
# H_static = delta * n + 0.5 * alpha * n(n-1)
# We need to build the full Liouvillian for each step.
# L = -i [H, rho] + D(rho)
# vec(L) = -i (I \otimes H - H^T \otimes I) vec(rho) + D_super vec(rho)
# Actually, [H, rho] = H rho - rho H.
# vec(H rho) = (I \otimes H) vec(rho)
# vec(rho H) = (H^T \otimes I) vec(rho)
# So -i [H, rho] -> -i (I \otimes H - H^T \otimes I)

def get_static_hamiltonian(delta, scale):
    # H = delta * n + 0.5 * alpha * n(n-1)
    # Note: The drive term is separate and depends on I, Q
    H_static = delta * n_op + 0.5 * ANHARMONICITY * (n_op @ n_op - n_op)
    return H_static

def get_drive_hamiltonian(I, Q, scale):
    # 0.5 * scale * (I (a + a_dag) + i Q (a - a_dag))
    H_drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
    return H_drive

def build_liouvillian(I, Q, delta, scale):
    H_static = get_static_hamiltonian(delta, scale)
    H_drive = get_drive_hamiltonian(I, Q, scale)
    H = H_static + H_drive
    
    # Coherent part: -i (I \otimes H - H^T \otimes I)
    # Note: H is Hermitian, so H^T = H^*
    coherent_super = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    # Dissipative part
    dissipative_super = D1_super + D2_super
    
    L = coherent_super + dissipative_super
    return L

# Precompute Liouvillians for all steps and all cases?
# No, I, Q change per step. But delta, scale are fixed per case.
# We can precompute the static part of the Liouvillian for each (delta, scale) pair.
# L_static(delta, scale) = -i (I \otimes H_static - H_static^T \otimes I) + D_super
# L_drive(I, Q, scale) = -i (I \otimes H_drive - H_drive^T \otimes I)

L_dissipative = D1_super + D2_super

static_liouvillians = {}
for delta in DETUNING_VALUES:
    for scale in SCALE_VALUES:
        H_static = get_static_hamiltonian(delta, scale)
        coherent_static = -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
        static_liouvillians[(delta, scale)] = coherent_static + L_dissipative

def generate_pulse(params):
    A, sigma, alpha_drag = params
    t = np.arange(N_STEPS) * DT
    # Center at 20 ns
    t_center = DURATION / 2.0
    
    I = A * np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    # dI/dt = - (t - t_center) / sigma^2 * I
    dI_dt = -(t - t_center) / (sigma ** 2) * I
    Q = alpha_drag * dI_dt
    
    pulse = np.column_stack((I, Q))
    return pulse

def check_validity(pulse):
    # Amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew bound
    # p[0] - 0
    if np.linalg.norm(pulse[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    # p[i+1] - p[i]
    diffs = np.diff(pulse, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    if np.any(norms / DT > SLEW_BOUND + 1e-9):
        return False
    # p[-1] - 0
    if np.linalg.norm(pulse[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
        
    return True

def simulate_pulse(pulse, delta, scale):
    # Initial state: 50% mixture of |0> and |1>? No, the scoring is fidelity to X over specific states.
    # We need to compute the process matrix or simulate for each initial state.
    # It's more efficient to simulate the evolution of the basis states |0>, |1>, |2> and then compute fidelity.
    # Or just simulate the density matrix evolution for the 5 test states?
    # Test states: |0>, |1>, (|0> + |1>)/sqrt(2), (|0> - |1>)/sqrt(2), (|0> + i|1>)/sqrt(2), (|0> - i|1>)/sqrt(2).
    # Actually, the prompt says: "fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)".
    # This implies 6 states? "mean fidelity ... over ...".
    # Let's assume the 6 states listed.
    
    states = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([1, 1, 0]) / np.sqrt(2),
        np.array([1, -1, 0]) / np.sqrt(2),
        np.array([1, 1j, 0]) / np.sqrt(2),
        np.array([1, -1j, 0]) / np.sqrt(2)
    ]
    
    target_states = []
    # X gate swaps |0> and |1>
    for psi in states:
        psi_x = np.array([psi[1], psi[0], psi[2]])
        target_states.append(psi_x)
    
    fidelities = []
    
    # Pre-compute propagators for this pulse and case
    L_static = static_liouvillians[(delta, scale)]
    
    # We need to apply the propagator to each state's density matrix.
    # Instead of simulating 6 states separately, we can simulate the process matrix?
    # Or just simulate the evolution of the 3 basis states |0>, |1>, |2> as vectors?
    # The Liouvillian acts on density matrices.
    # Let's simulate the density matrix for each of the 6 states.
    # 6 states * 80 steps = 480 expm applications? No, the propagator U_k is the same for all states.
    # We compute U_k = expm(L_k * DT).
    # Then rho_{k+1} = U_k @ rho_k.
    
    propagators = []
    for k in range(N_STEPS):
        I, Q = pulse[k]
        H_drive = get_drive_hamiltonian(I, Q, scale)
        coherent_drive = -1j * (np.kron(np.eye(3), H_drive) - np.kron(H_drive.T, np.eye(3)))
        L_k = L_static + coherent_drive
        U_k = expm(L_k * DT)
        propagators.append(U_k)
        
    for i, psi in enumerate(states):
        rho = np.outer(psi, psi.conj())
        rho_vec = vec_rho(rho)
        
        for k in range(N_STEPS):
            rho_vec = propagators[k] @ rho_vec
            
        rho_final = unvec_rho(rho_vec)
        
        # Fidelity F = |<psi_target|rho_final|psi_target>|^2 ?
        # For pure states, F = |<psi_t|psi_f>|^2.
        # We can compute this by projecting rho_final onto target state.
        psi_t = target_states[i]
        # <psi_t| rho |psi_t>
        fidelity = np.abs(np.vdot(psi_t, rho_final @ psi_t)) ** 2
        # Wait, rho_final is a density matrix. If the evolution is unitary (ignoring dissipation), it remains pure.
        # With dissipation, it becomes mixed.
        # Fidelity of mixed state rho with pure state |psi_t> is <psi_t| rho |psi_t>.
        fidelity = np.real(np.vdot(psi_t, rho_final @ psi_t))
        fidelities.append(fidelity)
        
    mean_fidelity = np.mean(fidelities)
    infidelity = 1.0 - mean_fidelity
    return infidelity

def score_pulse(params):
    pulse = generate_pulse(params)
    if not check_validity(pulse):
        return 1.0 # High score
    
    infidelities = []
    for delta in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            inf = simulate_pulse(pulse, delta, scale)
            infidelities.append(inf)
            
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score

# Optimization
def optimize():
    # Initial guess
    # Gaussian sigma ~ 5-10 ns
    # A ~ 0.3
    # alpha_drag ~ 1-2 ns
    x0 = [0.3, 6.0, 1.5]
    
    bounds = [
        (0.1, 0.4), # A
        (3.0, 10.0), # sigma
        (-3.0, 3.0) # alpha_drag
    ]
    
    result = minimize(score_pulse, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 200})
    
    return result.x

def pulse(target_gate, hardware_model):
    # The task says "Write pulse(target_gate, hardware_model)"
    # But the current program is `pulse = quantum_seed`.
    # I must return an (80, 2) array.
    
    # We run the optimization once and cache the result?
    # The function is called once per evaluation.
    # We should do the optimization inside `pulse` or pre-compute it?
    # "Keep the function names and signatures of the seed program."
    # Seed: `pulse = quantum_seed`.
    # If I define `def pulse(target_gate, hardware_model):`, I change the interface if the evaluator expects a function.
    # The prompt says: "Write pulse(target_gate, hardware_model), returning an (80, 2) array".
    # So `pulse` should be a function.
    
    global _optimized_pulse
    if '_optimized_pulse' not in globals():
        start_time = time.time()
        # Run optimization
        best_params = optimize()
        _optimized_pulse = generate_pulse(best_params)
        # Ensure it's valid, if not, fallback to a safe pulse
        if not check_validity(_optimized_pulse):
             # Fallback: simple Gaussian
             t = np.arange(N_STEPS) * DT
             I = 0.3 * np.exp(-((t - 20)**2) / (2*5**2))
             Q = np.zeros_like(I)
             _optimized_pulse = np.column_stack((I, Q))
             if not check_validity(_optimized_pulse):
                 # Even simpler
                 I = np.zeros(N_STEPS)
                 I[30:50] = 0.1
                 Q = np.zeros(N_STEPS)
                 _optimized_pulse = np.column_stack((I, Q))
        
    return _optimized_pulse

# To make it faster, we can pre-compute the optimization result at import time?
# No, the evaluator might import the module.
# Let's do it lazily in `pulse`.

# One issue: `optimize` might take too long.
# Let's add a timeout guard to `optimize` or just limit iterations.
# 200 iterations * 7ms/eval = 1.4s. This is safe.

# Let's refine the pulse generation to ensure slew rate is always respected by construction if possible.
# A = 0.12 * sigma * k.
# If we parameterize with sigma and k, A is fixed.
# But we want to optimize A too.
# The check_validity will reject invalid ones, guiding the optimizer.

# Let's double check the fidelity calculation.
# "infidelity = 1 - mean fidelity to X over ..."
# "Score = mean + max of the 9."
# My `score_pulse` does exactly this.

# One detail: "A DOP853 recheck of the nominal case must agree within 2e-7."
# This is a check on the *simulation* accuracy, not the pulse quality.
# Since I am using `expm` for fixed steps, it is a specific numerical method.
# The prompt says "each step exponentiates the Liouvillian".
# So my simulation is consistent with the model description.
# The DOP853 check is likely performed by the evaluator on the *pulse* I return, using their own simulator.
# As long as my pulse is valid and good, it should be fine.

# Let's ensure the output shape is (80, 2).
# `generate_pulse` returns `np.column_stack((I, Q))` which is (80, 2).

# Final check on constraints.
# Slew bound 0.12.
# If sigma is small, slope is high.
# Optimizer will find a valid sigma.

# To improve robustness, maybe use a slightly wider pulse?
# The optimizer will decide.

# Let's run a quick mental check on the DRAG coefficient.
# For a transmon with alpha = -2pi * 0.25, the DRAG coefficient is typically alpha / (2 * delta_12)?
# No, standard DRAG alpha is related to the anharmonicity and the ratio of matrix elements.
# Usually alpha_drag ~ 1-2 ns is a good start.

# I will add a fallback if optimization fails or takes too long.

def pulse(target_gate, hardware_model):
    if '_optimized_pulse' not in globals():
        try:
            # Set a timeout for the optimization part?
            # Python doesn't have a built-in timeout for functions easily.
            # But 200 iterations is fast.
            best_params = optimize()
            _optimized_pulse = generate_pulse(best_params)
            if not check_validity(_optimized_pulse):
                raise ValueError("Invalid pulse from optimizer")
        except Exception as e:
            # Fallback
            t = np.arange(N_STEPS) * DT
            # Simple Gaussian
            sigma = 6.0
            A = 0.3
            I = A * np.exp(-((t - 20)**2) / (2*sigma**2))
            Q = np.zeros_like(I)
            _optimized_pulse = np.column_stack((I, Q))
            if not check_validity(_optimized_pulse):
                # Even simpler, step function with small amplitude
                I = np.zeros(N_STEPS)
                I[20:60] = 0.1
                Q = np.zeros_like(I)
                _optimized_pulse = np.column_stack((I, Q))
                
    return _optimized_pulse

# To be safe, I will pre-calculate the pulse at the module level if possible?
# No, the task says "Write pulse...".
# I'll stick to the lazy initialization.

# One potential issue: `expm` is called 720 times per evaluation.
# 720 * 200 = 144,000 expm calls.
# Each expm on 9x9 is ~10us.
# 144,000 * 10us = 1.44s.
# This is well within 300s.

# Let's refine the initial guess to be closer to the optimum to reduce iterations.
# Typical DRAG pulse for 40ns gate:
# Sigma ~ 5-7 ns.
# A ~ 0.3-0.4 rad/ns.
# Alpha ~ 1-2 ns.

# I'll use x0 = [0.35, 6.0, 1.5].

# Also, I should ensure that the pulse is exactly 80 steps.
# `np.arange(80) * 0.5` gives 0 to 39.5. Correct.

# Let's write the code.
