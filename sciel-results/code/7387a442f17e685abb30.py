"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Layer-wise, back-to-front exact state projection.
Instead of optimizing all angles jointly, we compute the exact ground state |psi>.
We then work backwards from the last layer to the first.
For each layer l, we fix the angles of layers l+1..L.
We apply the inverse of the circuit from layer l+1 to L to the target state |psi>.
This gives a target state |phi_l> for the output of layer l's Ry gates (before CZ, if l < L).
Wait, the circuit is: Ry -> CZ -> Ry -> CZ -> Ry.
Let U = Ry_L * CZ * Ry_{L-1} * CZ * ... * Ry_1.
We want U |0> approx |psi>.
Let |psi> be the target.
Step 1 (Layer L):
  We need Ry_L |v_{L-1}> approx |psi>, where |v_{L-1}> is the state after CZ_{L-1}.
  Actually, let's work with the inverse.
  U^{-1} |psi> = Ry_1^{-1} CZ^{-1} ... Ry_L^{-1} |psi>.
  Let |t_L> = Ry_L^{-1} |psi>.
  Then Ry_{L-1} CZ |t_L> should approximate |0>? No.
  
  Let's rephrase:
  We want to find theta_1, theta_2, theta_3 such that:
  Ry(theta_3) CZ Ry(theta_2) CZ Ry(theta_1) |0> approx |psi>.
  
  Let |s_0> = |0>.
  |s_1> = Ry(theta_1) |s_0>.
  |s_2> = CZ |s_1>.
  |s_3> = Ry(theta_2) |s_2>.
  |s_4> = CZ |s_3>.
  |s_5> = Ry(theta_3) |s_4>.
  
  We want |s_5> approx |psi>.
  
  Back-to-front:
  1. Fix theta_2, theta_3? No, we don't know them yet.
  Let's assume we know the "effective target" for each step.
  
  Let's define the target for the output of Ry_l.
  For the last layer (l=3):
  We want Ry(theta_3) |s_4> approx |psi>.
  This implies |s_4> should be approx Ry(theta_3)^{-1} |psi>.
  But |s_4> is determined by theta_1, theta_2.
  
  Alternative approach: Layer-wise optimization.
  1. Initialize all theta to 0.
  2. For l = L down to 1:
     - Compute the state |psi_rest> = U_{l+1..L}^{-1} |psi>, where U_{l+1..L} is the circuit from layer l+1 to L.
       Wait, if we are at layer l, the state after layer l is |s_{l+1}>.
       The rest of the circuit is U_{l+1..L}.
       We want U_{l+1..L} |s_{l+1}> approx |psi>.
       So |s_{l+1}> should approx U_{l+1..L}^{-1} |psi>.
       
       Let |target_{l+1}> = U_{l+1..L}^{-1} |psi>.
       
       Now, |s_{l+1}> = (CZ if l < L else I) Ry(theta_l) |s_l>.
       Wait, the structure is:
       Layer 1: Ry(theta_1)
       Entangler: CZ
       Layer 2: Ry(theta_2)
       Entangler: CZ
       Layer 3: Ry(theta_3)
       
       So after Layer l, we have Ry(theta_l) applied to |s_l>.
       Then if l < L, CZ is applied.
       
       Let's define |s_l> as the state BEFORE layer l's Ry.
       |s_1> = |0>.
       After Ry(theta_1): |v_1> = Ry(theta_1) |s_1>.
       After CZ (if l=1): |s_2> = CZ |v_1>.
       
       After Ry(theta_2): |v_2> = Ry(theta_2) |s_2>.
       After CZ (if l=2): |s_3> = CZ |v_2>.
       
       After Ry(theta_3): |v_3> = Ry(theta_3) |s_3>.
       Final state: |psi_circ> = |v_3>.
       
       We want |v_3> approx |psi>.
       
       Back-to-front algorithm:
       1. Start with target for |v_3>: |T_3> = |psi>.
       2. Optimize theta_3 to minimize || Ry(theta_3) |s_3> - |T_3> ||.
          But |s_3> depends on theta_1, theta_2.
          This coupling is the problem.
          
       However, we can use a greedy layer-wise approach:
       1. Assume |s_3> is known (initially 0 or from previous iteration).
       2. Optimize theta_3 to make Ry(theta_3) |s_3> close to |psi>.
       3. Now we have a new target for |s_3>: It should be Ry(theta_3)^{-1} |psi>.
          Let |T_2_out> = Ry(theta_3)^{-1} |psi>.
          We want |s_3> approx |T_2_out>.
          But |s_3> = CZ |v_2> = CZ Ry(theta_2) |s_2>.
          So we want CZ Ry(theta_2) |s_2> approx |T_2_out>.
          This implies Ry(theta_2) |s_2> approx CZ^{-1} |T_2_out> = CZ |T_2_out> (since CZ is self-inverse).
          Let |T_2_in> = CZ |T_2_out>.
          Now we want Ry(theta_2) |s_2> approx |T_2_in>.
          Optimize theta_2 given |s_2> (which depends on theta_1).
          
       This suggests an iterative refinement:
       1. Initialize theta_1, theta_2, theta_3 to 0.
       2. Repeat for a few iterations:
          a. Forward pass to compute |s_1>, |s_2>, |s_3> using current thetas.
          b. Backward pass:
             - Layer 3:
               Target for |v_3> is |psi>.
               We have |s_3>.
               Optimize theta_3 to minimize || Ry(theta_3) |s_3> - |psi> ||.
               This is a product of independent 2-level optimizations?
               No, Ry(theta_3) acts on all qubits.
               Minimize || (⊗ Ry(theta_i)) |s_3> - |psi> ||^2.
               This is still non-convex if |s_3> is entangled.
               BUT, for small n=4,6, we can use gradient descent or even a grid search for this specific step?
               Actually, if we fix |s_3>, the function f(theta) = || Ry(theta) |s_3> - |psi> ||^2 is smooth.
               We can use L-BFGS-B to optimize theta_3 (n parameters) given |s_3>.
               
             - Layer 2:
               New target for |v_2> is CZ Ry(theta_3)^{-1} |psi>.
               Let |T_2> = CZ Ry(theta_3)^{-1} |psi>.
               We have |s_2>.
               Optimize theta_2 to minimize || Ry(theta_2) |s_2> - |T_2> ||.
               
             - Layer 1:
               New target for |v_1> is CZ Ry(theta_2)^{-1} CZ Ry(theta_3)^{-1} |psi>.
               Let |T_1> = CZ Ry(theta_2)^{-1} CZ Ry(theta_3)^{-1} |psi>.
               We have |s_1> = |0>.
               Optimize theta_1 to minimize || Ry(theta_1) |0> - |T_1> ||.
               Since |s_1> = |0> is a product state, Ry(theta_1) |0> is also a product state.
               Minimizing distance to a product state is a set of independent 1D problems!
               For each qubit i, we want <i| Ry(theta_i) |0> to match the reduced state?
               Actually, || Ry(theta) |0> - |T> ||^2 = 2 - 2 Re <0| Ry(theta)^{-1} |T>.
               <0| Ry(theta)^{-1} |T> = ⊗ <0| Ry(theta_i)^{-1} (Tr_{others} |T><T|) ... no, it's not separable.
               However, if we assume |T_1> is close to a product state, we can approximate.
               Or simply use L-BFGS-B for theta_1 as well.
               
       This "layer-wise back-propagation" with local optimization for each layer's angles should be much more effective than a single global optimization because it respects the circuit structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}
_EIG_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    H_IX = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_state(n, coupling, field):
    """Compute the exact ground state vector."""
    key = (n, coupling, field)
    if key in _EIG_CACHE:
        return _EIG_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    psi = eigvecs[:, idx]
    # Ensure deterministic phase
    max_idx = np.argmax(np.abs(psi))
    if psi[max_idx].real < 0:
        psi = -psi
    elif abs(psi[max_idx].real) < 1e-10 and psi[max_idx].imag < 0:
        psi = -psi
        
    _EIG_CACHE[key] = psi
    return psi

def _apply_ry(state, theta, n):
    """Apply Ry(theta) to each qubit of the state."""
    dim = 1 << n
    new_state = np.zeros_like(state)
    for i in range(n):
        c = np.cos(theta[i] / 2)
        s = np.sin(theta[i] / 2)
        mask = 1 << i
        for idx in range(dim):
            if (idx & mask) == 0:
                partner = idx | mask
                new_state[idx] = c * state[idx] - s * state[partner]
                new_state[partner] = s * state[idx] + c * state[partner]
    return new_state

def _apply_cz(state, n):
    """Apply CZ to every adjacent pair."""
    dim = 1 << n
    new_state = state.copy()
    for i in range(n - 1):
        mask = (1 << i) | (1 << (i + 1))
        for idx in range(dim):
            if (idx & mask) == mask:
                new_state[idx] *= -1.0
    return new_state

def _forward_pass(theta, n, layers):
    """Apply the full circuit and return intermediate states."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    intermediates = []
    for l in range(layers):
        state = _apply_ry(state, theta[l], n)
        intermediates.append(state)
        if l < layers - 1:
            state = _apply_cz(state, n)
            
    return intermediates

def _layer_optimization(theta_current, target_state, n, layer_idx, layers, start_time, deadline):
    """
    Optimize angles for a specific layer, given the state BEFORE the layer and the target state AFTER the layer.
    Note: 'target_state' here is the state we want the output of this layer's Ry to be.
    The input to this layer's Ry is 'input_state'.
    We minimize || Ry(theta) |input_state> - |target_state> ||.
    """
    input_state = theta_current['input_states'][layer_idx]
    
    def objective(params):
        new_state = _apply_ry(input_state, params, n)
        return np.vdot(new_state - target_state, new_state - target_state).real
    
    x0 = theta_current['angles'][layer_idx].copy()
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        if time.time() > deadline:
            return x0
        return res.x
    except Exception:
        return x0

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground state
    psi_target = _compute_exact_ground_state(n, coupling, field)
    
    # 2. Initialize angles
    theta = np.zeros((layers, n))
    
    # Iterative layer-wise optimization
    # We will iterate a few times, updating each layer's angles in back-to-front order.
    # Each iteration:
    #   1. Forward pass to get input states for each layer.
    #   2. Backward pass:
    #      - For layer L: Optimize theta_L to match psi_target.
    #      - For layer L-1: Optimize theta_{L-1} to match CZ Ry_L^{-1} psi_target.
    #      - ...
    
    max_iterations = 5
    for it in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Forward pass to get input states
        state = np.zeros(1 << n, dtype=np.complex128)
        state[0] = 1.0
        input_states = []
        
        for l in range(layers):
            input_states.append(state)
            state = _apply_ry(state, theta[l], n)
            if l < layers - 1:
                state = _apply_cz(state, n)
                
        # Backward pass
        # We need to propagate the target backwards.
        # Let target_after[l] be the desired state after layer l's Ry.
        # target_after[L-1] = psi_target
        # target_after[l-1] = CZ Ry_l^{-1} target_after[l]  (if l > 0)
        
        target_after = [None] * layers
        target_after[layers - 1] = psi_target
        
        for l in range(layers - 1, 0, -1):
            # Inverse of Ry(theta[l])
            theta_inv = -theta[l]
            # Apply inverse Ry
            state_tmp = _apply_ry(target_after[l], theta_inv, n)
            # Apply CZ (self-inverse)
            state_tmp = _apply_cz(state_tmp, n)
            target_after[l - 1] = state_tmp
            
        # Now optimize each layer
        for l in range(layers - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Optimize theta[l] given input_states[l] and target_after[l]
            new_theta_l = _layer_optimization(
                {'input_states': input_states, 'angles': theta}, 
                target_after[l], 
                n, 
                l, 
                layers, 
                start_time, 
                deadline
            )
            theta[l] = new_theta_l
            
    return theta

# EVOLVE-BLOCK-END
