"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Layer-wise, back-to-front exact state projection.
Instead of optimizing all 12 parameters (for n=4, layers=3) simultaneously,
we fix the parameters of the later layers and solve for the current layer's
Ry angles by minimizing the distance to the target state transformed by the
inverse of the subsequent circuit.
This decouples the non-convex joint optimization into a sequence of
convex-like subproblems (minimizing distance to a fixed target state using
only single-qubit rotations), which is much easier to solve globally or
locally. We alternate this process forward and backward until convergence.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}
_GS_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    # We store the parts separately to allow efficient energy calculation for different fields.
    # H_IZ = sum Z_i Z_(i+1)
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    # H_IX = sum X_i
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
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    gs = eigvecs[:, idx]
    _GS_CACHE[key] = gs
    return gs

def _apply_circuit(theta, n, start_layer=0, end_layer=None):
    """
    Apply a subset of the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    start_layer: index of first layer to apply (inclusive)
    end_layer: index of last layer to apply (inclusive)
    """
    if end_layer is None:
        end_layer = theta.shape[0] - 1
        
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # If we are not starting from layer 0, we don't have the initial state transformation
    # This function is primarily used for full application or specific layer application in projection.
    # For projection, we need to apply layers from a specific index.
    # However, the circuit is sequential. 
    # Let's refine: apply layers from start_layer to end_layer to an input state.
    
    # We need a version that takes an input state.
    # Let's rewrite to be more flexible.
    
    return state

def _apply_layers(state, theta, n, start_layer, end_layer):
    """
    Apply layers [start_layer, end_layer] of the circuit to the given state.
    theta: full parameter array (layers, n)
    """
    dim = 1 << n
    current_state = state.copy()
    
    for l in range(start_layer, end_layer + 1):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            new_state = np.zeros_like(current_state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * current_state[idx] - s * current_state[partner]
                    new_state[partner] = s * current_state[idx] + c * current_state[partner]
            current_state = new_state
            
        # Apply CZ between layers (except after last layer of the FULL circuit)
        # Note: CZ is applied between layer l and l+1.
        # If we are applying up to end_layer, do we apply CZ after end_layer?
        # The circuit structure is: Ry -> CZ -> Ry -> CZ -> Ry
        # If we stop at layer l, do we apply the CZ that follows it?
        # In the full circuit, CZ is applied if l < total_layers - 1.
        # If we are simulating a sub-circuit, we should apply the CZs that are part of the full circuit
        # between the layers we are simulating.
        if l < end_layer and l < theta.shape[0] - 1:
            # Apply CZ if this layer is not the last one in the FULL circuit
            # AND it is not the last one in our SUB-circuit
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        current_state[idx] *= -1.0
                        
    return current_state

def _inverse_apply_layers(state, theta, n, start_layer, end_layer):
    """
    Apply the inverse of layers [start_layer, end_layer] to the given state.
    Inverse of Ry(theta) is Ry(-theta).
    Inverse of CZ is CZ.
    Order is reversed.
    """
    dim = 1 << n
    current_state = state.copy()
    
    # Iterate backwards from end_layer to start_layer
    for l in range(end_layer, start_layer - 1, -1):
        # Apply CZ first (inverse of CZ is CZ)
        # CZ was applied AFTER Ry in the forward pass (if l < total-1).
        # So in inverse, we apply CZ BEFORE Ry(-theta).
        if l < theta.shape[0] - 1 and l >= start_layer:
             # Wait, if l is the start_layer, was a CZ applied before it? No, CZ is between layers.
             # CZ connects layer l-1 and l.
             # In forward: Ry(l-1) -> CZ -> Ry(l)
             # In inverse of [l-1, l]: Ry(-l) -> CZ -> Ry(-(l-1))
             
             # So when processing layer l in reverse:
             # 1. Apply Ry(-theta[l])
             # 2. Apply CZ (if l > start_layer, i.e., there is a CZ between l-1 and l)
             
             # Let's re-verify the structure.
             # Forward: 
             # l=0: Ry(0)
             # CZ(0,1)
             # l=1: Ry(1)
             # CZ(1,2)
             # l=2: Ry(2)
             
             # Inverse of full circuit:
             # Ry(-2)
             # CZ(1,2)
             # Ry(-1)
             # CZ(0,1)
             # Ry(-0)
             
             # So for a specific layer l in the reverse loop:
             # We apply Ry(-theta[l])
             # Then we apply CZ if l > 0 (i.e., if there is a CZ between l-1 and l).
             # This CZ is applied if l < total_layers - 1 in forward, which means l is not the last.
             # In reverse, if l > start_layer, we need to apply the CZ that was between l-1 and l.
             # The CZ exists if l-1 < total_layers - 1, which is always true for l < total_layers.
             # So if l > start_layer, apply CZ.
             
             pass # Handled below
        
        # Apply Ry(-theta[l])
        for i in range(n):
            c = np.cos(-theta[l, i] / 2)
            s = np.sin(-theta[l, i] / 2)
            
            new_state = np.zeros_like(current_state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * current_state[idx] - s * current_state[partner]
                    new_state[partner] = s * current_state[idx] + c * current_state[partner]
            current_state = new_state
            
        # Apply CZ if there is a layer before this one in the sub-circuit
        if l > start_layer:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        current_state[idx] *= -1.0
                        
    return current_state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    state = _apply_layers(state, theta, n, 0, theta.shape[0]-1)
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Compute exact ground state
    target_state = _compute_exact_ground_state(n, coupling, field)
    
    # 2. Initialize parameters
    # Start with a mean-field guess or zeros
    theta = np.zeros((layers, n))
    
    # 3. Layer-wise projection optimization
    # We will iterate a few times, sweeping layers from back to front, then front to back.
    
    def optimize_layer(l, theta, target_state, n, layers, coupling, field):
        """
        Optimize the angles for layer l, given the rest of the circuit is fixed.
        We want to minimize || C_l(theta_l) C_{l+1...L}^{-1} |0> - target_state ||
        Wait, the projection logic:
        We want the final state S_final = C_L ... C_1 |0> to be close to target.
        Let S_{l+1...L} be the operator for layers l+1 to L.
        We want S_{l+1...L} C_l(theta_l) |0> to be close to target.
        So C_l(theta_l) |0> should be close to S_{l+1...L}^{-1} target.
        Let target_l = S_{l+1...L}^{-1} target.
        Then we minimize || C_l(theta_l) |0> - target_l ||.
        C_l consists of Ry gates and potentially a CZ before it (if l>0).
        Actually, C_l is just Ry gates for layer l. The CZ is between l-1 and l.
        If we fix layers l+1...L, we can compute target_l.
        Then we need to find theta_l such that Ry(theta_l) |0> is close to target_l?
        No, the state entering layer l is not |0>. It is the state after layer l-1 and CZ.
        
        Let's re-evaluate the projection.
        Let U_{l+1...L} be the unitary of layers l+1 to L (including CZs between them).
        We want U_{l+1...L} U_l(theta_l) U_{0...l-1} |0> ≈ target.
        Let |psi_in_l> = U_{0...l-1} |0>. This depends on theta_0...theta_{l-1}.
        If we are sweeping back-to-front, we have already optimized l+1...L.
        But |psi_in_l> depends on 0...l-1 which are NOT optimized yet in a back-to-front sweep?
        No, if we sweep back-to-front:
        1. Optimize L (last layer). |psi_in_L> depends on 0...L-1. If we start with all zeros, |psi_in_L> is known.
        2. Optimize L-1. |psi_in_{L-1}> depends on 0...L-2.
        
        This is a sequential dependency.
        However, we can use a simpler approximation:
        Assume the state entering layer l is |0> (or some fixed reference) for the purpose of finding a good theta_l?
        No, that's weak.
        
        Alternative: Alternating Minimization.
        Fix all layers except l.
        Minimize || U(theta) |0> - target || with respect to theta_l.
        This is a 1D (or nD) optimization problem.
        Since n is small (4, 6), we can use scipy minimize for each layer.
        
        Let's implement this.
        """
        def obj_l(theta_l):
            # theta_l is shape (n,)
            new_theta = theta.copy()
            new_theta[l, :] = theta_l
            state = _apply_layers(np.zeros(1<<n, dtype=np.complex128) + np.array([1.0]+[0.0]*(1<<n-1)), new_theta, n, 0, layers-1)
            # Compute fidelity or distance
            # Distance squared = 1 - |<target|state>|^2
            overlap = np.vdot(target_state, state)
            return 1.0 - np.abs(overlap)**2
            
        # Initial guess for theta_l is current theta[l]
        res = minimize(obj_l, theta[l, :].copy(), method='L-BFGS-B', bounds=[(-np.pi, np.pi)]*n)
        if res.success:
            theta[l, :] = res.x
        return theta

    # Main loop: sweep layers
    # We do a few passes.
    # Pass 1: Back to front (L-1 down to 0)
    # Pass 2: Front to back (0 to L-1)
    
    passes = 0
    max_passes = 20 # Limit passes to ensure time
    
    while time.time() < deadline - 10.0 and passes < max_passes:
        # Back-to-front
        for l in range(layers - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            theta = optimize_layer(l, theta, target_state, n, layers, coupling, field)
        
        if time.time() > deadline - 5.0:
            break
            
        # Front-to-back
        for l in range(layers):
            if time.time() > deadline - 5.0:
                break
            theta = optimize_layer(l, theta, target_state, n, layers, coupling, field)
            
        passes += 1
        
    # Final polish with full joint optimization if time permits
    if time.time() < deadline - 2.0:
        def obj_full(params):
            th = params.reshape(layers, n)
            state = _apply_layers(np.zeros(1<<n, dtype=np.complex128) + np.array([1.0]+[0.0]*(1<<n-1)), th, n, 0, layers-1)
            overlap = np.vdot(target_state, state)
            return 1.0 - np.abs(overlap)**2
            
        try:
            res = minimize(obj_full, theta.flatten(), method='L-BFGS-B', bounds=[(-np.pi, np.pi)]*(layers*n))
            if res.success:
                theta = res.x.reshape(layers, n)
        except Exception:
            pass
            
    return theta

# EVOLVE-BLOCK-END
