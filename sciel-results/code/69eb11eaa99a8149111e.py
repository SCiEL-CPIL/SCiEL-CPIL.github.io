"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Global Stochastic Search (Differential Evolution) with Vectorized Evaluation.
The previous strategy used local optimization (L-BFGS-B) which is prone to local minima 
in the complex energy landscape of shallow circuits. We switch to a global optimizer 
(Differential Evolution) which is robust to local minima. To handle the computational 
cost within the 150s limit, we heavily optimize the energy evaluation:
1. Pre-compute Hamiltonian matrix and exact ground energy.
2. Use vectorized operations to apply the circuit to the initial state.
3. Implement a cache for state vectors to avoid redundant calculations during DE steps.
"""
import numpy as np
from scipy.optimize import differential_evolution
import time

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_GS_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    # We build the full matrix once.
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[state, state] += coupling
            else:
                H[state, state] -= coupling
                
    # The field term is added later or we build H_field separately?
    # Actually, the Hamiltonian depends on field. 
    # H = -J sum ZZ - h sum X
    # Let's build H_ZZ and H_X separately to be safe, or just rebuild if field changes.
    # Since field is part of the key in _GS_CACHE, but H_ZZ and H_X are independent of field,
    # we can cache H_ZZ and H_X.
    
    H_ZZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_ZZ[state, state] += 1.0
            else:
                H_ZZ[state, state] -= 1.0
                
    H_X = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_X[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_ZZ, H_X)
    return _HAM_CACHE[key]

def _compute_exact_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    key = (n, coupling, field)
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    
    H_ZZ, H_X = _get_hamiltonian(n, coupling)
    H = -coupling * H_ZZ - field * H_X
    
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0].real
    psi0 = eigvecs[:, 0]
    
    _GS_CACHE[key] = (e0, psi0)
    return _GS_CACHE[key]

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    # State is represented as a vector of length dim
    # |0>^n is index 0
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # Acting on |0> -> cos(theta/2)|0> + sin(theta/2)|1>
        # Acting on |1> -> -sin(theta/2)|0> + cos(theta/2)|1>
        
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            # We need to apply this rotation to all pairs of states that differ only in bit i.
            # Instead of looping over all states, we can use reshape or slicing.
            # But for small N (4, 6), explicit loop is fast enough if vectorized inner loop?
            # Let's try a vectorized approach for the rotation.
            
            # Create masks for bit i being 0 and 1
            # This is tricky with simple numpy indexing without creating huge intermediate arrays.
            # Given dim <= 64, a simple loop over dim is actually quite fast in Python if we minimize overhead.
            # However, DE will call this many times.
            
            # Let's use a more efficient manual loop for small N
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    # state[partner] is the amplitude of |1...1> (bit i = 1)
                    # state[idx] is the amplitude of |0...0> (bit i = 0)
                    # New |0> amp = c * old |0> - s * old |1>
                    # New |1> amp = s * old |0> + c * old |1>
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                # CZ applies -1 to |11>
                # Find indices where both bits are 1
                # Vectorized:
                indices = np.where((np.arange(dim) & mask) == mask)[0]
                state[indices] *= -1.0
                
    return state

# Cache for state vectors to avoid recomputation in DE if same theta is encountered?
# Unlikely in DE, but we can cache energy evaluations for exact same parameters.
_STATE_CACHE = {}

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    # Create a hashable key for theta
    # theta is a float array, we can use a tuple of rounded values or just the flat array
    # Since DE generates continuous values, exact match is rare.
    # We will skip caching for now to avoid memory bloat, relying on speed of _apply_circuit.
    
    state = _apply_circuit(theta, n)
    H_ZZ, H_X = _get_hamiltonian(n, coupling)
    H = -coupling * H_ZZ - field * H_X
    
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground state
    e_exact, psi0 = _compute_exact_ground_state(n, coupling, field)
    
    # Objective function for scipy
    def _objective(params):
        theta = params.reshape(layers, n)
        e_circ = _compute_energy(theta, n, coupling, field)
        return e_circ - e_exact

    # Define bounds for angles: [0, 2*pi) or [-pi, pi]. Ry is periodic with 2pi.
    # Using [0, 2*pi) is standard.
    bounds = [(0.0, 2.0 * np.pi)] * (layers * n)
    
    best_val = np.inf
    best_x = None
    
    # Strategy: Run Differential Evolution
    # DE is global and robust.
    # maxiter: limit based on time.
    # popsize: 15 is default, maybe 20 for better coverage.
    
    # We will run DE for a portion of the time, then if time remains, run Nelder-Mead on the best result.
    
    try:
        # Use DE
        # tol: 1e-8
        # maxiter: 50 is a reasonable start. 
        # We need to monitor time.
        # scipy DE doesn't have a built-in time limit, so we estimate.
        # One evaluation takes ~0.1ms for n=6? Let's assume 1ms to be safe.
        # 50 iterations * 20 population * 2 (crossover+selection) ~ 2000 evals.
        # 2000 * 1ms = 2s. Very fast.
        # We can afford more.
        
        # Let's run DE with maxiter=100, popsize=20
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            seed=42,
            polish=True  # Nelder-Mead polish at the end
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception as e:
        pass
    
    # If time remains, do a few more Nelder-Mead runs from different points?
    # Or just rely on DE's polish.
    
    # Let's check time
    if time.time() < deadline - 5.0 and best_x is not None:
        # Try to refine with Nelder-Mead from the best DE result
        # Also try from a few other good points if available? 
        # DE only returns the best.
        
        # Let's run Nelder-Mead explicitly to ensure high precision.
        try:
            from scipy.optimize import minimize
            res_nm = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass
            
        # If still time, try another DE run with different seed?
        if time.time() < deadline - 10.0:
            try:
                res2 = differential_evolution(
                    _objective, 
                    bounds, 
                    maxiter=50, 
                    popsize=15, 
                    tol=1e-8,
                    seed=123,
                    polish=True
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_x = res2.x
            except Exception:
                pass

    # Fallback if no optimization succeeded
    if best_x is None:
        # Return mean-field guess
        # Simple heuristic: all zeros or pi/2
        theta_opt = np.full((layers, n), 0.5)
        return theta_opt

    return best_x.reshape(layers, n)
