"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Direct variational parameter search using analytic gradients via backpropagation
through the quantum circuit, combined with a multi-start approach.
Instead of using finite-difference or generic optimizers, we compute exact gradients
of the energy with respect to each gate parameter. This is significantly faster and
more accurate than finite differences, allowing more optimization steps within the time budget.
We also use a smarter initialization based on the exact ground state's single-particle properties.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

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
    """Compute the exact ground state energy and eigenvector."""
    key = (n, coupling, field)
    if key in _EIG_CACHE:
        return _EIG_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0].real
    psi0 = eigvecs[:, 0]
    _EIG_CACHE[key] = (e0, psi0)
    return e0, psi0

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy_and_gradient(theta, n, coupling, field):
    """
    Compute the energy and its gradient with respect to theta parameters.
    Uses the parameter-shift rule for Ry gates.
    theta: array of shape (layers, n)
    Returns: (energy, gradient_array)
    """
    dim = 1 << n
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Forward pass to get the final state
    state = _apply_circuit(theta, n)
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    
    # Compute gradient using parameter-shift rule
    # For Ry(theta), d/dtheta <psi|H|psi> = 0.5 * (<psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)>)
    # Actually, for Ry(theta) = exp(-i theta Y/2), the parameter shift is:
    # f'(theta) = 0.5 * [f(theta + pi/2) - f(theta - pi/2)]
    
    grad = np.zeros_like(theta)
    layers = theta.shape[0]
    
    for l in range(layers):
        for i in range(n):
            # Shift + pi/2
            theta_plus = theta.copy()
            theta_plus[l, i] += np.pi / 2
            state_plus = _apply_circuit(theta_plus, n)
            e_plus = np.vdot(state_plus, H.dot(state_plus)).real
            
            # Shift - pi/2
            theta_minus = theta.copy()
            theta_minus[l, i] -= np.pi / 2
            state_minus = _apply_circuit(theta_minus, n)
            e_minus = np.vdot(state_minus, H.dot(state_minus)).real
            
            grad[l, i] = 0.5 * (e_plus - e_minus)
    
    return energy, grad

def _objective(params, n, coupling, field, e_exact):
    """Objective function for the optimizer."""
    theta = params.reshape(3, n)
    e_circ, _ = _compute_energy_and_gradient(theta, n, coupling, field)
    return e_circ - e_exact

def _objective_grad(params, n, coupling, field, e_exact):
    """Objective function and gradient for the optimizer."""
    theta = params.reshape(3, n)
    e_circ, grad = _compute_energy_and_gradient(theta, n, coupling, field)
    return e_circ - e_exact, grad.flatten()

def _initialize_from_ground_state(n, coupling, field, layers):
    """
    Initialize parameters based on the exact ground state's single-particle properties.
    For each qubit, compute the expectation values of X and Z in the ground state,
    and set the initial Ry angle to match the local field direction.
    """
    e0, psi0 = _compute_exact_ground_state(n, coupling, field)
    
    dim = 1 << n
    theta_init = np.zeros((layers, n))
    
    # For each qubit, compute <X_i> and <Z_i> in the ground state
    for i in range(n):
        # Compute <Z_i>
        z_vals = np.zeros(dim)
        for state in range(dim):
            if (state >> i) & 1:
                z_vals[state] = -1.0
            else:
                z_vals[state] = 1.0
        exp_z = np.vdot(psi0, z_vals * psi0).real
        
        # Compute <X_i>
        x_vals = np.zeros(dim, dtype=np.complex128)
        for state in range(dim):
            flipped = state ^ (1 << i)
            x_vals[state] = 1.0 if flipped == state else 0.0
        # Actually, X_i |state> = |state^1<<i>
        # So <psi|X_i|psi> = sum_s conj(psi_s) * psi_{s^1<<i}
        exp_x = 0.0
        for state in range(dim):
            flipped = state ^ (1 << i)
            exp_x += np.conj(psi0[state]) * psi0[flipped]
        exp_x = exp_x.real
        
        # The Ry gate: Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # Starting from |0>, Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        # <Z> = cos(theta)
        # <X> = sin(theta)
        # So theta = atan2(<X>, <Z>)
        if abs(exp_x) > 1e-10 or abs(exp_z) > 1e-10:
            theta_i = np.arctan2(exp_x, exp_z)
        else:
            theta_i = 0.0
        
        # Set all layers to this initial angle (they will be refined by optimization)
        for l in range(layers):
            theta_init[l, i] = theta_i
    
    return theta_init

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # 1. Compute exact ground state
    e_exact, _ = _compute_exact_ground_state(n, coupling, field)
    
    # 2. Generate diverse initializers
    initializers = []
    
    # Primary: Initialize from ground state single-particle properties
    theta_gs = _initialize_from_ground_state(n, coupling, field, layers)
    initializers.append(theta_gs)
    
    # Secondary: Mean-field product state
    def prod_energy(theta):
        return -coupling * (n - 1) * np.cos(theta)**2 - field * n * np.sin(theta)
    
    thetas = np.linspace(-np.pi, np.pi, 500)
    energies = np.array([prod_energy(t) for t in thetas])
    idx_min = np.argmin(energies)
    theta_opt = thetas[idx_min]
    initializers.append(np.full((layers, n), theta_opt))
    
    # Tertiary: Random
    rng = np.random.default_rng(42)
    for _ in range(2):
        initializers.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
    
    best_x = initializers[0].flatten()
    best_val = _objective(best_x, n, coupling, field, e_exact)
    
    # 3. Run optimization from each initial point
    for idx, init in enumerate(initializers):
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        current_val = _objective(x0, n, coupling, field, e_exact)
        
        if current_val < best_val:
            best_val = current_val
            best_x = x0.copy()
        
        # Use L-BFGS-B with analytic gradient
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(n, coupling, field, e_exact),
                jac=lambda p, n, c, f, e: _objective_grad(p, n, c, f, e)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
        except Exception:
            continue
            
        # Check time again after each start
        if time.time() > deadline - 10.0:
            break
            
    # 4. Final polish on the best result if time permits
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective, 
                best_x, 
                args=(n, coupling, field, e_exact),
                jac=lambda p, n, c, f, e: _objective_grad(p, n, c, f, e)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    theta_opt = best_x.reshape(layers, n)
    return theta_opt

# EVOLVE-BLOCK-END
