"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval

    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            H[b, b_flip] += -field

    return H


def _exact_ground_energy(H):
    """Compute the exact ground state energy via exact diagonalization."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _circuit_state_vector(params, n, layers):
    """
    Compute the state vector produced by the circuit.
    params: shape (layers, n)
    Start from |0>^n, apply Ry(theta) to every qubit in each layer,
    then CZ on adjacent pairs (ascending order) between layers.
    No entangler after final layer.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # For each basis state, if qubit q is 0, multiply by c and move to 1 with s
            # We need to be careful: apply Ry to each qubit
            # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
            # Ry(theta) |1> = -sin(theta/2)|0> + cos(theta/2)|1>
            # So for each basis index b, let bit = (b >> q) & 1
            # If bit == 0: new amplitude gets c * old[b] + s * old[b ^ (1<<q)]
            # Wait, that's not right. Let me think again.
            # Ry acts on qubit q. For a basis state |b>, the qubit q has value (b>>q)&1.
            # The transformation is:
            #   |0> -> c|0> + s|1>
            #   |1> -> -s|0> + c|1>
            # So if we have amplitude a_b for state |b>, after Ry on qubit q:
            #   new amplitude for state where qubit q is 0: 
            #     comes from old states where qubit q was 0 (contributing c * a_b) 
            #     and old states where qubit q was 1 (contributing -s * a_b)
            #   new amplitude for state where qubit q is 1:
            #     comes from old states where qubit q was 0 (contributing s * a_b)
            #     and old states where qubit q was 1 (contributing c * a_b)
            # 
            # Let's do it by iterating over all basis states and building new state.
            new_state = np.zeros(dim, dtype=np.complex128)
            for b in range(dim):
                bit = (b >> q) & 1
                if bit == 0:
                    # This state has qubit q = 0
                    # Contribution to new_state[b] (qubit q still 0): c * state[b]
                    # Contribution to new_state[b ^ (1<<q)] (qubit q becomes 1): s * state[b]
                    new_state[b] += c * state[b]
                    new_state[b ^ (1 << q)] += s * state[b]
                else:
                    # This state has qubit q = 1
                    # Contribution to new_state[b] (qubit q still 1): c * state[b]
                    # Contribution to new_state[b ^ (1<<q)] (qubit q becomes 0): -s * state[b]
                    new_state[b] += c * state[b]
                    new_state[b ^ (1 << q)] += -s * state[b]
            state = new_state

        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ: flips phase of |11> for qubits q and q+1
                for b in range(dim):
                    bit_q = (b >> q) & 1
                    bit_qp1 = (b >> (q + 1)) & 1
                    if bit_q == 1 and bit_qp1 == 1:
                        state[b] = -state[b]

    return state


def _circuit_energy_and_grad(params, n, layers, H, e_ground):
    """
    Compute the energy and gradient of the circuit energy with respect to params.
    Energy = <psi|H|psi> - e_ground, divided by n.
    Gradient computed via parameter shift rule for Ry gates.
    """
    dim = 1 << n
    state = _circuit_state_vector(params, n, layers)
    e_circuit = np.real(np.vdot(state, H @ state))
    energy = (e_circuit - e_ground) / n

    # Compute gradient using parameter shift rule
    # For Ry(theta), the parameter shift rule gives:
    #   d/dtheta <psi|H|psi> = (1/2) [ <psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)> ]
    # But this is expensive. Instead, let's use automatic differentiation or finite differences.
    # For n=4,6 and layers=3, we have 12 or 18 parameters. Finite differences with 2*12=24 or 2*18=36 evaluations
    # might be acceptable given the time budget.
    
    # Actually, let's use a more efficient approach: compute the gradient analytically
    # using the chain rule through the circuit.
    
    # For now, use finite differences for simplicity and correctness.
    grad = np.zeros_like(params)
    eps = 1e-7
    for layer in range(layers):
        for q in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[layer, q] += eps
            params_minus[layer, q] -= eps
            
            state_plus = _circuit_state_vector(params_plus, n, layers)
            state_minus = _circuit_state_vector(params_minus, n, layers)
            
            e_plus = np.real(np.vdot(state_plus, H @ state_plus))
            e_minus = np.real(np.vdot(state_minus, H @ state_minus))
            
            grad[layer, q] = (e_plus - e_minus) / (2 * eps * n)

    return energy, grad


def _objective(params_flat, n, layers, H, e_ground):
    """Objective function for scipy.optimize."""
    params = params_flat.reshape((layers, n))
    energy, _ = _circuit_energy_and_grad(params, n, layers, H, e_ground)
    return energy


def _objective_grad(params_flat, n, layers, H, e_ground):
    """Objective function and gradient for scipy.optimize."""
    params = params_flat.reshape((layers, n))
    energy, grad = _circuit_energy_and_grad(params, n, layers, H, e_ground)
    return energy, grad.flatten()


def _physics_init(n, field):
    """
    Generate a physics-based initial guess for the parameters.
    For the transverse-field Ising model, the ground state has a characteristic
    structure. We use a simple heuristic based on the field strength.
    """
    # For small field, the ground state is close to |0...0> or |1...1>
    # For large field, it's more delocalized
    # A reasonable initial guess: angles that rotate towards the expected ground state
    # 
    # For the single-qubit case, H = -field * X, ground state is (|0> + |1>)/sqrt(2)
    # which corresponds to Ry(pi/2)|0> = (|0> + |1>)/sqrt(2)
    # 
    # For the multi-qubit case with ZZ coupling, the structure is more complex.
    # Let's use a simple heuristic: start with angles proportional to field strength
    # and add some structure based on the coupling.
    
    params = np.zeros((3, n))
    
    # Layer 0: small rotations to start building correlations
    for q in range(n):
        params[0, q] = 0.2 * field
    
    # Layer 1: larger rotations
    for q in range(n):
        params[1, q] = 0.5 * field
    
    # Layer 2: final rotations
    for q in range(n):
        params[2, q] = 0.3 * field
    
    return params


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute Hamiltonian and ground energy
    H = _build_hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(H)

    # Generate multiple initial guesses
    np.random.seed(42)
    inits = []
    
    # Physics-based init
    inits.append(_physics_init(n, field))
    
    # All zeros
    inits.append(np.zeros((layers, n)))
    
    # Uniform angles
    inits.append(np.full((layers, n), np.pi / 4.0))
    inits.append(np.full((layers, n), -np.pi / 4.0))
    inits.append(np.full((layers, n), np.pi / 8.0))
    inits.append(np.full((layers, n), -np.pi / 8.0))
    
    # Random inits
    for _ in range(8):
        inits.append(np.random.uniform(-np.pi/2, np.pi/2, (layers, n)))

    best_params = inits[0].copy()
    best_val = _objective(best_params.flatten(), n, layers, H, e_ground)

    # Phase 1: Multi-restart with L-BFGS-B using gradient
    for restart, x_init in enumerate(inits):
        if time.time() > deadline - 5:
            break

        x_flat = x_init.flatten()

        def obj_flat(xf):
            return _objective(xf, n, layers, H, e_ground)

        def obj_grad_flat(xf):
            return _objective_grad(xf, n, layers, H, e_ground)

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_flat, method='L-BFGS-B',
                jac=obj_grad_flat,
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 500}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 2: Fine-tune with Nelder-Mead (no gradient, more robust)
    if time.time() < deadline - 3:
        x_best_flat = best_params.flatten()

        def obj_flat(xf):
            return _objective(xf, n, layers, H, e_ground)

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_best_flat, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 3: Final high-precision polish with L-BFGS-B
    if time.time() < deadline - 1:
        x_best_flat = best_params.flatten()

        def obj_flat(xf):
            return _objective(xf, n, layers, H, e_ground)

        def obj_grad_flat(xf):
            return _objective_grad(xf, n, layers, H, e_ground)

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_best_flat, method='L-BFGS-B',
                jac=obj_grad_flat,
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 1000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END
