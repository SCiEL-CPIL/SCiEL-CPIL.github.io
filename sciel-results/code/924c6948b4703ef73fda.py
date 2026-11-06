import numpy as np
import scipy.optimize as opt

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    Uses mean-field inspired initialization and multi-start L-BFGS-B.
    """
    dim = 2 ** n
    
    # Construct Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            val = z_i * z_ip1
            H[b, b] += -coupling * val
    
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_prime = b ^ (1 << i)
            H[b, b_prime] += -field
    
    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E_exact = eigvals[0].real
    
    # Efficient circuit application using tensor operations
    def apply_circuit(theta, state):
        """
        theta: (layers, n) array
        state: complex vector of size dim
        Returns final state after applying circuit.
        """
        state = state.copy()
        
        for l in range(layers):
            # Apply Ry(theta[l, i]) to qubit i for all i
            # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            # Qubit i corresponds to axis (n-1-i) in the tensor representation
            for i in range(n):
                theta_i = theta[l, i]
                c = np.cos(theta_i / 2)
                s = np.sin(theta_i / 2)
                
                # Reshape state to tensor
                state_t = state.reshape((2,) * n)
                axis = n - 1 - i
                
                # Move target axis to front
                state_t = np.moveaxis(state_t, axis, 0)
                
                # Apply Ry: new[0,:] = c*old[0,:] - s*old[1,:]
                #          new[1,:] = s*old[0,:] + c*old[1,:]
                new_state_t = np.empty_like(state_t)
                new_state_t[0] = c * state_t[0] - s * state_t[1]
                new_state_t[1] = s * state_t[0] + c * state_t[1]
                
                state = new_state_t.reshape(dim)
            
            # Apply CZ between layers (not after final layer)
            if l < layers - 1:
                for i in range(n - 1):
                    # CZ applies -1 phase when both qubits i and i+1 are |1>
                    for b in range(dim):
                        bit_i = (b >> i) & 1
                        bit_ip1 = (b >> (i + 1)) & 1
                        if bit_i == 1 and bit_ip1 == 1:
                            state[b] *= -1.0
        
        return state
    
    def energy_from_theta(theta_flat):
        """Compute energy for given flattened theta parameters."""
        theta = theta_flat.reshape(layers, n)
        state0 = np.zeros(dim, dtype=complex)
        state0[0] = 1.0
        state = apply_circuit(theta, state0)
        energy = np.vdot(state, H @ state).real
        return energy
    
    # Mean-field inspired initialization
    # For transverse field Ising model, in the paramagnetic phase (large field),
    # spins are roughly aligned with +Z direction, so Ry angles near pi/2
    # In the ferromagnetic phase (small field), more complex structure
    #
    # Simple heuristic: start with angles that produce states close to
    # the classical ground state configuration
    
    def meanfield_init(n, coupling, field):
        """Generate initial theta based on mean-field intuition."""
        theta = np.zeros((layers, n))
        
        # Effective transverse field vs longitudinal field
        # If field is large, system is paramagnetic: spins point along +Z
        # Ry(pi/2) on |0> gives |+> = (|0>+|1>)/sqrt(2)
        # We want to bias toward |0> for ferromagnetic, |+> for paramagnetic
        
        ratio = field / max(coupling, 1e-10)
        
        if ratio > 1.0:
            # Paramagnetic regime: angles near pi/2
            base_angle = np.pi / 2
        elif ratio > 0.5:
            # Intermediate: angles between 0 and pi/2
            base_angle = np.pi / 2 * (1.0 - 0.5 * ratio)
        else:
            # Ferromagnetic regime: angles near 0 (keep |0>)
            base_angle = np.pi / 2 * ratio
        
        # Apply with some variation across layers
        for l in range(layers):
            for i in range(n):
                # Add small position-dependent variation
                pos_factor = 1.0 - 0.1 * abs(i - (n - 1) / 2.0) / max((n - 1) / 2.0, 1)
                theta[l, i] = base_angle * pos_factor
        
        return theta
    
    # Generate multiple initial guesses
    inits = []
    
    # 1. Mean-field initialization
    inits.append(meanfield_init(n, coupling, field))
    
    # 2. All pi/2 (paramagnetic guess)
    theta_pi2 = np.full((layers, n), np.pi / 2)
    inits.append(theta_pi2)
    
    # 3. All zeros (ferromagnetic guess)
    inits.append(np.zeros((layers, n)))
    
    # 4. Random inits
    rng = np.random.default_rng(42)
    for _ in range(4):
        inits.append(rng.uniform(0, 2 * np.pi, size=(layers, n)))
    
    # 5. Physics-inspired: alternate angles
    theta_alt = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n):
            if i % 2 == 0:
                theta_alt[l, i] = np.pi / 3
            else:
                theta_alt[l, i] = 2 * np.pi / 3
    inits.append(theta_alt)
    
    # Optimize from each initial guess, keep best
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    best_energy = np.inf
    best_theta = None
    
    for init in inits:
        try:
            res = opt.minimize(
                energy_from_theta,
                init.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            e = res.fun
            if e < best_energy:
                best_energy = e
                best_theta = res.x.reshape(layers, n)
        except Exception:
            continue
    
    # Fallback: if all optimizations failed, return mean-field init
    if best_theta is None:
        best_theta = meanfield_init(n, coupling, field)
    
    return best_theta
