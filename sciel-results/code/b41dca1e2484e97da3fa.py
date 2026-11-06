import numpy as np
import scipy.optimize as opt

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    """
    dim = 2 ** n
    
    # Construct Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # We use a sparse-like construction or just dense since n is small.
    # For n=6, dim=64. Dense is very fast.
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[b, b] += -coupling * z_i * z_ip1
            
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_prime = b ^ (1 << i)
            H[b, b_prime] += -field

    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E_exact = eigvals[0].real
    
    def apply_circuit(theta, state):
        """Apply the full circuit to state vector."""
        for l in range(layers):
            # Apply Ry(theta[l, i]) to each qubit i
            for i in range(n):
                th = theta[l, i]
                c = np.cos(th / 2)
                s = np.sin(th / 2)
                # Apply Ry to qubit i
                state_tensor = state.reshape((2,) * n)
                axis = n - 1 - i
                state_tensor = np.swapaxes(state_tensor, axis, 0)
                new_tensor = np.empty_like(state_tensor)
                new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
                new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
                state_tensor = new_tensor
                state_tensor = np.swapaxes(state_tensor, 0, axis)
                state = state_tensor.reshape(dim)
            
            # Apply CZ between layers (not after final layer)
            if l < layers - 1:
                for i in range(n - 1):
                    # CZ: phase -1 when both qubits i and i+1 are 1
                    mask = np.zeros(dim, dtype=bool)
                    for b in range(dim):
                        if ((b >> i) & 1) == 1 and ((b >> (i + 1)) & 1) == 1:
                            mask[b] = True
                    state[mask] *= -1
        
        return state

    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        state = apply_circuit(theta, state)
        energy = np.vdot(state, H @ state).real
        return energy

    # Global Optimization using Differential Evolution
    # This is robust against local minima which L-BFGS-B struggles with in multi-layer circuits.
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    try:
        # DE is stochastic, so we set a seed for reproducibility if needed, but default is fine.
        # maxiter=100 and popsize=15 is a good balance for n=4,6 (params=12,18).
        res_de = opt.differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            seed=42
        )
        
        best_theta = res_de.x.reshape(layers, n)
        
        # Polish with L-BFGS-B to get high precision
        res_polish = opt.minimize(
            objective, 
            best_theta.ravel(), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res_polish.fun < res_de.fun:
            best_theta = res_polish.x.reshape(layers, n)
            
    except Exception:
        # Fallback to a simple heuristic if DE fails or times out
        best_theta = np.full((layers, n), np.pi / 4)

    return best_theta
