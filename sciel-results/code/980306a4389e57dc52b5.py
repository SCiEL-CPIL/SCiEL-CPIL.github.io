import numpy as np
import time
from scipy.optimize import minimize

def _simulate_state(theta, n):
    """
    Simulate the circuit and return the state vector.
    theta: shape (layers, n)
    n: number of qubits
    """
    layers = theta.shape[0]
    # Start state |0...0>
    # Index 0 corresponds to |00...0>
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            t = theta[l, i]
            # Ry(t) = cos(t/2) * I - i * sin(t/2) * X
            c = np.cos(t/2)
            s = np.sin(t/2)
            # Apply to qubit i
            # We can do this by iterating over basis states or using tensor operations.
            # Given small n, iterating is fine, but tensor reshape is cleaner.
            # Reshape to [2]*n
            state = state.reshape([2]*n)
            # Apply Ry on axis i
            # Ry matrix: [[c, -1j*s], [-1j*s, c]]
            # new_state[i] = c * old_state[i] - 1j*s * old_state[i^1]
            # We need to swap the axis i to the front, apply, then swap back?
            # Or just use np.tensordot?
            # Simpler: for each basis state, update.
            # But vectorized is better.
            # Move axis i to 0
            state = np.moveaxis(state, i, 0)
            # Apply matrix
            # state[0] is the |0...> component, state[1] is |1...> component
            new_state_0 = c * state[0] - 1j * s * state[1]
            new_state_1 = -1j * s * state[0] + c * state[1]
            state = np.stack([new_state_0, new_state_1], axis=0)
            # Move axis back
            state = np.moveaxis(state, 0, i)
            state = state.reshape(2**n)
        
        # Apply CZ to adjacent pairs if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                # Apply CZ on qubits i and i+1
                # CZ is diagonal: 1 everywhere except |11> which is -1
                # We need to flip the sign of the amplitudes where qubit i is 1 AND qubit i+1 is 1.
                # Create a mask of indices where bits i and i+1 are 1.
                # Bit i is 1 if (index >> i) & 1 == 1
                # Bit i+1 is 1 if (index >> (i+1)) & 1 == 1
                # We can generate these indices or use a vectorized approach.
                # Vectorized approach:
                # Reshape state to [2]*n
                s = state.reshape([2]*n)
                # We want to multiply by -1 where s[1, ..., 1, ...] (at positions i, i+1)
                # Let's extract the sub-blocks.
                # This is slightly tricky with moveaxis.
                # Alternative: iterate over the 2^(n-2) combinations of other qubits.
                # Since n is small, let's just find the indices.
                indices = np.arange(2**n)
                # Mask for bit i
                mask_i = (indices >> i) & 1
                # Mask for bit i+1
                mask_ip1 = (indices >> (i+1)) & 1
                # Both are 1
                both_one = (mask_i == 1) & (mask_ip1 == 1)
                state[both_one] *= -1.0
                # state is already flattened, so this works directly.
                
    return state

def _energy(state, n, coupling, field):
    """
    Calculate the energy of the state for the Hamiltonian H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    """
    # Z_i Z_(i+1)
    # Z = diag(1, -1)
    # <Z_i Z_(i+1)> = sum_psi |psi|^2 * z_i * z_(i+1)
    # z_i is 1 if bit i is 0, -1 if bit i is 1.
    indices = np.arange(2**n)
    probs = np.abs(state)**2
    
    energy = 0.0
    # ZZ terms
    for i in range(n - 1):
        z_i = np.where((indices >> i) & 1 == 1, -1.0, 1.0)
        z_ip1 = np.where((indices >> (i+1)) & 1 == 1, -1.0, 1.0)
        zz_exp = np.sum(probs * z_i * z_ip1)
        energy -= coupling * zz_exp
        
    # X terms
    # <X_i> = sum_{p} psi_p^* psi_{p^i} + c.c. = 2 * Re(sum_{p, bit_i=0} psi_p^* psi_{p + 2^i})
    # Or simpler: X_i flips bit i.
    # <X_i> = sum_p psi_p^* (X_i psi)_p
    # (X_i psi)_p = psi_{p^i}
    # So <X_i> = sum_p psi_p^* psi_{p^i}
    # Note: X is Hermitian, so expectation is real.
    for i in range(n):
        # Flip bit i
        indices_flipped = indices ^ (1 << i)
        # psi_flipped[p] = psi[indices_flipped[p]]
        psi_flipped = state[indices_flipped]
        # <X_i> = sum psi^* * psi_flipped
        x_exp = np.sum(np.conjugate(state) * psi_flipped)
        # Should be real
        energy -= field * np.real(x_exp)
        
    return np.real(energy)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Objective function for optimizer
    def objective(theta_flat):
        theta = theta_flat.reshape((layers, n))
        state = _simulate_state(theta, n)
        e = _energy(state, n, coupling, field)
        return e
    
    # Initial guesses
    # 1. Zeros
    # 2. Random
    # 3. Heuristic: try to mimic |+> for X terms, but CZ creates entanglement.
    # Ground state of Ising in transverse field is a superposition.
    
    inits = []
    inits.append(np.zeros(layers * n))
    # Add a few random inits
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
    
    best_theta = inits[0]
    best_energy = objective(best_theta)
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success and res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x
        except Exception:
            pass

    # If we have time, do a polish pass with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                objective, 
                best_theta, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res2.success and res2.fun < best_energy:
                best_energy = res2.fun
                best_theta = res2.x
        except Exception:
            pass

    return best_theta.reshape((layers, n))
