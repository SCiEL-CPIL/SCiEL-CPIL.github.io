"""
Optimize shallow circuit parameters for Ising ground states using global Nelder-Mead search.
Strategy:
1. Construct the Hamiltonian H for the open transverse-field Ising chain.
2. Compute the exact ground energy E_0 via exact diagonalization.
3. Implement a vectorized state-vector simulator for the specific 3-layer Ry/CZ circuit.
   This avoids Python loops over basis states, significantly speeding up evaluation.
4. Use scipy.optimize.minimize with the Nelder-Mead method (derivative-free global/local hybrid).
5. Perform multiple restarts with different initial parameters to escape local minima.
6. Manage time budget strictly to ensure completion within limits.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _simulate_energy(params, n, layers, H):
    """
    Vectorized simulation of the circuit to compute energy.
    params: shape (layers, n)
    H: pre-computed Hamiltonian matrix
    Returns: energy value
    """
    dim = 1 << n
    # Start state |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Pre-compute basis indices for vectorization
    # For each qubit q, we need indices where bit q is 0 and where bit q is 1.
    # Let's pre-compute these masks/indices once outside the loop if possible, 
    # but here we can just use numpy operations on the state array.
    
    # To speed up Ry application, we can reshape the state into a tensor of shape (2,)*n
    # However, applying Ry on a specific qubit is easy with reshape and transpose.
    
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Reshape state to tensor: shape (2, 2, ..., 2)
            # Qubit q is at axis q.
            # We want to apply [[c, -s], [s, c]] to axis q.
            # state_tensor shape: (2,)*n
            # Move axis q to the end for easier manipulation? Or just use np.tensordot?
            # Let's use a trick:
            # Split state into two halves based on bit q.
            # This is tricky with pure numpy indexing without loops if n varies.
            # Alternative: use matrix multiplication on reshaped blocks.
            
            # Efficient way:
            # Permute axes so q is last.
            axes = [ax for ax in range(n) if ax != q] + [q]
            state_perm = np.transpose(state, axes)
            # Now state_perm has shape (2, 2, ..., 2, 2) where last axis is q.
            # Reshape to (2^(n-1), 2)
            state_mat = state_perm.reshape(-1, 2)
            
            # Ry matrix
            Ry = np.array([[c, -s], [s, c]], dtype=complex)
            
            # Apply
            new_state_mat = state_mat @ Ry.T # Note: state_mat is (N, 2), Ry is (2, 2).
            # Wait, if state is a column vector, and we apply U, it's U @ state.
            # If state_mat is (N, 2), each row is a vector [amp_0, amp_1].
            # New vector = Ry @ [amp_0, amp_1].
            # So new_state_mat[i, :] = Ry @ state_mat[i, :]
            # This is (Ry.T @ state_mat.T).T
            new_state_mat = (Ry @ state_mat.T).T
            
            # Reshape back
            new_state_perm = new_state_mat.reshape((2,) * n)
            # Move axis back to position q
            inv_axes = [0] * n
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            state = np.transpose(new_state_perm, inv_axes)
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for q in range(n - 1):
                # CZ on q, q+1
                # CZ is diagonal: -1 if both bits are 1, else 1.
                # We can vectorize this by creating a mask.
                mask_q = 1 << q
                mask_q1 = 1 << (q + 1)
                
                # Indices where both bits are 1
                # This is hard to vectorize without a loop over basis states or complex masking.
                # Given dim is small (max 64), a simple loop over basis states is actually very fast in Python if we avoid numpy overhead per element?
                # No, numpy vectorized ops are better.
                
                # Let's stick to the loop for CZ as it's simpler and dim is small.
                # But we can make it faster:
                # state[b] *= -1 if (b & mask_q) and (b & mask_q1)
                
                # Vectorized CZ:
                # Create an array of basis states 0..dim-1
                # But we don't have it precomputed here.
                # Let's just use the loop for CZ, it's only (n-1)*2 * dim operations.
                for b in range(dim):
                    if (b & mask_q) and (b & mask_q1):
                        state[b] = -state[b]
                        
    # Compute energy
    Hpsi = H @ state
    energy = np.vdot(state, Hpsi).real
    return energy

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit (5s buffer)
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _ground_energy(n, coupling, field)
    
    def objective(params_flat):
        params = params_flat.reshape(layers, n)
        return _simulate_energy(params, n, layers, H)
        
    best_energy = np.inf
    best_params = np.zeros((layers, n))
    
    # Multi-restart Nelder-Mead
    num_restarts = 50
    # We will run as many restarts as time permits.
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Uniform in [-pi, pi]
        params_init = np.random.uniform(-np.pi, np.pi, (layers, n))
        
        # Nelder-Mead options
        # maxfev is crucial to prevent hanging
        # We want to be fast.
        options = {
            'maxfev': 200,
            'xatol': 1e-4,
            'fatol': 1e-5,
            'adaptive': True
        }
        
        try:
            res = minimize(objective, params_init.flatten(), method='Nelder-Mead', options=options)
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.reshape(layers, n)
                
        except Exception as e:
            # If optimization fails, try the initial point?
            try:
                e_init = objective(params_init.flatten())
                if e_init < best_energy:
                    best_energy = e_init
                    best_params = params_init
            except:
                pass

    # If no valid result found (shouldn't happen), fallback
    if best_energy == np.inf:
        # Simple heuristic fallback
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
        
    return best_params
# EVOLVE-BLOCK-END
