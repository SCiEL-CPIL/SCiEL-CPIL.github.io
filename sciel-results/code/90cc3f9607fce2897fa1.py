import numpy as np
import scipy.optimize as opt
import time

def _apply_single_qubit_gate(state, qubit, gate, n):
    """
    Apply a 2x2 gate to a specific qubit in the state vector.
    state: 1D array of size 2^n
    qubit: index of qubit (0 to n-1)
    gate: 2x2 matrix
    n: number of qubits
    """
    # Reshape state to tensor of shape (2,)*n
    # Note: qubit i corresponds to bit i. 
    # In standard QIS, qubit 0 is usually the least significant bit or most?
    # The task says "qubit i is bit i". Usually bit 0 is LSB.
    # Let's assume standard numpy reshape where axis 0 is the most significant bit?
    # Or we can just use indices.
    # Let's use the tensor approach.
    # If state is indexed by integer k, k = sum b_i 2^i.
    # Numpy reshape: state.reshape([2]*n). The first axis corresponds to the most significant bit (2^(n-1)).
    # So axis 0 is qubit n-1, axis n-1 is qubit 0.
    # Let's verify. 
    # Index 1 (0b01) -> qubit 0 is 1.
    # In reshape([2,2] for n=2), index 1 is [0,1]. So axis 1 is qubit 0.
    # So axis i in the tensor corresponds to qubit (n-1-i)? No.
    # Let's trace:
    # n=2. State size 4.
    # |00> = 0. |01> = 1. |10> = 2. |11> = 3.
    # Reshape to (2,2).
    # [0, 1] -> |00>, |01>. So second index is qubit 0. First index is qubit 1.
    # So axis k corresponds to qubit (n-1-k).
    
    tensor = state.reshape([2]*n)
    # We want to apply gate to qubit `qubit`.
    # The axis corresponding to qubit `qubit` is `n - 1 - qubit`.
    axis = n - 1 - qubit
    
    # Move the axis to the last position for easy matrix multiplication
    # Or just use np.tensordot
    # Let's move axis to 0, apply, move back? 
    # Easier: use np.einsum or tensordot.
    
    # Move the target axis to the front
    tensor = np.moveaxis(tensor, axis, 0)
    
    # Apply gate: gate @ tensor[0] + ... ? No.
    # The gate acts on the subspace of that qubit.
    # tensor is now shape (2, 2, ..., 2).
    # We want to compute sum_j gate[i,j] * tensor[j, ...]
    # This is a matrix multiplication along the first axis.
    new_tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    # new_tensor shape is (2, 2, ..., 2) but the first axis is the new qubit state.
    # We need to move it back to the original position.
    new_tensor = np.moveaxis(new_tensor, 0, axis)
    
    return new_tensor.reshape(-1)

def _apply_two_qubit_gate(state, q1, q2, gate, n):
    """
    Apply a 4x4 gate to two specific qubits.
    q1, q2: indices of qubits.
    gate: 4x4 matrix.
    """
    tensor = state.reshape([2]*n)
    
    # Determine axes
    axis1 = n - 1 - q1
    axis2 = n - 1 - q2
    
    # We need to combine these two axes.
    # Let's move axis1 to 0 and axis2 to 1.
    # This is tricky if they are not adjacent.
    # Alternative: Use explicit indices. For n<=6, 2^n is small.
    # But let's try to be general.
    
    # A simpler way for small n: iterate over all basis states? No, that's slow in Python.
    # Let's use the tensor approach carefully.
    
    # Move axis1 to 0
    tensor = np.moveaxis(tensor, axis1, 0)
    # Move axis2 (which might have shifted) to 1
    # If axis2 > axis1, its index decreases by 1 after moving axis1 to 0.
    # If axis2 < axis1, its index stays same?
    # Let's just re-calculate the position.
    # Original axes: 0, 1, ..., n-1.
    # We moved axis1 to 0.
    # The remaining axes are sorted.
    # Let's find the new position of axis2.
    
    # Actually, it's easier to just permute the axes so that q1 and q2 are at the front.
    # Order of axes: [q1, q2, others...]
    
    # Get the list of axes
    axes = list(range(n))
    # Remove q1 and q2 axes
    other_axes = [a for a in axes if a != axis1 and a != axis2]
    new_order = [axis1, axis2] + other_axes
    
    tensor = np.transpose(tensor, new_order)
    
    # Now tensor is shape (2, 2, 2^(n-2))
    # Reshape to (4, 2^(n-2))
    # The 4x4 gate acts on the first 2 axes.
    # Basis ordering for the 4x4 gate:
    # Standard CZ is diag(1,1,1,-1) in basis |00>, |01>, |10>, |11>.
    # This corresponds to q1 being the most significant bit of the pair?
    # In the tensor [q1, q2, ...], index [i, j] corresponds to q1=i, q2=j.
    # The combined index is i*2 + j.
    # This matches standard binary representation if q1 is MSB and q2 is LSB of the pair.
    
    tensor_2d = tensor.reshape(4, -1)
    
    # Apply gate
    new_tensor_2d = gate @ tensor_2d
    
    # Reshape back
    new_tensor = new_tensor_2d.reshape([2]*n)
    
    # Inverse transpose
    inv_order = np.argsort(new_order)
    new_tensor = np.transpose(new_tensor, inv_order)
    
    return new_tensor.reshape(-1)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Compute Exact Ground State Energy
    # Build Hamiltonian H
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z_i Z_{i+1} is diagonal
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            # Z eigenvalue is 1 if bit is 0, -1 if bit is 1
            val_i = 1 if bit_i == 0 else -1
            val_ip1 = 1 if bit_ip1 == 0 else -1
            zz_val = val_i * val_ip1
            H[state, state] += -coupling * zz_val
            
    # X_i terms
    for i in range(n):
        for state in range(dim):
            # X flips bit i
            new_state = state ^ (1 << i)
            H[state, new_state] += -field
            
    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    E_gs = np.real(eigvals[0])
    
    # 2. Define Objective Function
    def objective(params):
        theta = params.reshape(layers, n)
        
        # Initialize state |0...0>
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        # Apply layers
        for l in range(layers):
            # Apply Ry to all qubits
            for i in range(n):
                th = theta[l, i]
                ry = np.array([
                    [np.cos(th/2), -np.sin(th/2)],
                    [np.sin(th/2), np.cos(th/2)]
                ], dtype=complex)
                state = _apply_single_qubit_gate(state, i, ry, n)
                
            # Apply CZ if not last layer
            if l < layers - 1:
                for i in range(n - 1):
                    cz = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]
                    ], dtype=complex)
                    state = _apply_two_qubit_gate(state, i, i+1, cz, n)
                    
        # Compute Energy <psi|H|psi>
        # E = <psi| H |psi>
        # H is Hermitian
        # We can compute H @ state then inner product
        H_state = H @ state
        energy = np.real(np.vdot(state, H_state))
        
        return energy - E_gs
    
    # 3. Multi-start Optimization
    best_params = None
    best_val = np.inf
    
    n_params = layers * n
    
    # Number of restarts
    n_restarts = 50
    
    # Random seed for reproducibility in evaluation? 
    # The task doesn't specify, but using a fixed seed is safer for consistent behavior.
    # However, different random starts might be needed to escape local minima.
    # Let's use a fixed seed for the first few, then random?
    # Or just fixed seed. The problem is small, likely few local minima.
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initial guess
        if r == 0:
            # Start with zeros? Or pi/2?
            x0 = np.zeros(n_params)
        else:
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            
        # Optimize
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        best_params = np.zeros(n_params)
        
    return best_params.reshape(layers, n)
