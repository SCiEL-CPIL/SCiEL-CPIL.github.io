"""Rewritten program using an MPS-inspired initialization strategy.

Approach:
1. For each (n, coupling, field), compute the exact ground state via exact diagonalization.
2. Use a simplified variational ansatz consistent with the fixed circuit structure:
   The circuit is: |0>^n -> Ry(theta_0) on all -> CZ on adj pairs -> Ry(theta_1) on all -> CZ on adj pairs -> Ry(theta_2) on all.
   
   We approximate the optimal angles by:
   - Computing the exact ground state |psi_gs>.
   - For the final layer (layer 2, no CZ after), the state before the final Ry is some state |phi> such that applying Ry(theta_2) to each qubit gives |psi_gs>.
   - We can estimate theta_2 by looking at the single-qubit reduced density matrices of |psi_gs> and finding the rotation that maps a "natural" state to match the marginals.
   
   However, a more direct approach for initialization:
   - Use the fact that for the transverse-field Ising model, in the strong field limit, the ground state is close to |+>^n, and in the strong coupling limit, it's close to |0>^n or |1>^n.
   - We can use a simple heuristic: initialize all theta to values that interpolate between 0 and pi/2 based on the ratio field/coupling.
   
   A better approach: use the exact ground state to fit the angles via a simplified local optimization:
   - Start with a reasonable guess (e.g., all theta = pi/2 * (field/(field+coupling)) or similar).
   - Use the exact ground state to compute the "ideal" single-qubit rotations for the last layer.
   
   Since we must return a fixed array and cannot do iterative optimization within the 120s limit for multiple configurations, we'll use a pre-computed lookup or a fast analytical approximation.

   Practical approach:
   - Pre-compute the optimal angles for the specific test cases (n=4,6; coupling=1; fields=0.45,1,1.65) by running a quick optimization offline... but we can't do that at runtime.
   
   Instead, use a physics-based initialization:
   - The ground state of the transverse-field Ising model can be approximated by mean-field theory.
   - In mean-field, each spin feels an effective field. The optimal single-qubit rotation angle theta satisfies tan(theta) = effective_field_transverse / effective_field_longitudinal.
   - For the open chain, the boundary effects are significant, so we use a position-dependent initialization.

   Let's use a simpler, robust approach:
   - Initialize all theta to pi/2 (product |+> state) as a baseline.
   - For stronger coupling, reduce theta towards 0.
   - Use a smooth interpolation: theta_i = pi/2 * (1 - coupling/(coupling + field)) for bulk, with boundary corrections.

   Actually, let's use the exact ground state to do a one-shot fit:
   - For the last layer (no CZ after), the state after layer 2 is the circuit output.
   - If we assume the state before the last Ry is approximately |0>^n (which is true if the first two layers produce a state close to |0>), then theta_2 should match the single-qubit angles of the ground state.
   - We can extract the "effective" single-qubit angles from the ground state by computing the reduced density matrices and finding the rotation that diagonalizes them.

   Let's implement a practical version:
   1. Compute exact ground state |psi>.
   2. For each qubit i, compute the reduced density matrix rho_i.
   3. Find the rotation angle theta_i such that applying Ry(theta_i) to |0> gives a state whose density matrix matches rho_i as closely as possible.
      - This means: |0><0| -> Ry(theta_i) |0><0| Ry(theta_i)^dagger = (1/2)(I + theta_i_vec . sigma)
      - We want to match rho_i = (1/2)(I + r_i . sigma), so theta_i_vec should be proportional to r_i.
      - Since Ry(theta) rotates around y-axis, the resulting Bloch vector is (sin(theta), 0, cos(theta)) for |0> rotated by Ry(theta).
      - Wait: Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, so the Bloch vector is (sin(theta), 0, cos(theta)).
      - So we want (sin(theta_i), 0, cos(theta_i)) to match the x and z components of r_i.
      - theta_i = atan2(r_i_x, r_i_z).

   4. For the earlier layers, we can use a similar approach but it's more complex due to CZ gates.
      - As an initialization, set theta_0 and theta_1 to values that would produce a state close to |0>^n after the CZ gates, so that the last layer can then rotate to the ground state.
      - A simple heuristic: set theta_0 = theta_1 = 0 (no rotation), so the state remains |0>^n through the first two layers (CZ on |00> is |00>, so no change). Then theta_2 is set to the angles extracted from the ground state.

   This gives us a valid initialization. We can refine it slightly by noting that the CZ gates do nothing to |00>, so if we start with |0>^n and apply Ry(0) = I, the state stays |0>^n. Then the final Ry(theta_2) gives us a product state. But the ground state is not generally a product state, so this won't be optimal. However, it's a valid starting point.

   To do better, we can use the fact that the ground state has correlations. We can try to set theta_0 and theta_1 to small values that introduce some correlation.

   Given the time constraint, let's use the following:
   - theta_2[i] = atan2(r_i_x, r_i_z) where r_i is the Bloch vector of the i-th qubit in the ground state.
   - theta_0[i] = theta_1[i] = 0.

   This is a valid, physics-informed initialization that should be better than all pi/2.
"""

import numpy as np


def _compute_exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain."""
    dim = 2 ** n
    # Build Hamiltonian: H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    # Use Pauli matrices
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    H = np.zeros((dim, dim), dtype=complex)
    
    # Interaction terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        term = np.array([[1.0]])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I2)
        H -= coupling * term
    
    # Field terms: -field * X_i
    for i in range(n):
        term = np.array([[1.0]])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I2)
        H -= field * term
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0]
    return ground_state


def _extract_single_qubit_angles(psi, n):
    """Extract the optimal Ry angles for a product state approximation of psi."""
    dim = 2 ** n
    angles = np.zeros(n)
    for i in range(n):
        # Compute reduced density matrix for qubit i
        # Reshape psi to tensor shape (2, 2, ..., 2)
        psi_tensor = psi.reshape([2] * n)
        # Trace out all other qubits
        # Move qubit i to the front, then trace out the rest
        axes = list(range(n))
        # We want to keep qubit i and trace out others
        # Easier: use einsum or manual tracing
        # Let's do it by iterating
        rho_i = np.zeros((2, 2), dtype=complex)
        # Sum over all other qubits
        for other_vals in range(2 ** (n - 1)):
            # Build the index for the full state
            idx = [0] * n
            idx[i] = 0  # We'll handle i separately
            # Actually, let's use a different approach
            # Use np.tensordot or just build the reduced density matrix directly
            pass
        
        # Simpler approach: use the fact that rho_i = Tr_{all except i}(|psi><psi|)
        # We can compute this by reshaping and tracing
        psi_mat = psi.reshape((2 ** i, 2, 2 ** (n - i - 1)))
        rho_i = psi_mat @ psi_mat.conj().T
        # Wait, that's not quite right. Let me do it properly.
        
        # Proper way:
        # psi is a vector of length 2^n
        # We want rho_i[a,b] = sum_{all other indices} psi[a, other] * conj(psi[b, other])
        # where a,b are the values of qubit i
        
        # Reshape psi to [2, 2, ..., 2] with n dimensions
        psi_n = psi.reshape([2] * n)
        
        # For each value a of qubit i, and b of qubit i, sum over all other qubits
        rho_i = np.zeros((2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                # Create a slice where qubit i is fixed to a for psi and b for conj(psi)
                # This is tricky with numpy. Let's use einsum
                pass
        
        # Use einsum:
        # psi_n has indices [i0, i1, ..., i_{n-1}]
        # We want rho_i[a,b] = sum_{i0,...,i_{i-1},i_{i+1},...,i_{n-1}} psi_n[i0,...,a,...,i_{n-1}] * conj(psi_n[i0,...,b,...,i_{n-1}])
        
        # Let's construct the indices
        in_labels = [chr(97 + j) for j in range(n)]  # a, b, c, ...
        # For qubit i, we fix it to a for the first psi and b for the second
        # Actually, let's just do it with a loop since n is small (4 or 6)
        
        for a in range(2):
            for b in range(2):
                total = 0.0 + 0.0j
                # Iterate over all combinations of the other n-1 qubits
                for other_mask in range(2 ** (n - 1)):
                    # Build the full index
                    idx = [0] * n
                    # Set qubit i
                    # We need to map other_mask to the other qubits
                    # Let's build the index by inserting a at position i
                    idx_no_i = [0] * (n - 1)
                    for k in range(n - 1):
                        idx_no_i[k] = (other_mask >> k) & 1
                    # Now insert a at position i
                    idx[:i] = idx_no_i[:i]
                    idx[i] = a
                    idx[i:] = idx_no_i[i-1:] if i < n-1 else []
                    # This is getting complicated. Let's use a cleaner method.
                    pass
        
        # Cleanest method for small n:
        # Use np.einsum with explicit indices
        # For n=4: rho_i[a,b] = sum_{j,k,l} psi[j,i,k,l] * conj(psi[j,i',k,l]) where i is the fixed qubit
        # But we need to handle variable n.
        
        # Let's just use the matrix multiplication approach properly:
        # Reshape psi to a matrix of shape (2^i * 2^{n-i-1}, 2) where the last dimension is qubit i
        # Actually, reshape to (2^i, 2, 2^{n-i-1})
        left_dim = 2 ** i
        right_dim = 2 ** (n - i - 1)
        psi_reshaped = psi.reshape(left_dim, 2, right_dim)
        # Now rho_i = psi_reshaped @ psi_reshaped^dagger, but we need to sum over left and right
        # rho_i[a,b] = sum_{l,r} psi_reshaped[l,a,r] * conj(psi_reshaped[l,b,r])
        rho_i = np.einsum('lar,lsr->ab', psi_reshaped, psi_reshaped.conj())
        
        # Extract Bloch vector
        rx = np.real(rho_i[0, 1] + rho_i[1, 0])
        ry = np.imag(rho_i[1, 0] - rho_i[0, 1])
        rz = np.real(rho_i[0, 0] - rho_i[1, 1])
        
        # For Ry(theta)|0>, the Bloch vector is (sin(theta), 0, cos(theta))
        # So we want theta such that (sin(theta), cos(theta)) matches (rx, rz)
        # theta = atan2(rx, rz)
        angles[i] = np.arctan2(rx, rz)
    
    return angles


def angles(n, coupling, field, layers):
    """Return the circuit parameters as a (layers, n) array of angles."""
    # Compute exact ground state
    psi = _compute_exact_ground_state(n, coupling, field)
    
    # Extract optimal angles for the final layer
    theta_final = _extract_single_qubit_angles(psi, n)
    
    # For the earlier layers, use a heuristic:
    # Set them to small values that depend on the coupling/field ratio
    # In the strong field limit, we want |+> so theta ~ pi/2
    # In the strong coupling limit, we want |0> so theta ~ 0
    # Use a smooth interpolation
    ratio = field / (field + coupling)
    theta_bulk = np.pi / 2 * ratio
    
    # Initialize all layers with the bulk value, then override the final layer
    theta = np.tile(theta_bulk, (layers, 1))
    
    # Set the final layer to the extracted angles
    theta[-1] = theta_final
    
    # Ensure all values are finite
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
