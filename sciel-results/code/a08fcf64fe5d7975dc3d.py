def _build_fast(gates, qubits):
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    # Precompute eye
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            g = np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ], dtype=complex)
            
            # Apply U = (I x ... x G x ... x I) @ U
            # This is equivalent to:
            # For each row r and column c of U, the element is updated?
            # No, it's a matrix multiplication.
            # We can use np.einsum or reshape.
            # Reshape U to shape (2, 2, ..., 2, 2, 2, ..., 2)
            # Multiply along the axis corresponding to q.
            
            # This is complex to implement correctly for arbitrary q.
            # Let's stick to kron but optimize the loop.
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == q:
                    op = np.kron(op, g)
                else:
                    op = np.kron(op, eye)
            U = op @ U
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            U = U[index ^ (bit << (qubits - 1 - target)), :]
    return U
