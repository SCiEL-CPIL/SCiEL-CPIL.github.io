            full = np.array([[1]], dtype=complex)
            for k in range(qubits - 1, -1, -1):
                if k == q:
                    full = np.kron(full, op)
                else:
                    full = np.kron(full, np.eye(2, dtype=complex))
            U = full @ U
