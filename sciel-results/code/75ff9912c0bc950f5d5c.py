    toffoli_3 = np.eye(8, dtype=complex)
    toffoli_3[6,6] = 0
    toffoli_3[6,7] = 1
    toffoli_3[7,6] = 1
    toffoli_3[7,7] = 0
    # Standard Toffoli circuit
    toffoli_circuit = [
        ['u3', 2, math.pi/2, 0, 0],
        ['cx', 1, 2],
        ['u3', 2, math.pi/4, 0, 0],
        ['cx', 0, 2],
        ['u3', 2, -math.pi/4, 0, 0],
        ['cx', 1, 2],
        ['u3', 2, math.pi/4, 0, 0],
        ['cx', 0, 2],
        ['u3', 2, -math.pi/4, 0, 0],
        ['cx', 1, 2],
        ['u3', 2, math.pi/2, 0, 0]
    ]
