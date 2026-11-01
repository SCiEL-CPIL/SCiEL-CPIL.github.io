# Try deterministic assignments
for c in range(min(m, 20)):
    voltages = np.array([(i * c + 1) % m for i in range(n_base_edges)], dtype=np.int32)
    sums = constraint_matrix @ voltages
    if not np.any((sums % m) == 0):
        return voltages
