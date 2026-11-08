perm = np.zeros(16, dtype=int)
for I in range(2):
    for J in range(2):
        for i2 in range(2):
            for j2 in range(2):
                r_kron = I*8 + J*4 + i2*2 + j2
                r_std = I*8 + i2*4 + J*2 + j2
                perm[r_std] = r_kron
